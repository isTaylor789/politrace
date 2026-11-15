"""Dinámica temporal de tópicos: agregación por periodo y ventanas con alineación.

Dos formas de mirar la evolución, que responden a preguntas distintas:

**Vía A — un modelo, θ agregada por periodo.** Se ajusta un único modelo a todo
el corpus y cada documento recibe su θ. Después se promedia θ por hablante-año
(o hablante-trimestre). Ventaja: los tópicos son los mismos en todos los
periodos, así que la serie es comparable. Límite: un tópico que aparece después
del año 2024 o que cambia de vocabulario no puede manifestarse.

**Vía B — un modelo por ventana + alineación.** Se ajusta un modelo por
hablante-año (con ≥ `WINDOW_MIN_DOCS` discursos) y se emparejan los tópicos de
ventanas consecutivas por coseno de sus vectores de términos, con asignación
óptima húngara (`linear_sum_assignment`). Ahí sí se ven tópicos que nacen, se
fusionan o desaparecen, y se mide **deriva** (similitud entre el mismo tópico en
dos años). Es más caro y menos estable en ventanas pequeñas: se reporta como
contraste, no como cifra principal.

La regla del plan base se respeta: la agregación es **siempre por hablante**
(el corpus está desbalanceado 2,294 : 30) y la comparación *entre* líderes se
limita a 2023-2024.
"""

import collections

import numpy as np

import mt_scripts.config as config
from mt_scripts import models as MO
from mt_scripts.matrices import build_subspace


# --- Periodos ----------------------------------------------------------------
def period_of(date: str, by: str = "year") -> str:
    """``YYYY`` o ``YYYY-Qn`` a partir de una fecha ISO."""
    year = str(date)[:4]
    if by == "year":
        return year
    month = int(str(date)[5:7] or 1)
    return f"{year}-Q{(month - 1) // 3 + 1}"


def doc_topics_table(ix, model) -> list:
    """Una fila por documento con θ, tópico dominante y metadatos."""
    rows = []
    dominant = model.dominant_topic()
    for i, doc_id in enumerate(model.space.doc_ids):
        doc_id = int(doc_id)
        theta = model.doc_topic[i]
        rows.append({
            "doc_id": doc_id,
            "speech_id": ix.doc_ids[doc_id],
            "speaker": ix.speakers[doc_id],
            "date": ix.dates[doc_id],
            "year": period_of(ix.dates[doc_id]),
            "quarter": period_of(ix.dates[doc_id], "quarter"),
            "topic": int(dominant[i]),
            "theta": [round(float(x), 5) for x in theta],
            "max_theta": round(float(theta[dominant[i]]), 5),
        })
    return rows


# --- Vía A: agregación de θ --------------------------------------------------
def shares_by(ix, model, by: str = "year") -> list:
    """Participación media de cada tópico por (hablante, periodo).

    La media de θ sobre los documentos del grupo es una participación: como cada
    fila de θ suma 1, la suma sobre tópicos de un grupo también suma 1. Así la
    comparación entre hablantes es proporcional y no está dominada por el que
    tiene 2,294 discursos.
    """
    groups = collections.defaultdict(list)
    for i, doc_id in enumerate(model.space.doc_ids):
        doc_id = int(doc_id)
        groups[(ix.speakers[doc_id], period_of(ix.dates[doc_id], by))].append(i)

    rows = []
    for (speaker, period), idx in sorted(groups.items()):
        mean_theta = model.doc_topic[idx].mean(axis=0)
        for topic, share in enumerate(mean_theta):
            rows.append({
                "speaker": speaker,
                "period": period,
                "topic": topic,
                "share": float(share),
                "docs": len(idx),
            })
    return rows


def speaker_shares(ix, model) -> list:
    """Participación media de cada tópico por hablante (todo el periodo)."""
    groups = collections.defaultdict(list)
    for i, doc_id in enumerate(model.space.doc_ids):
        groups[ix.speakers[int(doc_id)]].append(i)
    rows = []
    for speaker, idx in sorted(groups.items()):
        mean_theta = model.doc_topic[idx].mean(axis=0)
        for topic, share in enumerate(mean_theta):
            rows.append({"speaker": speaker, "topic": topic,
                         "share": float(share), "docs": len(idx)})
    return rows


def topic_series(ix, model, topic: int, by: str = "quarter", speaker: str = None) -> list:
    """Serie temporal de la participación de un tópico (para graficar)."""
    rows = [r for r in shares_by(ix, model, by)
            if r["topic"] == topic and (speaker is None or r["speaker"] == speaker)]
    return rows


# --- Vía B: modelos por ventana y alineación --------------------------------
def window_models(ix, *, family: str = "nmf", k: int = config.WINDOW_K,
                  min_docs: int = config.WINDOW_MIN_DOCS, verbose: bool = True) -> list:
    """Ajusta un modelo por (hablante, año) con material suficiente.

    Se ordenan por hablante y año para poder alinear años consecutivos. Cada
    elemento lleva el modelo, sus documentos y la participación de cada tópico.
    """
    from mt_scripts.models import fit

    groups = collections.defaultdict(list)
    for doc_id in range(ix.n_docs):
        groups[(ix.speakers[doc_id], ix.dates[doc_id][:4])].append(doc_id)

    items = []
    for (speaker, year), doc_ids in sorted(groups.items()):
        if len(doc_ids) < min_docs:
            continue
        space = build_subspace(ix, doc_ids)
        if space.n_terms < 30:
            if verbose:
                print(f"  ventana {speaker} {year}: {space.n_terms} términos, se omite")
            continue
        kk = max(2, min(k, len(doc_ids) // 5, space.n_terms // 10))
        model = fit(family, space, kk)
        items.append({
            "speaker": speaker,
            "year": year,
            "doc_ids": doc_ids,
            "model": model,
            "shares": [float(x) for x in model.share()],
        })
        if verbose:
            print(f"  ventana {speaker} {year}: {len(doc_ids)} docs, k={kk}, "
                  f"términos={space.n_terms}")
    return items


def align_windows(items: list, n_terms: int = config.ALIGN_TERMS) -> tuple:
    """Alinea tópicos entre años consecutivos del mismo hablante.

    Devuelve ``(drift, summary)``:

    - ``drift``: una fila por par de tópicos emparejados, con la similitud de
      coseno entre sus vectores de términos (deriva del tópico entre dos años);
    - ``summary``: por par de años, similitud media, nº de tópicos emparejados
      con ``sim ≥ WINDOW_MIN_SIM`` y nº de tópicos nuevos/desaparecidos.
    """
    by_speaker = collections.defaultdict(list)
    for item in items:
        by_speaker[item["speaker"]].append(item)

    drift, summary = [], []
    for speaker, seq in by_speaker.items():
        seq = sorted(seq, key=lambda x: x["year"])
        for prev, cur in zip(seq, seq[1:]):
            mapping = MO.align_topics(prev["model"], cur["model"], n=n_terms)
            sims = []
            for topic_a, (topic_b, sim) in sorted(mapping.items()):
                matched = sim >= config.WINDOW_MIN_SIM
                drift.append({
                    "speaker": speaker,
                    "year_from": prev["year"],
                    "year_to": cur["year"],
                    "topic_from": topic_a,
                    "topic_to": topic_b,
                    "similarity": float(sim),
                    "matched": bool(matched),
                })
                if matched:
                    sims.append(sim)
            summary.append({
                "speaker": speaker,
                "year_from": prev["year"],
                "year_to": cur["year"],
                "mean_similarity": float(np.mean(sims)) if sims else 0.0,
                "matched": int(len(sims)),
                "topics_from": prev["model"].k,
                "topics_to": cur["model"].k,
                "new_topics": int(cur["model"].k - len(sims)),
                "lost_topics": int(prev["model"].k - len(sims)),
            })
    return drift, summary


def drift_persistence(drift: list) -> list:
    """Deriva acumulada por tópico: sigue cada tópico a lo largo de los años.

    Encadena los emparejamientos (a→b, b→c, ...) y devuelve, por hablante y
    tópico inicial, la lista de similitudes y el año final alcanzado. Un tópico
    que se mantiene con similitud alta durante 5 años es "estable"; uno que cae
    por debajo de ``WINDOW_MIN_SIM`` se considera que mutó o desapareció.
    """
    chains = []
    by_speaker = collections.defaultdict(list)
    for row in drift:
        by_speaker[row["speaker"]].append(row)

    for speaker, rows in by_speaker.items():
        by_pair = collections.defaultdict(dict)
        for row in rows:
            by_pair[(row["year_from"], row["year_to"])][row["topic_from"]] = row
        pairs = sorted(by_pair)
        if not pairs:
            continue

        # arranca una cadena en cada tópico del primer par
        first_from, first_to = pairs[0]
        for topic_from in sorted(by_pair[pairs[0]]):
            chain = {"speaker": speaker, "topic_start": topic_from,
                     "year_start": first_from, "path": [], "similarities": []}
            current = topic_from
            for pair in pairs:
                row = by_pair[pair].get(current)
                if row is None:
                    break
                chain["path"].append((pair[0], pair[1], row["topic_from"], row["topic_to"]))
                chain["similarities"].append(float(row["similarity"]))
                current = row["topic_to"]
                chain["year_end"] = pair[1]
                chain["topic_end"] = row["topic_to"]
            chain["mean_similarity"] = float(np.mean(chain["similarities"])) \
                if chain["similarities"] else 0.0
            chain["n_years"] = len(chain["similarities"]) + 1
            chains.append(chain)
    return chains
