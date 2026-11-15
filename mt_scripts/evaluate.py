"""Evaluación del módulo 6: coherencia, estabilidad, anclas y pureza.

No existe un "gold" de tópicos para 4,531 discursos, así que la evaluación
combina cuatro señales independientes:

1. **intrínseca** — c_v y UMass (en `coherence.py`), diversidad y exclusividad;
2. **estabilidad** — remuestreo de documentos: se reajusta el modelo sobre el
   80 % de los discursos y se mide el solapamiento (Jaccard) del top-10 de cada
   tópico tras alinear los tópicos por coseno;
3. **extrínseca con anclas** — el gold anual que ya usa el módulo 5
   (`rat_scripts/gold/year_gold.json`) lista los temas clave de 4 casos; se mide
   si los tópicos descubiertos los cubren;
4. **extrínseca con gold manual** — 25 discursos etiquetados a mano con su área
   temática (`gold/topics_gold.json`); se mide la pureza: ¿todos los discursos
   de la misma área caen en el mismo tópico dominante?

Las cuatro son necesarias: c_v premia tópicos "coherentes" que pueden ser
triviales, la estabilidad premia tópicos fáciles, las anclas dependen de una
lista escrita por nosotros y el gold manual es pequeño. Juntas son difíciles de
satisfacer a la vez sin que los tópicos sean realmente interpretables.
"""

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mt_scripts.config as config
from mt_scripts import coherence as CO
from mt_scripts import models as MO
from mt_scripts.matrices import Space
from rat_scripts.evaluate import markdown_table

YEAR_GOLD = Path(config.PROJECT_ROOT) / "rat_scripts" / "gold" / "year_gold.json"


# --- Anclas del gold anual (módulo 5) ---------------------------------------
def load_year_gold(path: Path = None) -> list:
    path = Path(path or YEAR_GOLD)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["items"]


def _anchor_hit(terms: set, spec: dict) -> bool:
    """Misma semántica que `key_topic_recall` del módulo 5."""
    covered = all(s in terms for s in spec.get("all", []))
    alternatives = spec.get("any", [])
    if alternatives:
        covered = covered and any(s in terms for s in alternatives)
    return covered


def anchor_recall(ix, model, case: dict, *, n_topics: int = 3,
                  n_terms: int = config.TOP_TERMS_WINDOW) -> dict:
    """Cobertura de los temas clave de un (hablante, año) por los tópicos del modelo.

    Se toman los `n_topics` tópicos con mayor participación media en los
    discursos de ese año y se unen sus top-n términos. Un ancla se cubre si sus
    stems están en esa unión.
    """
    rows = [i for i, d in enumerate(model.space.doc_ids)
            if ix.speakers[int(d)] == case["speaker"] and ix.dates[int(d)].startswith(case["year"])]
    if not rows:
        return {"speaker": case["speaker"], "year": case["year"], "docs": 0,
                "topic_recall": 0.0, "hits": {}, "top_topics": []}

    mean_theta = model.doc_topic[rows].mean(axis=0)
    top = np.argsort(-mean_theta)[:n_topics]
    terms = set()
    for t in top:
        terms.update(model.top_terms(int(t), n_terms))

    hits = {spec["label"]: _anchor_hit(terms, spec) for spec in case["key_topics"]}
    return {
        "speaker": case["speaker"],
        "year": case["year"],
        "docs": len(rows),
        "topic_recall": float(sum(hits.values()) / max(1, len(hits))),
        "hits": hits,
        "top_topics": [(int(t), float(mean_theta[t])) for t in top],
    }


def run_anchor_benchmark(ix, model, cases: list = None) -> list:
    return [anchor_recall(ix, model, c) for c in (cases or load_year_gold())]


# --- Gold manual de áreas temáticas -----------------------------------------
def load_gold_topics(path: Path = None) -> list:
    path = Path(path or config.GOLD_TOPICS_JSON)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["items"] if isinstance(data, dict) else data


def gold_purity(model, items: list) -> dict:
    """Pureza del tópico dominante respecto al gold manual.

    ``pureza = (1/N) Σ_tema max_t nº(docs del tema con tópico dominante t)``.
    Es invariante a la permutación de etiquetas: sólo comprueba que el modelo
    separa las áreas temáticas, no que las numere como nosotros.

    Se reporta además ``purity@2``: la proporción de discursos cuyo tópico
    dominante **o segundo** coincide con el tópico modal de su tema. El gold es
    de "área principal" y muchos discursos son multimodales, así que la lectura
    estricta subestima.
    """
    pos = {int(d): i for i, d in enumerate(model.space.doc_ids)}
    usable = [it for it in items if int(it["doc_id"]) in pos]
    if not usable:
        return {"purity": 0.0, "purity_at2": 0.0, "n": 0, "by_theme": []}

    order = np.argsort(-model.doc_topic, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    tables = collections.defaultdict(lambda: collections.Counter())
    for it in usable:
        i = pos[int(it["doc_id"])]
        tables[it["theme"]][int(top1[i])] += 1

    hits = sum(c.most_common(1)[0][1] for c in tables.values())

    # purity@2 con el tópico modal de cada tema
    modal = {theme: int(c.most_common(1)[0][0]) for theme, c in tables.items()}
    hits2 = 0
    for it in usable:
        i = pos[int(it["doc_id"])]
        if modal[it["theme"]] in (int(top1[i]), int(top2[i])):
            hits2 += 1

    by_theme = []
    for theme, counter in sorted(tables.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(counter.values())
        topic, n = counter.most_common(1)[0]
        by_theme.append({
            "theme": theme, "docs": total, "modal_topic": int(topic),
            "modal_share": float(n / total),
            "topics": {int(k): int(v) for k, v in counter.most_common()},
        })
    return {
        "purity": float(hits / len(usable)),
        "purity_at2": float(hits2 / len(usable)),
        "n": len(usable),
        "n_themes": len(tables),
        "by_theme": by_theme,
    }


# --- Estabilidad -------------------------------------------------------------
def _subset_space(space: Space, rows: np.ndarray) -> Space:
    """Mismo vocabulario/IDF que el espacio base, sólo cambian las filas."""
    return Space(
        doc_ids=space.doc_ids[rows], terms_ids=space.terms_ids, terms=space.terms,
        X=space.X[rows].tocsr(), W=space.W[rows].tocsr(), W_l2=space.W_l2[rows].tocsr(),
        df=space.df, idf=space.idf, global_vocab_size=space.global_vocab_size,
    )


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(sa & sb) / max(1, len(sa | sb))


def stability(ix, model, *, runs: int = None, subsample: float = None,
              cooc: dict = None, n_terms: int = config.TOP_TERMS_DIVERSITY,
              verbose: bool = True) -> dict:
    """Remuestreo de documentos: ¿los tópicos sobreviven a perder el 20 %?

    Para cada corrida se reajusta el mismo modelo (familia y k) sobre una
    submuestra de documentos, se alinean sus tópicos con los del modelo base por
    coseno (húngaro) y se mide el Jaccard del top-n. Se reporta la media, la
    desviación y el Jaccard por tópico.
    """
    runs = config.STABILITY_RUNS if runs is None else runs
    if model.family == "lda":
        runs = min(runs, config.STABILITY_RUNS_LDA)
    subsample = config.STABILITY_SUBSAMPLE if subsample is None else subsample

    rng = np.random.default_rng(config.SEED)
    n = model.n_docs
    base = [set(model.top_ids(t, n_terms).tolist()) for t in range(model.k)]
    per_topic = collections.defaultdict(list)
    cv_runs = []

    for run in range(runs):
        size = max(50, int(round(subsample * n)))
        rows = np.sort(rng.choice(n, size=size, replace=False))
        kwargs = {}
        if model.family == "lda":
            kwargs = {"max_iter": config.STABILITY_LDA_ITERS, "compute_perplexity": False}
        sub = _subset_space(model.space, rows)
        refit = MO.fit(model.family, sub, model.k, seed=config.SEED + run, **kwargs)

        mapping = MO.align_topics(model, refit)
        for topic, (topic_b, _sim) in mapping.items():
            per_topic[topic].append(_jaccard(base[topic], set(refit.top_ids(topic_b, n_terms).tolist())))
        if cooc is not None:
            cv_runs.append(CO.evaluate_model(refit, cooc, cooc["cand_ids"])["c_v"])
        if verbose:
            print(f"  estabilidad {run + 1}/{runs}: "
                  f"jaccard medio={np.mean([x for v in per_topic.values() for x in v]):.3f}")

    values = [x for v in per_topic.values() for x in v]
    return {
        "runs": runs,
        "subsample": subsample,
        "jaccard_mean": float(np.mean(values)),
        "jaccard_std": float(np.std(values)),
        "jaccard_per_topic": {int(t): float(np.mean(v)) for t, v in per_topic.items()},
        "c_v_runs": [float(x) for x in cv_runs],
        "c_v_std": float(np.std(cv_runs)) if cv_runs else None,
    }


# --- Tablas ------------------------------------------------------------------
def sweep_table(rows: list) -> str:
    out = []
    for r in rows:
        out.append({
            "familia": MO.FAMILY_LABEL.get(r["family"], r["family"]),
            "k": r["k"],
            "c_v": round(r["c_v"], 3),
            "UMass": round(r["umass"], 3),
            "diversidad": round(r["diversity"], 3),
            "exclusividad": round(r["exclusivity"], 3),
            "tiempo (s)": round(r.get("seconds", 0.0), 1),
        })
    return markdown_table(
        out, columns=["familia", "k", "c_v", "UMass", "diversidad", "exclusividad", "tiempo (s)"],
        headers=["familia", "k", "c_v", "UMass", "diversidad", "exclusividad", "tiempo (s)"],
    )


def topics_table(label_rows: list, shares: list) -> str:
    share_map = {(r["speaker"], r["topic"]): r["share"] for r in shares}
    speakers = sorted({s for s, _ in share_map})
    out = []
    for row in label_rows:
        item = {
            "tópico": row["topic"],
            "nombre": row["name"],
            "términos": row["top_terms"],
            "anclas": row["anchors"],
        }
        for spk in speakers:
            item[spk] = round(share_map.get((spk, row["topic"]), 0.0), 3)
        out.append(item)
    return markdown_table(
        out, columns=["tópico", "nombre", "términos", "anclas"] + speakers,
        headers=["tópico", "nombre", "términos", "anclas"] + speakers,
    )


def timeline_table(shares: list, *, by: str = "year", topics: int = None) -> str:
    df = pd.DataFrame(shares)
    df["cell"] = df["speaker"].str.split().str[0] + " " + df["period"].astype(str)
    pivot = df.pivot_table(index="topic", columns="cell", values="share", aggfunc="mean")
    if topics is not None:
        pivot = pivot.loc[[t for t in pivot.index if t in topics]]
    pivot = pivot.fillna(0.0).round(3)
    pivot.insert(0, "tópico", pivot.index)
    rows = pivot.reset_index(drop=True).to_dict(orient="records")
    cols = ["tópico"] + list(pivot.columns[1:])
    return markdown_table(rows, columns=cols, headers=cols)


def anchors_table(results: list) -> str:
    out = []
    for r in results:
        out.append({
            "caso": f"{r['speaker']} {r['year']}",
            "discursos": r["docs"],
            "recall": round(r["topic_recall"], 3),
            "anclas cubiertas": sum(1 for v in r["hits"].values() if v),
            "anclas": len(r["hits"]),
            "tópicos dominantes": ", ".join(f"T{t}" for t, _ in r["top_topics"]),
        })
    return markdown_table(
        out, columns=["caso", "discursos", "recall", "anclas cubiertas", "anclas", "tópicos dominantes"],
        headers=["caso", "discursos", "recall", "anclas cubiertas", "anclas", "tópicos dominantes"],
    )


def purity_table(result: dict, label_names: dict) -> str:
    out = []
    for row in result["by_theme"]:
        out.append({
            "área temática (manual)": row["theme"],
            "discursos": row["docs"],
            "tópico modal": f"T{row['modal_topic']}",
            "nombre del tópico": label_names.get(row["modal_topic"], ""),
            "coincidencia": round(row["modal_share"], 3),
        })
    return markdown_table(
        out, columns=["área temática (manual)", "discursos", "tópico modal", "nombre del tópico", "coincidencia"],
        headers=["área temática (manual)", "discursos", "tópico modal", "nombre del tópico", "coincidencia"],
    )


def stability_table(result: dict) -> str:
    return markdown_table(
        [{
            "corridas": result["runs"],
            "submuestra": result["subsample"],
            "Jaccard medio": round(result["jaccard_mean"], 3),
            "Jaccard σ": round(result["jaccard_std"], 3),
            "c_v σ": round(result["c_v_std"], 3) if result["c_v_std"] is not None else "",
        }],
        columns=["corridas", "submuestra", "Jaccard medio", "Jaccard σ", "c_v σ"],
        headers=["corridas", "submuestra", "Jaccard medio", "Jaccard σ", "c_v σ"],
    )


def window_table(items: list, drift_summary: list, surfaces: dict = None) -> str:
    """Tabla de ventanas a partir de la estructura serializada en JSON."""
    out = []
    for item in items:
        topics = item["topics"]
        dom = max(topics, key=lambda t: t["share"])
        out.append({
            "hablante": item["speaker"],
            "año": item["year"],
            "discursos": item["docs"],
            "k": item["k"],
            "tópico dominante": ", ".join(dom["top_terms"][:6]),
            "participación": round(float(dom["share"]), 3),
        })
    return markdown_table(
        out, columns=["hablante", "año", "discursos", "k", "tópico dominante", "participación"],
        headers=["hablante", "año", "discursos", "k", "tópico dominante", "participación"],
    )


def drift_table(drift_summary: list) -> str:
    out = [{
        "hablante": r["speaker"],
        "transición": f"{r['year_from']}→{r['year_to']}",
        "similitud media": round(r["mean_similarity"], 3),
        "emparejados": r["matched"],
        "nuevos": r["new_topics"],
        "perdidos": r["lost_topics"],
    } for r in drift_summary]
    return markdown_table(
        out, columns=["hablante", "transición", "similitud media", "emparejados", "nuevos", "perdidos"],
        headers=["hablante", "transición", "similitud media", "emparejados", "nuevos", "perdidos"],
    )
