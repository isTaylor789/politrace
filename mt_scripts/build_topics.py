"""Genera todos los artefactos del módulo 6: tópicos, series temporales y eval.

    source ../env/bin/activate
    python mt_scripts/build_topics.py                 # barrido completo (~10-15 min)
    python mt_scripts/build_topics.py --force         # reconstruye tokens y mapa
    python mt_scripts/build_topics.py --families nmf --k 6,8 --skip-stability

Escribe en `mt_scripts/artifacts/`:

- ``sweep.json``            barrido familia × k con coherencia y tiempos
- ``topics_global.json``    tópicos del modelo elegido (términos, nombres, docs)
- ``doc_topics.jsonl``      θ de cada documento (y tópico dominante)
- ``timeline.json``         series por año/trimestre, modelos por ventana y deriva
- ``topics_by_speaker.json`` tópicos de un modelo por hablante
- ``topics_by_window.json``  tópicos por hablante-año y alineación entre años
- ``eval.json``             anclas, pureza, estabilidad y parámetros
- ``tables.md``             tablas del reporte
- ``cooc_npz.npz``          NPMI y co-ocurrencia documental de los candidatos
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ri_scripts.index import load_index  # noqa: E402
from ri_scripts.preprocess import fold  # noqa: E402

import mt_scripts.config as config  # noqa: E402
from mt_scripts import coherence as CO  # noqa: E402
from mt_scripts import evaluate as EV  # noqa: E402
from mt_scripts import labels as LAB  # noqa: E402
from mt_scripts import models as MO  # noqa: E402
from mt_scripts import timeline as TL  # noqa: E402
from mt_scripts.matrices import (build_space, build_stem_surface,  # noqa: E402
                                 build_subspace, build_tokens,
                                 verify_reconstruction)

SWEEP_FAMILIES = ("lsa", "nmf", "lda")


# --- Utilidades --------------------------------------------------------------
def _f(value):
    """float nativo (evita np.float64 en json)."""
    return float(value)


def snippet(text: str, words: list, width: int = config.SNIPPET_CHARS) -> str:
    """Extracto del discurso alrededor de la primera aparición de un término."""
    low = fold(text)
    best = -1
    for word in words:
        idx = low.find(word.lower())
        if idx >= 0 and (best < 0 or idx < best):
            best = idx
    if best < 0:
        return " ".join(text.split())[:width]
    start = max(0, best - 90)
    return " ".join(text[start:start + width].split())


def topic_record(ix, model, topic: int, surfaces: dict, *, n_docs: int = config.N_EVIDENCE_DOCS) -> dict:
    """Ficha completa de un tópico: nombre, términos, participación y evidencia."""
    from ri_scripts.corpus import read_doc

    info = LAB.topic_label(model, topic, surfaces)
    idx = model.top_idx(topic, config.TOP_TERMS)
    weights = model.topic_word[topic, idx]
    col_sum = model.topic_word.sum(axis=0)
    top_terms = [{
        "term": model.terms[int(j)],
        "surface": LAB.readable(model.terms[int(j)], surfaces),
        "weight": _f(w),
        "exclusive": _f(w / max(col_sum[int(j)], 1e-12)),
    } for j, w in zip(idx, weights)]

    # documentos representativos: mayor θ en el tópico
    order = np.argsort(-model.doc_topic[:, topic])[:n_docs]
    docs = []
    for row in order:
        doc_id = int(model.space.doc_ids[row])
        text = read_doc(doc_id, ix.corpus_path, ix.offsets)["text"]
        docs.append({
            "doc_id": doc_id,
            "speech_id": ix.doc_ids[doc_id],
            "speaker": ix.speakers[doc_id],
            "date": ix.dates[doc_id],
            "theta": _f(model.doc_topic[row, topic]),
            "snippet": snippet(text, [x["surface"] for x in top_terms[:5]]),
        })

    row = model.doc_topic[:, topic]
    return {
        "topic": topic,
        "name": info["name"],
        "auto_label": info["auto"],
        "manual_label": info["manual"],
        "manual_anchors": info["anchors"],
        "share": _f(row.mean()),
        "exclusivity": _f(np.mean([x["exclusive"] for x in top_terms])),
        "top_terms": top_terms,
        "top_docs": docs,
    }


def shares_pivot(rows: list, value: str = "share") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["cell"] = df["speaker"].str.split().str[0] + " " + df["period"].astype(str)
    return df.pivot_table(index="topic", columns="cell", values=value, aggfunc="mean")


def _markdown_named_table(rows: list, columns: list, headers: list) -> str:
    return EV.markdown_table(rows, columns=columns, headers=headers)


# --- Construcción ------------------------------------------------------------
def build_sweep(ix, space, families=SWEEP_FAMILIES, ks=config.K_GRID) -> tuple:
    """Ajusta todas las combinaciones familia × k y mide tiempo y coherencia."""
    models, rows = {}, []
    for family in families:
        for k in ks:
            t0 = time.time()
            model = MO.fit(family, space, int(k))
            seconds = time.time() - t0
            models[(family, int(k))] = model
            rows.append({
                "family": family, "k": int(k), "seconds": round(seconds, 2),
                "extras": {key: (round(val, 5) if isinstance(val, float) else val)
                           for key, val in model.extras.items()},
            })
            print(f"  {MO.FAMILY_LABEL[family]:18s} k={k:2d}  {seconds:6.1f}s  "
                  f"términos top: {', '.join(model.top_terms(0, 5))}")
    return models, rows


def select_model(models: dict, rows: list) -> dict:
    """Elige familia y k con una regla de **un error estándar**.

    La c_v sube de forma casi monótona con k (más tópicos = listas más
    específicas), así que quedarse con el máximo favorece siempre el k mayor y
    produce tópicos fragmentados. Se usa la regla de un error estándar, habitual
    en selección de modelos:

    1. familia = la de mayor c_v (la comparación familia × k ya está hecha);
    2. dentro de esa familia, ``SE = σ(c_v por tópico) / sqrt(k)`` del mejor k;
    3. se elige el **menor** k con ``c_v >= c_v_max − max(SE, tolerancia)``.

    Es una regla de parsimonia: acepta un modelo más pequeño si su coherencia es
    estadísticamente indistinguible del mejor. La diversidad ≥
    ``config.MIN_DIVERSITY`` actúa de filtro previo (un modelo con tópicos
    solapados no es elegible aunque tenga c_v alta).
    """
    eligible = [r for r in rows if r["diversity"] >= config.MIN_DIVERSITY] or rows
    best_row = max(eligible, key=lambda r: (round(r["c_v"], 4), round(r["umass"], 4)))
    family_rows = [r for r in eligible if r["family"] == best_row["family"]]

    per_topic = np.asarray(best_row.get("c_v_per_topic") or [best_row["c_v"]])
    se = float(per_topic.std(ddof=1) / np.sqrt(len(per_topic))) if len(per_topic) > 1 else 0.0
    band = max(se, config.SELECTION_TOLERANCE)
    threshold = float(best_row["c_v"] - band)

    chosen = min((r for r in family_rows if r["c_v"] >= threshold),
                 key=lambda r: r["k"], default=best_row)
    return {
        "family": chosen["family"], "k": chosen["k"],
        "c_v": chosen["c_v"], "umass": chosen["umass"],
        "diversity": chosen["diversity"], "exclusivity": chosen["exclusivity"],
        "best_in_family": {"k": best_row["k"], "c_v": best_row["c_v"], "se": se,
                           "band": band},
        "threshold": threshold,
        "criterion": "menor k con c_v >= c_v_max - max(SE, 0.05) (regla 1-SE)",
    }


def speaker_models(ix, family: str, surfaces: dict) -> list:
    """Un modelo por hablante (k = SPEAKER_K) con sus tópicos y participaciones."""
    out = []
    for speaker, (start, end) in ix.speaker_blocks.items():
        doc_ids = np.arange(start, end)
        space = build_subspace(ix, doc_ids)
        k = max(2, min(config.SPEAKER_K, len(doc_ids) // 5, space.n_terms // 10))
        model = MO.fit(family, space, k)
        topics = []
        for t in range(k):
            info = LAB.topic_label(model, t, surfaces)
            topics.append({
                "topic": t,
                "name": info["name"],
                "auto_label": info["auto"],
                "manual_label": info["manual"],
                "manual_anchors": info["anchors"],
                "share": _f(model.share()[t]),
                "exclusivity": _f(model.exclusivity()[t]),
                "top_terms": [LAB.readable(x, surfaces) for x in model.top_terms(t, config.TOP_TERMS)],
            })
        out.append({
            "speaker": speaker,
            "docs": int(len(doc_ids)),
            "k": k,
            "terms": int(space.n_terms),
            "low_confidence": bool(len(doc_ids) < config.SPEAKER_LOW_CONFIDENCE),
            "topics": sorted(topics, key=lambda x: -x["share"]),
        })
        print(f"  hablante {speaker:32s} {len(doc_ids):5d} docs  k={k}  "
              f"términos={space.n_terms}  tópico 0: {', '.join(topics[0]['top_terms'][:5])}")
    return out


def window_analysis(ix, surfaces: dict) -> dict:
    """Vía B: modelos por hablante-año, alineación y cadenas de deriva."""
    items = TL.window_models(ix, family="nmf", k=config.WINDOW_K)
    drift, summary = TL.align_windows(items)
    chains = TL.drift_persistence(drift)

    payload_items = []
    for item in items:
        model = item["model"]
        payload_items.append({
            "speaker": item["speaker"],
            "year": item["year"],
            "docs": len(item["doc_ids"]),
            "k": model.k,
            "topics": [{
                "topic": t,
                "name": LAB.topic_label(model, t, surfaces)["name"],
                "share": _f(item["shares"][t]),
                "top_terms": [LAB.readable(x, surfaces) for x in model.top_terms(t, 8)],
            } for t in range(model.k)],
        })
    return {
        "params": {"family": "nmf", "k": config.WINDOW_K,
                   "min_docs": config.WINDOW_MIN_DOCS, "min_sim": config.WINDOW_MIN_SIM},
        "items": payload_items,
        "drift": drift,
        "summary": summary,
        "chains": [{key: (value if not isinstance(value, (np.floating, np.integer)) else value.item())
                    for key, value in chain.items()} for chain in chains],
    }


# --- Tablas ------------------------------------------------------------------
def build_tables(payload: dict) -> str:
    """Construye `tables.md` con todas las tablas del reporte."""
    meta = payload["meta"]
    best = meta["chosen"]

    parts = ["# Tablas — Topic modeling y evolución temporal (módulo 6)\n",
             "Generado por `mt_scripts/build_topics.py`.\n",
             "## 0. Base de RI reutilizada\n",
             _markdown_named_table(
                 [{
                     "documentos": meta["verification"]["n_docs"] if "n_docs" in meta["verification"] else meta["n_docs"],
                     "vocabulario RI": meta["verification"]["n_ri"],
                     "diferencia máx. con w_global": meta["verification"]["max_abs_diff"],
                     "reconstrucción exacta": "sí" if meta["verification"]["ok"] else "no",
                     "vocabulario MT": meta["space"]["n_terms"],
                     "términos en co-ocurrencia": meta["cooc"]["n_candidates"],
                 }],
                 ["documentos", "vocabulario RI", "diferencia máx. con w_global",
                  "reconstrucción exacta", "vocabulario MT", "términos en co-ocurrencia"],
                 ["documentos", "vocabulario RI", "diferencia máx. con w_global",
                  "reconstrucción exacta", "vocabulario MT", "términos en co-ocurrencia"],
             ),
             "Términos del vocabulario de MT: df ≥ 10, df ≤ 50 % de los documentos, sin dígitos, "
             f"sin la lista de puesta en escena (`config.BOILERPLATE_TERMS`: cargos, gentilicios "
             f"de autorreferencia, nombres de los hablantes, dícticos), tope {config.MAX_TERMS} "
             f"por masa df·idf. Ventana de co-ocurrencia: {config.COOC_WINDOW} tokens.\n",
             "## 1. Barrido familia × k (coherencia)\n",
             EV.sweep_table(payload["sweep"]),
             f"Modelo elegido: **{MO.FAMILY_LABEL[best['family']]} con k = {best['k']}** "
             f"(c_v = {best['c_v']:.3f}, UMass = {best['umass']:.3f}, "
             f"diversidad = {best['diversity']:.3f}). Regla 1-SE: c_v máxima "
             f"{best['best_in_family']['c_v']:.3f} en k = {best['best_in_family']['k']}, "
             f"SE = {best['best_in_family']['se']:.3f}, banda = {best['best_in_family']['band']:.3f}, "
             f"umbral = {best['threshold']:.3f}; se toma el menor k que supera el umbral.\n",
             "## 2. Tópicos del modelo elegido\n",
             EV.topics_table(payload["label_rows"], payload["speaker_shares"]),
             "La columna *anclas* es el número de anclas del gazetteer que dispararon el nombre "
             "manual; cuando es 0 el nombre es automático (los tres términos de mayor peso).\n",
             "## 3. Evolución temporal (vía A: un modelo, θ por periodo)\n",
             "Participación media de cada tópico por hablante-año (cada fila de θ suma 1; "
             "*no* son conteos, así que el desbalance del corpus no domina la tabla).\n",
             EV.timeline_table(payload["timeline"]["year"]),
             "### 3.1 Ventana limpia 2023-2024\n",
             EV.timeline_table(payload["timeline"]["year_clean"]),
             "Única ventana donde 4 de los 5 líderes están activos (plan base, restricción 3).\n",
             "## 4. Tópicos por hablante (modelo independiente, k = "
             f"{config.SPEAKER_K})\n",
             _markdown_named_table(
                 [{
                     "hablante": item["speaker"],
                     "discursos": item["docs"],
                     "términos": item["terms"],
                     "tópico principal": item["topics"][0]["name"],
                     "participación": round(item["topics"][0]["share"], 3),
                     "términos del tópico": ", ".join(item["topics"][0]["top_terms"][:8]),
                     "confianza": "baja" if item["low_confidence"] else "normal",
                 } for item in payload["speakers"]],
                 ["hablante", "discursos", "términos", "tópico principal", "participación",
                  "términos del tópico", "confianza"],
                 ["hablante", "discursos", "términos", "tópico principal", "participación",
                  "términos del tópico", "confianza"],
             ),
             "## 5. Dinámica por ventanas (vía B)\n",
             EV.window_table(payload["windows"]["items"], payload["windows"]["summary"], payload["surfaces"]),
             EV.drift_table(payload["windows"]["summary"]),
             "Deriva = similitud de coseno entre el vector de términos del mismo tópico en dos años "
             f"consecutivos, tras alineación húngara. `sim ≥ {config.WINDOW_MIN_SIM}` cuenta como "
             "tópico emparejado; lo demás son tópicos nuevos/perdidos.\n",
             "## 6. Evaluación\n",
             "### 6.1 Anclas del gold anual del módulo 5\n",
             EV.anchors_table(payload["eval"]["anchors"]),
             "Un ancla se cubre si sus stems están entre los 30 términos más pesados de los 3 tópicos "
             "dominantes del año.\n",
             "### 6.2 Pureza contra el gold manual de áreas temáticas\n",
             f"Pureza = {payload['eval']['purity']['purity']:.3f} "
             f"(pureza@2 = {payload['eval']['purity']['purity_at2']:.3f}, "
             f"n = {payload['eval']['purity']['n']}).\n",
             EV.purity_table(payload["eval"]["purity"], payload["label_names"]),
             "### 6.3 Sensibilidad a k (misma familia)\n",
             _markdown_named_table(
                 [{
                     "k": r["k"],
                     "c_v": round(r["c_v"], 3),
                     "diversidad": round(r["diversity"], 3),
                     "pureza": round(r["purity"], 3),
                     "pureza@2": round(r["purity_at2"], 3),
                     "anclas": round(r["anchor_recall"], 3),
                     "elegido": "←" if r["selected"] else "",
                 } for r in payload["eval"]["sensitivity_k"]],
                 ["k", "c_v", "diversidad", "pureza", "pureza@2", "anclas", "elegido"],
                 ["k", "c_v", "diversidad", "pureza", "pureza@2", "anclas", "elegido"],
             ),
             "### 6.4 Estabilidad (remuestreo de documentos)\n",
             EV.stability_table(payload["eval"]["stability"]) if payload["eval"].get("stability") else "*no calculada*\n",
             "## 7. Pares con mayor NPMI en el corpus\n",
             _markdown_named_table(
                 [{"término A": p["term_a"], "término B": p["term_b"],
                   "NPMI": round(p["npmi"], 3), "pares": p["pairs"]}
                  for p in payload["pairs"]],
                 ["término A", "término B", "NPMI", "pares"],
                 ["término A", "término B", "NPMI", "pares"],
             ),
             "NPMI sobre pares de tokens a distancia < 10 (ver `mt_scripts/coherence.py`).\n"]
    return "\n".join(parts)


# --- Main --------------------------------------------------------------------
def main(families=SWEEP_FAMILIES, ks=config.K_GRID, force: bool = False,
         skip_stability: bool = False) -> None:
    t_start = time.time()
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("cargando índice de RI...")
    ix = load_index()
    verification = verify_reconstruction(ix)
    print(f"  reconstrucción de w_global desde postings+idf: "
          f"max|Δ|={verification['max_abs_diff']:.2e} ok={verification['ok']}")

    print("streams de tokens y formas superficiales...")
    tokens = build_tokens(ix, force=force)
    surfaces = build_stem_surface(ix, force=force)
    print(f"  {tokens['n_tokens']:,} tokens, {len(surfaces):,} stems con forma legible")

    print("\nconstruyendo matriz de trabajo...")
    space = build_space(ix)
    print(f"  {space.n_docs} docs x {space.n_terms} términos ({space.X.nnz:,} no-cero)")

    print("\nbarrido familia x k...")
    models, sweep = build_sweep(ix, space, families, ks)

    print("\nco-ocurrencia y NPMI para los candidatos...")
    cand = np.unique(np.concatenate(
        [m.top_ids(t, config.COOC_TOP_TERMS) for m in models.values() for t in range(m.k)]
    ))
    cooc = CO.build_cooc(ix, cand)
    CO.save_cooc(cooc)

    for row in sweep:
        row.update(CO.evaluate_model(models[(row["family"], row["k"])], cooc, cand))
        print(f"  {MO.FAMILY_LABEL[row['family']]:18s} k={row['k']:2d}  "
              f"c_v={row['c_v']:.3f}  UMass={row['umass']:.3f}  div={row['diversity']:.3f}")

    chosen = select_model(models, sweep)
    final = models[(chosen["family"], chosen["k"])]
    print(f"\nmodelo elegido: {MO.FAMILY_LABEL[chosen['family']]} k={chosen['k']} "
          f"(c_v={chosen['c_v']:.3f})")

    print("fichas de tópicos...")
    topics = [topic_record(ix, final, t, surfaces) for t in range(final.k)]
    label_rows = LAB.label_table(final, surfaces)
    label_names = {t: topics[t]["name"] for t in range(final.k)}

    print("series temporales...")
    shares_year = TL.shares_by(ix, final, "year")
    shares_quarter = TL.shares_by(ix, final, "quarter")
    speaker_shares = TL.speaker_shares(ix, final)
    shares_year_clean = [r for r in shares_year if r["period"] in config.CLEAN_WINDOW]
    shares_quarter_clean = [r for r in shares_quarter if r["period"][:4] in config.CLEAN_WINDOW]

    print("modelos por hablante...")
    speakers = speaker_models(ix, chosen["family"], surfaces)

    print("modelos por ventana y alineación...")
    windows = window_analysis(ix, surfaces)

    print("evaluación...")
    gold = EV.load_gold_topics()
    anchors = EV.run_anchor_benchmark(ix, final)
    purity = EV.gold_purity(final, gold)

    # Sensibilidad a k: la misma familia, todos los k del barrido. Muestra si la
    # elección por parsimonia cuesta coherencia o, al contrario, gana pureza.
    sensitivity = []
    for row in sorted((r for r in sweep if r["family"] == chosen["family"]), key=lambda r: r["k"]):
        model = models[(row["family"], row["k"])]
        pur = EV.gold_purity(model, gold)
        anc = EV.run_anchor_benchmark(ix, model)
        sensitivity.append({
            "k": row["k"], "c_v": row["c_v"], "diversity": row["diversity"],
            "purity": pur["purity"], "purity_at2": pur["purity_at2"],
            "anchor_recall": float(np.mean([a["topic_recall"] for a in anc])),
            "selected": row["k"] == chosen["k"],
        })
        print(f"  sensibilidad k={row['k']:2d}: c_v={row['c_v']:.3f} pureza={pur['purity']:.3f} "
              f"anclas={np.mean([a['topic_recall'] for a in anc]):.3f}")
    print(f"  anclas: recall medio={np.mean([a['topic_recall'] for a in anchors]):.3f}")
    print(f"  pureza gold manual: {purity['purity']:.3f} (n={purity['n']})")
    stab = None
    if not skip_stability:
        stab = EV.stability(ix, final, cooc=cooc)
        print(f"  estabilidad: Jaccard={stab['jaccard_mean']:.3f} ± {stab['jaccard_std']:.3f}")

    pairs = CO.strongest_pairs(cooc, ix, 20)

    meta = {
        "params": {
            "min_df": config.MIN_DF, "max_df_ratio": config.MAX_DF_RATIO,
            "max_terms": config.MAX_TERMS, "k_grid": list(ks),
            "families": list(families), "seed": config.SEED,
            "cooc_window": config.COOC_WINDOW, "cooc_top_terms": config.COOC_TOP_TERMS,
            "nmf": {"init": config.NMF_INIT, "max_iter": config.NMF_MAX_ITER},
            "lda": {"max_iter": config.LDA_MAX_ITER,
                    "learning_method": config.LDA_LEARNING_METHOD},
            "lsa_components": config.LSA_COMPONENTS,
            "window": {"k": config.WINDOW_K, "min_docs": config.WINDOW_MIN_DOCS,
                       "min_sim": config.WINDOW_MIN_SIM},
            "stability": {"runs": config.STABILITY_RUNS,
                          "subsample": config.STABILITY_SUBSAMPLE},
            "clean_window": list(config.CLEAN_WINDOW),
        },
        "verification": {**verification, "n_docs": ix.n_docs, "n_ri": ix.vocab_size},
        "n_docs": ix.n_docs,
        "space": {"n_terms": space.n_terms, "nnz": int(space.X.nnz)},
        "cooc": {"n_candidates": int(len(cand)), "window": config.COOC_WINDOW},
        "chosen": chosen,
        "seconds": round(time.time() - t_start, 1),
    }

    # --- escritura de artefactos --------------------------------------------
    for path, data in (
        (config.SWEEP_JSON, {"meta": meta, "rows": sweep}),
        (config.TOPICS_JSON, {"meta": meta, "chosen": chosen,
                              "topics": topics, "label_rows": label_rows,
                              "speaker_shares": speaker_shares}),
        (config.TIMELINE_JSON, {"meta": meta, "year": shares_year,
                                "quarter": shares_quarter,
                                "year_clean": shares_year_clean,
                                "quarter_clean": shares_quarter_clean,
                                "speaker_shares": speaker_shares}),
        (config.SPEAKER_TOPICS_JSON, {"meta": meta, "items": speakers}),
        (config.WINDOW_TOPICS_JSON, {"meta": meta, **windows}),
        (config.EVAL_JSON, {"meta": meta, "anchors": anchors, "purity": purity,
                            "stability": stab, "gold_items": len(gold),
                            "sensitivity_k": sensitivity, "pairs": pairs}),
    ):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=_f)
        print(f"  -> {path.name}")

    with open(config.DOC_TOPICS_JSONL, "w", encoding="utf-8") as fh:
        for row in TL.doc_topics_table(ix, final):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  -> {config.DOC_TOPICS_JSONL.name}")

    payload = {
        "meta": meta, "sweep": sweep, "label_rows": label_rows,
        "label_names": label_names, "speaker_shares": speaker_shares,
        "timeline": {"year": shares_year, "year_clean": shares_year_clean},
        "speakers": speakers, "windows": windows, "surfaces": surfaces,
        "eval": {"anchors": anchors, "purity": purity, "stability": stab,
                 "sensitivity_k": sensitivity},
        "pairs": pairs,
    }
    with open(config.TABLES_MD, "w", encoding="utf-8") as fh:
        fh.write(build_tables(payload))
    print(f"  -> {config.TABLES_MD.name}")

    print(f"\ntotal: {time.time() - t_start:.1f}s")
    print(f"artefactos en {config.ARTIFACTS}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="reconstruye tokens y mapa stem->superficie")
    parser.add_argument("--families", default=",".join(SWEEP_FAMILIES),
                        help="lista separada por comas: lsa,nmf,lda")
    parser.add_argument("--k", default=",".join(str(k) for k in config.K_GRID),
                        help="lista de k separada por comas")
    parser.add_argument("--skip-stability", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        families=tuple(x.strip() for x in args.families.split(",") if x.strip()),
        ks=tuple(int(x) for x in args.k.split(",") if x.strip()),
        force=args.force,
        skip_stability=args.skip_stability,
    )
