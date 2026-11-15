"""Genera los artefactos del módulo RAT: evaluación + resúmenes de ejemplo.

    python rat_scripts/build_summaries.py

Escribe en `rat_scripts/artifacts/`:

- ``eval.json``        métricas por documento y por método + ablación de pesos
- ``summaries_demo.jsonl``  resúmenes híbridos de los 5 discursos de referencia
- ``tables.md``        todas las tablas del reporte en markdown
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ri_scripts import config as ri_config  # noqa: E402
from ri_scripts.index import load_index  # noqa: E402
from rat_scripts import config  # noqa: E402
from rat_scripts.evaluate import (aggregate, evaluate_summary, gold_markdown,  # noqa: E402
                                  load_gold, markdown_table, run_benchmark)
from rat_scripts.summarize import summarize_doc, summarize_docs  # noqa: E402

MULTI_SPEAKER = "Pedro Sánchez"
MULTI_YEAR = "2023"
MULTI_DOCS = 5

ABLATIONS = ["centroid", "textrank", "discourse", "lsa", "luhn"]


def _json_default(value):
    """Convierte escalares de NumPy (int64/float64) a tipos nativos."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"no serializable: {type(value)}")


def _summary_row(summary) -> dict:
    stats = summary.stats
    return {
        "doc_id": summary.doc_id,
        "speaker": summary.speaker,
        "date": summary.date,
        "method": summary.method,
        "compression": stats.get("compression", 0.0),
        "coverage": stats.get("coverage", 0.0),
        "redundancy": stats.get("redundancy", 0.0),
        "n_selected": stats.get("n_selected", 0),
        "mean_sent_words": stats.get("mean_sent_words", 0.0),
    }


def ablation(ix, gold) -> list:
    """ROUGE del híbrido quitando un componente cada vez (importancia real)."""
    base = dict(config.WEIGHTS)
    rows = []
    for drop in [None] + ABLATIONS:
        weights = dict(base)
        if drop is not None:
            weights[drop] = 0.0
        r1, r2, rl, cov = [], [], [], []
        for item in gold:
            summary = summarize_doc(item["doc_id"], ix=ix, weights=weights)
            metrics = evaluate_summary(summary, item["reference"])
            r1.append(metrics["rouge1_fmeasure"])
            r2.append(metrics["rouge2_fmeasure"])
            rl.append(metrics["rougeL_fmeasure"])
            cov.append(metrics["coverage"])
        rows.append({
            "variante": "completo" if drop is None else f"sin {drop}",
            "rouge1": float(np.mean(r1)),
            "rouge2": float(np.mean(r2)),
            "rougeL": float(np.mean(rl)),
            "coverage": float(np.mean(cov)),
        })
    return rows


def build_tables(meta: dict, gold: list, rows: list, agg: list, abl: list,
                 demo: list, multi, multi_docs: list) -> str:
    out = []
    out.append("# Tablas del módulo RAT\n")
    out.append(f"Generado por `rat_scripts/build_summaries.py` · N = {meta['n_docs']} discursos · "
               f"vocabulario = {meta['vocab_size']} términos.\n")

    out.append("## 1. Parámetros de calibración\n")
    out.append(markdown_table([{"parámetro": k, "valor": v} for k, v in meta["params"].items()],
                              columns=["parámetro", "valor"]))

    out.append("\n## 2. Referencias escritas a mano (gold)\n")
    out.append(gold_markdown(gold))

    out.append("\n## 3. ROUGE por método (media de los 5 discursos)\n")
    out.append(markdown_table(
        agg,
        columns=["method", "rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure",
                 "compression", "coverage", "redundancy", "mean_sent_words"],
        headers=["método", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "compresión",
                 "cobertura", "redundancia", "pal/oración"],
    ))

    out.append("\n## 4. Híbrido vs línea base, por discurso\n")
    per_doc = []
    for item in gold:
        block = {r["method"]: r for r in rows if r["doc_id"] == item["doc_id"]}
        per_doc.append({
            "doc_id": item["doc_id"],
            "hablante": block["hybrid"]["speaker"],
            "lead R1": block["lead"]["rouge1_fmeasure"],
            "híbrido R1": block["hybrid"]["rouge1_fmeasure"],
            "híbrido R2": block["hybrid"]["rouge2_fmeasure"],
            "híbrido RL": block["hybrid"]["rougeL_fmeasure"],
            "cobertura": block["hybrid"]["coverage"],
        })
    out.append(markdown_table(per_doc))

    out.append("\n## 5. Ablación del híbrido\n")
    out.append(markdown_table(abl, headers=["variante", "ROUGE-1", "ROUGE-2", "ROUGE-L", "cobertura"]))

    out.append("\n## 6. Resúmenes de ejemplo (método híbrido)\n")
    for summary in demo:
        stats = summary.stats
        out.append(f"### doc_id {summary.doc_id} — {summary.speaker} · {summary.date}\n")
        out.append(f"*{summary.title}*\n")
        out.append(f"Unidades: {stats['n_units']} · seleccionadas: {stats['n_selected']} · "
                   f"compresión: {stats['compression']:.1%} · cobertura: {stats['coverage']:.2f} · "
                   f"redundancia: {stats['redundancy']:.2f}\n")
        out.append(f"> {summary.text}\n")

    out.append(f"\n## 7. Resumen multi-documento ({MULTI_SPEAKER}, {MULTI_YEAR})\n")
    out.append(f"Discursos usados ({len(multi_docs)}): {', '.join(str(d) for d in multi_docs)}. "
               f"Criterio: los más cercanos al centroide del conjunto.\n")
    stats = multi.stats
    out.append(f"Seleccionadas: {stats['n_selected']} de {stats['n_units']} oraciones fusionadas · "
               f"compresión sobre el total: {stats['compression']:.2%} · cobertura: {stats['coverage']:.2f}\n")
    out.append(f"> {multi.text}\n")
    out.append("Tópicos comunes (stem, nº de discursos, frecuencia total):\n")
    out.append(markdown_table(
        [{"término": t, "discursos": d, "frecuencia": int(f)} for t, d, f in multi.common_terms],
        headers=["término", "discursos", "frecuencia"],
    ))
    return "\n".join(out) + "\n"


def main() -> None:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ix = load_index()
    gold = load_gold()

    rows = run_benchmark(ix, gold, seeds=3)
    agg = aggregate(rows)
    abl = ablation(ix, gold)

    demo = [summarize_doc(item["doc_id"], ix=ix) for item in gold]

    docs = [i for i in range(len(ix.speakers))
            if ix.speakers[i] == MULTI_SPEAKER and ix.dates[i].startswith(MULTI_YEAR)]
    multi = summarize_docs(docs, ix=ix, k_docs=MULTI_DOCS) if docs else None

    meta = {
        "n_docs": ix.n_docs,
        "vocab_size": ix.vocab_size,
        "params": {
            "MIN_SENT_WORDS": config.MIN_SENT_WORDS,
            "MAX_SENT_WORDS": config.MAX_SENT_WORDS,
            "SUMMARY_RATIO": config.SUMMARY_RATIO,
            "MIN_SENTENCES": config.MIN_SENTENCES,
            "MAX_SENTENCES": config.MAX_SENTENCES,
            "WORD_BUDGET": config.WORD_BUDGET,
            "MMR_LAMBDA": config.MMR_LAMBDA,
            "DUP_JACCARD": config.DUP_JACCARD,
            "PAGERANK_DAMPING": config.PAGERANK_DAMPING,
            "LEXRANK_THRESHOLD": config.LEXRANK_THRESHOLD,
            "LSA_K": config.LSA_K,
            "WEIGHTS": config.WEIGHTS,
            "DISCOURSE_WEIGHTS": config.DISCOURSE_WEIGHTS,
            "idf": "global del índice RI (min_df=%d)" % ri_config.MIN_DF,
        },
    }

    # --- eval.json -----------------------------------------------------------
    payload = {
        "meta": meta,
        "gold": gold,
        "per_doc": rows,
        "aggregate": agg,
        "ablation": abl,
        "multi_doc": {
            "speaker": MULTI_SPEAKER,
            "year": MULTI_YEAR,
            "doc_ids": multi.stats.get("doc_ids") if multi else [],
            "stats": multi.stats if multi else {},
            "summary": multi.text if multi else "",
            "common_terms": [{"term": t, "docs": d, "tf": int(f)}
                             for t, d, f in (multi.common_terms if multi else [])],
        },
    }
    with open(config.EVAL_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=_json_default)

    # --- summaries_demo.jsonl ------------------------------------------------
    with open(config.DEMO_JSONL, "w", encoding="utf-8") as fh:
        for summary in demo:
            record = summary.to_dict()
            record["component_scores"] = summary.component_scores
            fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    # --- tables.md -----------------------------------------------------------
    tables = build_tables(meta, gold, rows, agg, abl, demo, multi,
                          multi.stats.get("doc_ids", []) if multi else [])
    config.TABLES_MD.write_text(tables, encoding="utf-8")

    print(f"docs evaluados: {len(gold)}  ·  métodos: {len(agg)}  ·  {time.time() - t0:.1f}s")
    print(f"artefactos: {config.EVAL_JSON.name}, {config.DEMO_JSONL.name}, {config.TABLES_MD.name}")
    best = max(agg, key=lambda r: r["rougeL_fmeasure"])
    print(f"mejor método por ROUGE-L: {best['method']} "
          f"(R1={best['rouge1_fmeasure']:.3f} R2={best['rouge2_fmeasure']:.3f} "
          f"RL={best['rougeL_fmeasure']:.3f})")


if __name__ == "__main__":
    main()
