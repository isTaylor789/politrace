"""Genera los artefactos del módulo 5: resumen multi-documento por año.

    python rat_scripts/build_multidoc.py

Escribe en `rat_scripts/artifacts/multidoc/`:

- ``eval_multidoc.json``    métricas por caso y variante + resúmenes
- ``summaries_year.jsonl``  resumen base de cada caso, con grupos y oraciones
- ``tables_multidoc.md``    tablas del reporte
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ri_scripts.index import load_index  # noqa: E402
from rat_scripts import config  # noqa: E402
from rat_scripts.evaluate import markdown_table  # noqa: E402
from rat_scripts.evaluate_multidoc import (aggregate, build_tables,  # noqa: E402
                                           load_year_gold, run_benchmark)
from rat_scripts.multi import year_availability  # noqa: E402

ARTIFACTS = config.ARTIFACTS / "multidoc"


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"no serializable: {type(value)}")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ix = load_index()
    items = load_year_gold()

    rows, base = run_benchmark(ix, items, return_summaries=True)
    agg = aggregate(rows)
    summaries = [summary for _, summary in base]

    availability = [{"hablante": r["speaker"], "año": r["year"], "discursos": r["docs"]}
                    for r in year_availability(ix)
                    if r["docs"] >= config.YEAR_MIN_DOCS]

    meta = {
        "params": {
            "YEAR_MIN_DOCS": config.YEAR_MIN_DOCS,
            "CLUSTER_MIN/MAX": [config.CLUSTER_MIN, config.CLUSTER_MAX],
            "CLUSTER_SEED": config.CLUSTER_SEED,
            "GENERIC_DF_RATIO": config.GENERIC_DF_RATIO,
            "DOCS_PER_GROUP": config.DOCS_PER_GROUP,
            "PER_DOC_SENTENCES": config.PER_DOC_SENTENCES,
            "MULTI_MAX_SENTENCES": config.MULTI_MAX_SENTENCES,
            "MULTI_WORD_BUDGET": config.MULTI_WORD_BUDGET,
            "MULTI_IDF_MODE": config.MULTI_IDF_MODE,
            "IDF_BLEND_ALPHA": config.IDF_BLEND_ALPHA,
            "MULTI_YEAR_WEIGHT": config.MULTI_YEAR_WEIGHT,
            "MMR_LAMBDA": config.MMR_LAMBDA,
        },
        "n_corpus": ix.n_docs,
    }

    payload = {
        "meta": meta,
        "gold": items,
        "per_case": rows,
        "aggregate": agg,
        "summaries": [{
            "speaker": s.speaker,
            "year": s.date,
            "title": s.title,
            "text": s.text,
            "stats": s.stats,
            "groups": s.groups,
            "common_terms": [{"term": t, "groups": d, "weight": w}
                             for t, d, w in s.common_terms],
            "sentences": s.sentences,
        } for s in summaries],
    }
    with open(ARTIFACTS / "eval_multidoc.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=_json_default)

    with open(ARTIFACTS / "summaries_year.jsonl", "w", encoding="utf-8") as fh:
        for s in summaries:
            fh.write(json.dumps({
                "speaker": s.speaker,
                "year": s.date,
                "text": s.text,
                "stats": s.stats,
                "groups": s.groups,
                "common_terms": s.common_terms,
            }, ensure_ascii=False, default=_json_default) + "\n")

    tables = build_tables(rows, agg, items, summaries)
    headers = ["hablante", "año", "discursos"]
    extra = "\n## 0. Años con material suficiente (≥ %d discursos)\n" % config.YEAR_MIN_DOCS
    extra += markdown_table(availability, headers=headers)
    tables = tables.replace("\n## 1. Casos de referencia", extra + "\n## 1. Casos de referencia")
    (ARTIFACTS / "tables_multidoc.md").write_text(tables, encoding="utf-8")

    print(f"casos: {len(items)} · variantes: {len(agg)} · {time.time() - t0:.1f}s")
    print(f"artefactos en {ARTIFACTS}")
    best = max(agg, key=lambda r: r["topic_recall"])
    print(f"mejor variante por recall de tópicos: {best['variant']} "
          f"({best['topic_recall']:.3f}); ROUGE-1 = {best['rouge1']:.3f}")


if __name__ == "__main__":
    main()
