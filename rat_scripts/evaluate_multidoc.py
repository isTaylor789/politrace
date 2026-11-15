"""Evaluación del resumen multi-documento por año (temario 4.6).

Métricas:

* **ROUGE-1/2/L** contra un resumen de referencia escrito a mano.
* **Recall de tópicos clave**: cada caso tiene 7-8 tópicos; se cuenta cuántos
  aparecen en el resumen (hit si contiene alguno de sus stems). Es la métrica
  más informativa para un resumen de un año entero, porque ROUGE castiga mucho
  la paráfrasis y el resumen es extractivo.
* **Similitud con el centroide del año** y **cobertura temática** (proporción de
  documentos del año cuyo grupo temático está representado).
* Compresión, redundancia y tamaño.

Se comparan variantes del pipeline: IDF global / mezclado / local, con y sin
agrupamiento, con y sin cuotas, y una selección aleatoria como suelo.
"""

import json
from pathlib import Path

import numpy as np

from ri_scripts.preprocess import analyze

from rat_scripts import config
from rat_scripts.evaluate import markdown_table, rouge_l, rouge_n
from rat_scripts.multi import fuse_year, prepare_year
from rat_scripts.select import render
from rat_scripts.summarize import Summary

GOLD_JSON = config.PKG_DIR / "gold" / "year_gold.json"

# cada variante: (nombre, params de fuse_year, params extra de prepare_year)
VARIANTS = [
    ("base (global)", {"idf_mode": "global"}, {}),
    ("idf blend", {"idf_mode": "blend"}, {}),
    ("idf local", {"idf_mode": "local"}, {}),
    ("sin agrupamiento", {"idf_mode": "global"}, {"n_clusters": 1}),
    ("sin cuotas", {"idf_mode": "global", "use_quotas": False}, {}),
    ("cuotas min-1", {"idf_mode": "global", "quota_min_share": 0.0,
                      "quota_min_raw": 0.25}, {}),
]


# --- Gold --------------------------------------------------------------------
def load_year_gold(path: Path = None) -> list:
    path = GOLD_JSON if path is None else Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["items"]


def key_topic_recall(summary_text: str, topics: list) -> tuple:
    """(recall, lista de (etiqueta, cubierto)) para los tópicos clave.

    Un tópico puede exigir varios stems a la vez (`all`) y aceptar alternativas
    (`any`): `{"all": ["europe"], "any": ["rotatori", "consej"]}`.
    """
    terms = set(analyze(summary_text))
    hits = []
    for topic in topics:
        covered = all(stem in terms for stem in topic.get("all", []))
        alternatives = topic.get("any", [])
        if alternatives:
            covered = covered and any(stem in terms for stem in alternatives)
        hits.append((topic["label"], covered))
    recall = sum(1 for _, ok in hits if ok) / max(1, len(hits))
    return recall, hits


# --- Evaluación --------------------------------------------------------------
def evaluate_year(summary: Summary, item: dict) -> dict:
    reference = item["reference"]
    recall, hits = key_topic_recall(summary.text, item["key_topics"])
    stats = summary.stats
    out = {
        "speaker": item["speaker"],
        "year": item["year"],
        "rouge1": rouge_n(summary.text, reference, 1)["fmeasure"],
        "rouge2": rouge_n(summary.text, reference, 2)["fmeasure"],
        "rougeL": rouge_l(summary.text, reference)["fmeasure"],
        "topic_recall": recall,
        "topics_hit": {label: ok for label, ok in hits},
        "centroid_similarity": stats.get("coverage", 0.0),
        "theme_coverage": stats.get("theme_coverage", 0.0),
        "compression": stats.get("compression", 0.0),
        "redundancy": stats.get("redundancy", 0.0),
        "n_selected": stats.get("n_selected", 0),
        "words_summary": stats.get("words_summary", 0),
        "n_groups": stats.get("n_groups", 0),
        "docs_represented": stats.get("docs_represented", 0),
        "summary": summary.text,
    }
    return out


def random_year_summary(ix, fused, *, seed: int = 0, word_budget: int = None) -> Summary:
    """Línea base: oraciones al azar del mismo material fusionado."""
    from rat_scripts.multi import _redundancy, _year_coverage
    from rat_scripts.representation import build_representation

    word_budget = config.MULTI_WORD_BUDGET if word_budget is None else word_budget
    rng = np.random.default_rng(seed)
    n = len(fused.units)
    order = rng.permutation(n)
    chosen, words = [], 0
    for i in order:
        if words + fused.units[i].n_words > word_budget and chosen:
            continue
        chosen.append(int(i))
        words += fused.units[i].n_words
    chosen.sort()

    rep = build_representation(fused.units, idf_mode=config.MULTI_IDF_MODE)
    group_counts = {}
    for i in chosen:
        gid = fused.units[i].group_id
        group_counts[gid] = group_counts.get(gid, 0) + 1
    stats = {
        "n_selected": len(chosen),
        "words_summary": words,
        "compression": words / max(1, fused.words_original),
        "redundancy": _redundancy(rep, chosen),
        "coverage": _year_coverage(ix, fused, fused.units, chosen),
        "theme_coverage": (sum(g.size for g in fused.groups
                               if group_counts.get(g.group_id, 0) > 0)
                           / max(1, len(fused.doc_ids))),
        "n_groups": len(fused.groups),
        "docs_represented": len({fused.units[i].doc_id for i in chosen}),
    }
    return Summary(text=render(fused.units, chosen), method="random",
                   stats=stats, sentences=[])


def run_benchmark(ix, items: list = None, variants: list = None, *,
                  seeds: int = 3, return_summaries: bool = False):
    """Evalúa cada variante sobre los casos de referencia.

    Con `return_summaries=True` devuelve también `[(item, Summary)]` de la
    primera variante (la base), que es la que se guarda como ejemplo.
    """
    items = load_year_gold() if items is None else items
    variants = VARIANTS if variants is None else variants
    rows = []
    base_summaries = []

    for item in items:
        fused = prepare_year(ix, item["year"], item["speaker"])
        fused_cache = {}

        for name, fuse_params, prep_params in variants:
            if prep_params:
                key = tuple(sorted(prep_params.items()))
                if key not in fused_cache:
                    fused_cache[key] = prepare_year(
                        ix, item["year"], item["speaker"], **prep_params)
                fused_v = fused_cache[key]
            else:
                fused_v = fused
            summary = fuse_year(ix, fused_v, **fuse_params)
            row = evaluate_year(summary, item)
            row["variant"] = name
            rows.append(row)
            if name == variants[0][0]:
                base_summaries.append((item, summary))

        # línea base aleatoria (media de varias semillas)
        rnd = [evaluate_year(random_year_summary(ix, fused, seed=s), item)
               for s in range(seeds)]
        row = {
            "speaker": item["speaker"], "year": item["year"], "variant": "aleatorio",
            "rouge1": float(np.mean([r["rouge1"] for r in rnd])),
            "rouge2": float(np.mean([r["rouge2"] for r in rnd])),
            "rougeL": float(np.mean([r["rougeL"] for r in rnd])),
            "topic_recall": float(np.mean([r["topic_recall"] for r in rnd])),
            "centroid_similarity": float(np.mean([r["centroid_similarity"] for r in rnd])),
            "theme_coverage": float(np.mean([r["theme_coverage"] for r in rnd])),
            "compression": float(np.mean([r["compression"] for r in rnd])),
            "redundancy": float(np.mean([r["redundancy"] for r in rnd])),
            "n_selected": float(np.mean([r["n_selected"] for r in rnd])),
            "words_summary": float(np.mean([r["words_summary"] for r in rnd])),
            "n_groups": float(np.mean([r["n_groups"] for r in rnd])),
            "docs_represented": float(np.mean([r["docs_represented"] for r in rnd])),
            "topics_hit": {},
            "summary": "",
        }
        rows.append(row)
    if return_summaries:
        return rows, base_summaries
    return rows


def aggregate(rows: list, order: list = None) -> list:
    """Media por variante."""
    order = order or list(dict.fromkeys(r["variant"] for r in rows))
    metrics = ["rouge1", "rouge2", "rougeL", "topic_recall", "centroid_similarity",
               "theme_coverage", "compression", "redundancy", "words_summary",
               "n_groups", "docs_represented"]
    out = []
    for variant in order:
        block = [r for r in rows if r["variant"] == variant]
        if not block:
            continue
        row = {"variant": variant, "n_cases": len(block)}
        for m in metrics:
            row[m] = float(np.mean([r[m] for r in block]))
        out.append(row)
    return out


# --- Reporte -----------------------------------------------------------------
def gold_markdown(items: list) -> str:
    lines = ["| hablante | año | tópicos clave |", "|---|---|---|"]
    for item in items:
        topics = ", ".join(t["label"] for t in item["key_topics"])
        lines.append(f"| {item['speaker']} | {item['year']} | {topics} |")
    return "\n".join(lines) + "\n"


def build_tables(rows: list, agg: list, items: list, summaries: list) -> str:
    out = ["# Tablas del resumen multi-documento por año (módulo 5)\n"]
    out.append("Generado por `rat_scripts/build_multidoc.py`.\n")

    out.append("## 1. Casos de referencia\n")
    out.append(gold_markdown(items))

    out.append("\n## 2. Variantes del pipeline (media de los casos)\n")
    out.append(markdown_table(
        agg,
        columns=["variant", "rouge1", "rouge2", "rougeL", "topic_recall",
                 "centroid_similarity", "theme_coverage", "compression",
                 "redundancy", "words_summary"],
        headers=["variante", "ROUGE-1", "ROUGE-2", "ROUGE-L", "tópicos",
                 "sim. centroide", "cobertura temas", "compresión",
                 "redundancia", "palabras"],
    ))

    out.append("\n## 3. Detalle por caso (variante base)\n")
    detail = []
    for row in rows:
        if row["variant"] != "base (global)":
            continue
        detail.append({
            "caso": f"{row['speaker']} {row['year']}",
            "grupos": int(row["n_groups"]),
            "docs": row["docs_represented"],
            "oraciones": row["n_selected"],
            "palabras": row["words_summary"],
            "ROUGE-1": row["rouge1"],
            "ROUGE-L": row["rougeL"],
            "tópicos": row["topic_recall"],
            "temas": row["theme_coverage"],
        })
    out.append(markdown_table(detail))

    out.append("\n## 4. Tópicos clave cubiertos (variante base)\n")
    topic_rows = []
    for row in rows:
        if row["variant"] != "base (global)":
            continue
        for label, ok in row["topics_hit"].items():
            topic_rows.append({
                "caso": f"{row['speaker']} {row['year']}",
                "tópico": label,
                "cubierto": "sí" if ok else "no",
            })
    out.append(markdown_table(topic_rows, headers=["caso", "tópico", "cubierto"]))

    out.append("\n## 5. Resúmenes (variante base)\n")
    for summary in summaries:
        stats = summary.stats
        out.append(f"### {summary.speaker} · {summary.date}\n")
        out.append(f"Grupos: {stats['n_groups']} · oraciones: {stats['n_selected']} · "
                   f"palabras: {stats['words_summary']} · "
                   f"compresión: {stats['compression']:.3%} · "
                   f"sim. centroide: {stats['coverage']:.2f} · "
                   f"temas cubiertos: {stats['theme_coverage']:.0%}\n")
        out.append(f"> {summary.text}\n")
        out.append("| grupo | docs | etiqueta | oraciones |")
        out.append("|---:|---:|---|---:|")
        for group in summary.groups:
            out.append(f"| {group['group_id']} | {group['size']} | "
                       f"{group['label']} | {group['selected']} |")
        out.append("")
    return "\n".join(out) + "\n"
