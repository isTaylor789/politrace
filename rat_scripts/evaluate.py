"""Evaluación de resúmenes (temario 4.5) y comparación de métodos.

Se implementan **ROUGE-N** y **ROUGE-L** sobre términos analizados (stem, sin
stopwords) con conteo recortado (*clipping*), la definición de Lin (2004):

    ROUGE-N(p, r) = Σ_{g ∈ n-gramas(r)} min(c_p(g), c_r(g))
    P = ROUGE-N / |n-gramas(p)| ,  R = ROUGE-N / |n-gramas(r)|
    F1 = 2PR / (P + R)

y para ROUGE-L, la subsecuencia común más larga (LCS) de las dos secuencias de
términos. La referencia es un resumen **escrito a mano** (`gold/`), más dos
líneas base: las primeras k oraciones del discurso y k oraciones al azar.

Además de ROUGE se reportan métricas intrínsecas que no necesitan referencia:
compresión, cobertura del tópico (peso TF-IDF de los 20 conceptos principales
que sí aparece en el resumen), redundancia media y longitud media de oración.
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from ri_scripts.preprocess import analyze

from rat_scripts import config
from rat_scripts.select import render, target_sentences
from rat_scripts.summarize import METHODS, summarize_doc
from rat_scripts.units import build_units


# --- ROUGE -------------------------------------------------------------------
def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def rouge_n(candidate: str, reference: str, n: int = 1) -> dict:
    """ROUGE-N con conteo recortado sobre términos analizados."""
    cand, ref = analyze(candidate), analyze(reference)
    c_ng, r_ng = _ngrams(cand, n), _ngrams(ref, n)
    overlap = sum(min(count, r_ng[g]) for g, count in c_ng.items())
    precision = overlap / max(1, sum(c_ng.values()))
    recall = overlap / max(1, sum(r_ng.values()))
    return {"precision": precision, "recall": recall, "fmeasure": _f1(precision, recall)}


def lcs_length(a: list, b: list) -> int:
    """Longitud de la subsecuencia común más larga (programación dinámica)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> dict:
    """ROUGE-L (F1) basado en la LCS de términos."""
    cand, ref = analyze(candidate), analyze(reference)
    lcs = lcs_length(cand, ref)
    precision = lcs / max(1, len(cand))
    recall = lcs / max(1, len(ref))
    return {"precision": precision, "recall": recall, "fmeasure": _f1(precision, recall)}


def evaluate_summary(summary, reference: str) -> dict:
    """ROUGE-1/2/L + métricas intrínsecas de un resumen frente a su referencia."""
    out = {}
    for name, n in (("rouge1", 1), ("rouge2", 2)):
        metrics = rouge_n(summary.text, reference, n)
        for key, value in metrics.items():
            out[f"{name}_{key}"] = value
    metrics = rouge_l(summary.text, reference)
    for key, value in metrics.items():
        out[f"rougeL_{key}"] = value

    out["compression"] = summary.stats.get("compression", 0.0)
    out["coverage"] = summary.stats.get("coverage", 0.0)
    out["redundancy"] = summary.stats.get("redundancy", 0.0)
    out["mean_sent_words"] = summary.stats.get("mean_sent_words", 0.0)
    out["n_selected"] = summary.stats.get("n_selected", 0)
    return out


# --- Líneas base -------------------------------------------------------------
def _random_summary(text: str, k: int, seed: int = 0) -> str:
    units = build_units(text)
    if not units:
        return ""
    rng = random.Random(seed)
    k = min(k, len(units))
    chosen = sorted(rng.sample(range(len(units)), k))
    return render(units, chosen)


# --- Banco de pruebas --------------------------------------------------------
def load_gold(path: Path = None) -> list:
    path = config.GOLD_JSON if path is None else Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["items"] if isinstance(data, dict) else data


def run_benchmark(ix, gold: list = None, methods: list = None, *,
                  seeds: int = 3, extra: dict = None) -> list:
    """Evalúa cada método sobre el conjunto de referencias.

    Devuelve una lista de filas: una por (documento, método), lista para
    convertir a DataFrame. Las líneas base `lead` y `random` usan el mismo
    número de oraciones que el método híbrido, para que la comparación sea justa.
    """
    from ri_scripts.corpus import read_doc

    gold = load_gold() if gold is None else gold
    methods = METHODS if methods is None else methods
    extra = extra or {}
    rows = []

    for item in gold:
        doc_id = int(item["doc_id"])
        doc = read_doc(doc_id, ix.corpus_path, ix.offsets)
        reference = item["reference"]

        for method in methods:
            summary = summarize_doc(doc_id, ix=ix, method=method, **extra)
            metrics = evaluate_summary(summary, reference)
            metrics.update(doc_id=doc_id, speaker=doc["speaker"], date=doc["date"],
                           method=method)
            rows.append(metrics)

        # línea base aleatoria: se promedia sobre varias semillas
        k = target_sentences(len(build_units(doc["text"])),
                             ratio=extra.get("ratio"),
                             max_sentences=extra.get("max_sentences"))
        rand = []
        for seed in range(seeds):
            text = _random_summary(doc["text"], k, seed=seed)
            rand.append({
                "rouge1_fmeasure": rouge_n(text, reference, 1)["fmeasure"],
                "rouge2_fmeasure": rouge_n(text, reference, 2)["fmeasure"],
                "rougeL_fmeasure": rouge_l(text, reference)["fmeasure"],
            })
        rows.append({
            "doc_id": doc_id, "speaker": doc["speaker"], "date": doc["date"],
            "method": "random",
            "rouge1_fmeasure": float(np.mean([r["rouge1_fmeasure"] for r in rand])),
            "rouge2_fmeasure": float(np.mean([r["rouge2_fmeasure"] for r in rand])),
            "rougeL_fmeasure": float(np.mean([r["rougeL_fmeasure"] for r in rand])),
            "compression": 0.0, "coverage": 0.0, "redundancy": 0.0,
            "mean_sent_words": 0.0, "n_selected": k,
        })
    return rows


def aggregate(rows: list, methods: list = None) -> list:
    """Media y desviación estándar por método (las filas ya vienen por doc)."""
    methods = methods or list(dict.fromkeys(r["method"] for r in rows))
    metrics = ["rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure",
               "compression", "coverage", "redundancy", "mean_sent_words"]
    out = []
    for method in methods:
        block = [r for r in rows if r["method"] == method]
        if not block:
            continue
        row = {"method": method, "n_docs": len(block)}
        for m in metrics:
            values = [r.get(m, 0.0) for r in block]
            row[m] = float(np.mean(values))
            row[f"{m}_std"] = float(np.std(values))
        out.append(row)
    return out


def gold_markdown(items: list, path: Path = None) -> str:
    """Tabla de las referencias escritas a mano (para el reporte)."""
    lines = [
        "| doc_id | hablante | fecha | palabras ref. | referencia |",
        "|---:|---|---|---:|---|",
    ]
    for item in items:
        ref = item["reference"].replace("|", "\\|")
        lines.append(
            f"| {item['doc_id']} | {item['speaker']} | {item['date']} | "
            f"{len(ref.split())} | {ref} |"
        )
    return "\n".join(lines) + "\n"


def markdown_table(rows: list, columns: list = None, headers: list = None) -> str:
    """Tabla markdown simple a partir de una lista de dicts."""
    columns = columns or list(rows[0].keys())
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"
