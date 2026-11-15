"""Métricas y tablas del módulo 7-lite."""

import numpy as np
import pandas as pd

from rat_scripts.evaluate import markdown_table

BLOCK_KIND = {
    "estilo": "forma",
    "funcion": "forma",
    "estilo_controlado": "forma (controlado)",
    "caracteres": "diagnóstico (fuga de contenido)",
    "caracteres_enmascarados": "diagnóstico (fuga de contenido)",
    "tema": "contenido",
    "fecha": "periodo",
    "todo": "cota superior",
}

PROFILE_FEATURES = [
    ("n_words", "palabras"),
    ("words_per_sent", "palabras/oración"),
    ("sd_words_per_sent", "σ palabras/oración"),
    ("short_sent_ratio", "oraciones ≤8 palabras"),
    ("ttr", "TTR"),
    ("mtld", "MTLD"),
    ("yules_k", "Yule's K"),
    ("hapax_ratio", "háрах"),
    ("mean_word_len", "letras/palabra"),
    ("long_word_ratio", "palabras ≥7 letras"),
    ("comas", "comas/1000"),
    ("interrogacion", "? /1000"),
    ("exclamacion", "! /1000"),
    ("uppercase_ratio", "mayúsculas no iniciales"),
    ("first_sing_1000", "yo/1000"),
    ("first_plur_1000", "nosotros/1000"),
    ("yo_nosotros", "(yo−nosotros)/(yo+nosotros)"),
    ("orality_1000", "muletillas/1000"),
    ("vocatives_1000", "vocativos/1000"),
    ("hedges_1000", "atenuadores/1000"),
    ("fernandez_huerta", "Fernández-Huerta"),
    ("frase_o_sea_1000", "\"o sea\"/1000"),
]


def ablation_table(rows: list) -> str:
    out = [{
        "bloque": r["block"],
        "tipo": BLOCK_KIND.get(r["block"], ""),
        "modelo": {"svm": "SVM", "nb": "NB", "lr": "LR"}[r["model"]],
        "macro-F1": round(r["macro_f1"], 3),
        "± σ folds": round(r["fold_std"], 3),
        "accuracy (OOF)": round(r["accuracy"], 3),
        "seg": r["seconds"],
    } for r in rows]
    return markdown_table(out)


def per_class_table(result: dict, short=lambda s: s.split()[0]) -> str:
    out = [{
        "clase": short(row["label"]),
        "precisión": round(row["precision"], 3),
        "recall": round(row["recall"], 3),
        "F1": round(row["f1"], 3),
        "docs": row["support"],
    } for row in result["per_class"]]
    return markdown_table(out)


def confusion_table(result: dict, short=lambda s: s.split()[0]) -> str:
    labels = result["labels"]
    rows = []
    for i, label in enumerate(labels):
        row = {"real \\ pred": short(label)}
        for j, other in enumerate(labels):
            row[short(other)] = int(result["confusion"][i][j])
        rows.append(row)
    cols = ["real \\ pred"] + [short(l) for l in labels]
    return markdown_table(rows, columns=cols, headers=cols)


def learning_table(rows: list) -> str:
    out = [{"fracción de train": f"{r['fraction']:.0%}",
            "macro-F1": round(r["macro_f1"], 3),
            "± σ": round(r["f1_std"], 3)} for r in rows]
    return markdown_table(out)


def profile_table(style: pd.DataFrame) -> str:
    grouped = style.groupby("speaker")
    rows = []
    for feature, name in PROFILE_FEATURES:
        row = {"rasgo": name}
        for speaker, group in grouped:
            row[speaker.split()[0]] = round(float(group[feature].median()), 3)
        rows.append(row)
    cols = ["rasgo"] + [s.split()[0] for s in grouped.groups.keys()]
    return markdown_table(rows, columns=cols, headers=cols)


def distinctive_table(style: pd.DataFrame, n: int = 4) -> str:
    """Rasgos más distintivos de cada hablante (z de la mediana respecto al corpus)."""
    features = [f for f, _ in PROFILE_FEATURES]
    names = dict(PROFILE_FEATURES)
    global_mean = style[features].mean()
    global_std = style[features].std().replace(0, 1.0)
    rows = []
    for speaker, group in style.groupby("speaker"):
        z = (group[features].mean() - global_mean) / global_std
        top = z.sort_values(ascending=False).head(n)
        bottom = z.sort_values().head(n)
        rows.append({
            "hablante": speaker.split()[0],
            "más alto": ", ".join(f"{names[f]} ({z[f]:+.1f}σ)" for f in top.index),
            "más bajo": ", ".join(f"{names[f]} ({z[f]:+.1f}σ)" for f in bottom.index),
        })
    return markdown_table(rows, columns=["hablante", "más alto", "más bajo"],
                          headers=["hablante", "más alto", "más bajo"])


def window_table(rows: list) -> str:
    out = [{
        "bloque": r["block"],
        "tipo": BLOCK_KIND.get(r["block"], ""),
        "macro-F1": round(r["macro_f1"], 3),
        "± σ folds": round(r["fold_std"], 3),
        "clases": len(r["labels"]),
        "docs": r["n_docs"],
    } for r in rows]
    return markdown_table(out, columns=["bloque", "tipo", "macro-F1", "± σ folds", "clases", "docs"],
                          headers=["bloque", "tipo", "macro-F1", "± σ folds", "clases", "docs"])


def summary_row(name: str, result: dict) -> dict:
    return {"caso": name, "macro-F1": round(result["macro_f1"], 3),
            "± σ": round(result["fold_f1_std"], 3),
            "accuracy": round(result["oof_accuracy"], 3)}


def baseline_majority(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())
