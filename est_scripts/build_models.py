"""CLI del módulo 7-lite: perfiles estilométricos + clasificación de hablante + ablación.

    source ../env/bin/activate
    python est_scripts/build_models.py            # usa la caché de features (~2-4 min)
    python est_scripts/build_models.py --force    # recalcula features (~5 min)
    python est_scripts/build_models.py --skip-window

Escribe en `est_scripts/artifacts/`:

- ``style_profiles.csv``   rasgos estilométricos por documento
- ``eval.json``            ablación, mejor modelo, ventana, curva de aprendizaje
- ``oof_predictions.jsonl`` predicciones out-of-fold del mejor modelo controlado
- ``tables.md``            tablas del reporte
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import est_scripts.config as config  # noqa: E402
from est_scripts import classify as C  # noqa: E402
from est_scripts import evaluate as E  # noqa: E402
from est_scripts.features import build_dataset, window_mask  # noqa: E402

# Bloques y modelos. NB no se aplica a bloques mixtos (contienen rasgos con signo).
MODELS_BY_BLOCK = {
    "estilo": ["svm", "nb", "lr"],
    "funcion": ["svm", "nb", "lr"],
    "estilo_controlado": ["svm", "lr"],
    "caracteres": ["svm"],
    "caracteres_enmascarados": ["svm", "nb"],
    "tema": ["svm", "nb"],
    "fecha": ["svm", "nb"],
    "todo": ["svm"],
}
CONTROLLED = ("estilo", "funcion", "estilo_controlado")


def _subset_blocks(blocks: dict, mask: np.ndarray) -> dict:
    return {name: mat[mask] for name, mat in blocks.items()}


def _jsonable(obj):
    """Convierte tipos de NumPy a tipos nativos para json."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    return obj


def build_tables(payload: dict) -> str:
    meta = payload["meta"]
    best = payload["best_controlled"]
    parts = [
        "# Tablas — estilometría y clasificación de hablante (módulo 7-lite)\n",
        "Generado por `est_scripts/build_models.py`.\n",
        "## 0. Corpus y validación\n",
        E.markdown_table(
            [{
                "documentos": meta["n_docs"],
                "descartados (<120 palabras)": meta["n_dropped"],
                "grupos de casi-duplicados": meta["n_duplicate_groups"],
                "mayor grupo": meta["largest_group"],
                "folds": config.N_SPLITS,
                "umbral de duplicado (coseno)": config.DUP_COSINE,
                "accuracy de la clase mayoritaria": round(meta["majority_accuracy"], 3),
            }],
            columns=["documentos", "descartados (<120 palabras)", "grupos de casi-duplicados",
                     "mayor grupo", "folds", "umbral de duplicado (coseno)",
                     "accuracy de la clase mayoritaria"],
            headers=["documentos", "descartados (<120 palabras)", "grupos de casi-duplicados",
                     "mayor grupo", "folds", "umbral de duplicado (coseno)",
                     "accuracy de la clase mayoritaria"],
        ),
        "La clase mayoritaria es AMLO (2,293 de 4,525): **accuracy no es una métrica útil aquí**; "
        "todo se reporta como macro-F1 y por clase.\n",
        "## 1. Ablación: ¿qué aprende el clasificador?\n",
        E.ablation_table(payload["ablation"]),
        "`forma` = rasgos que no dependen del tema (riqueza, puntuación, deixis, palabras de función). "
        "`diagnóstico` = n-gramas de carácter, que **sí** filtran contenido (nombres, lugares, "
        "consignas). `contenido` = TF-IDF de tema del módulo 6. `periodo` = año/mes/trimestre. "
        "`cota superior` = forma + contenido + periodo.\n",
        f"**Mejor bloque controlado**: `{best['block']}` con {C.MODEL_LABEL[best['model']]} "
        f"(macro-F1 = {best['macro_f1']:.3f} ± {best['fold_std']:.3f}).\n",
        "## 2. Mejor modelo controlado: por clase y matriz de confusión\n",
        E.per_class_table(payload["best_result"]),
        E.confusion_table(payload["best_result"]),
        "## 3. Control de ventana 2023-2024 (3 líderes con ≥ 30 discursos)\n",
        E.window_table(payload["window"]),
        "En la ventana limpia el periodo deja de identificar; si el F1 de `tema` sigue alto, el "
        "clasificador usa de qué habla cada líder, no solo cuándo gobernó.\n",
        "## 4. Curva de aprendizaje (bloque controlado)\n",
        E.learning_table(payload["learning"]),
        "## 5. Perfil estilométrico por líder (medianas)\n",
        E.profile_table(payload["dataset"].style),
        "## 6. Rasgos más distintivos de cada líder (z de la media respecto al corpus)\n",
        E.distinctive_table(payload["dataset"].style),
        "Son rasgos de **texto**, no de personalidad: describen cómo se escribe/habla en el corpus "
        "(transcripciones), no cómo es la persona.\n",
    ]
    return "\n".join(parts)


def main(force: bool = False, skip_window: bool = False) -> None:
    t0 = time.time()
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("construyendo/leyendo features...")
    dataset = build_dataset(force=force)
    y = dataset.speakers
    groups = dataset.groups
    print(f"  {dataset.n_docs} documentos, {len(set(y))} clases, "
          f"{dataset.n_duplicate_groups} grupos de duplicados")

    print("\nablación de bloques (CV agrupada, 5 folds)...")
    ablation = C.run_ablation(dataset.blocks, y, groups, MODELS_BY_BLOCK)

    best_controlled = max((r for r in ablation if r["block"] in CONTROLLED),
                          key=lambda r: r["macro_f1"])
    best_result = C.cv_run(dataset.blocks[best_controlled["block"]], y, groups,
                           best_controlled["model"])
    print(f"\nmejor controlado: {best_controlled['block']} / {best_controlled['model']} "
          f"→ macro-F1={best_result['macro_f1']:.3f}")

    print("\ncurva de aprendizaje...")
    learning = C.learning_curve(dataset.blocks[best_controlled["block"]], y, groups,
                                best_controlled["model"], block=best_controlled["block"])

    window_rows = []
    if not skip_window:
        print("\ncontrol de ventana 2023-2024...")
        mask = window_mask(dataset)
        y_w = y[mask]
        groups_w = dataset.groups[mask]
        blocks_w = _subset_blocks(dataset.blocks, mask)
        print(f"  {int(mask.sum())} documentos, clases: "
              f"{ {s: int((y_w == s).sum()) for s in sorted(set(y_w))} }")
        for block, model in (("estilo_controlado", "svm"), ("caracteres_enmascarados", "svm"),
                             ("tema", "svm"), ("fecha", "svm")):
            result = C.cv_run(blocks_w[block], y_w, groups_w, model, block=block)
            window_rows.append({
                "block": block, "model": model,
                "macro_f1": result["macro_f1"], "fold_std": result["fold_f1_std"],
                "labels": result["labels"], "n_docs": int(mask.sum()),
                "per_class": result["per_class"],
            })
            print(f"  {block:24s} macro-F1={result['macro_f1']:.3f}")

    meta = {
        "n_docs": int(dataset.n_docs),
        "n_dropped": int(4531 - dataset.n_docs),
        "n_duplicate_groups": int(dataset.n_duplicate_groups),
        "largest_group": int(dataset.largest_group),
        "majority_accuracy": round(E.baseline_majority(y), 3),
        "params": {
            "min_words": config.MIN_WORDS, "folds": config.N_SPLITS,
            "dup_cosine": config.DUP_COSINE, "char_ngrams": list(config.CHAR_NGRAMS),
            "char_max_features": config.CHAR_MAX_FEATURES,
            "char_max_chars": config.CHAR_MAX_CHARS,
            "function_min_df": config.FUNCTION_MIN_DF,
            "window": list(config.WINDOW), "window_min_docs": config.WINDOW_MIN_DOCS,
            "seed": config.SEED,
        },
        "seconds": round(time.time() - t0, 1),
    }

    payload = {
        "meta": meta,
        "ablation": [{k: v for k, v in r.items() if k not in ("per_class", "confusion", "labels")}
                     for r in ablation],
        "best_controlled": {k: best_controlled[k] for k in
                            ("block", "model", "macro_f1", "fold_std", "accuracy", "seconds")},
        "best_result": best_result,
        "window": window_rows,
        "learning": learning,
        "dataset": dataset,
    }

    config.STYLE_PROFILES_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.style.to_csv(config.STYLE_PROFILES_CSV, index=False)
    print(f"  -> {config.STYLE_PROFILES_CSV.name}")

    with open(config.EVAL_JSON, "w", encoding="utf-8") as fh:
        json.dump({
            "meta": meta,
            "ablation": payload["ablation"],
            "best_controlled": payload["best_controlled"],
            "best_result": {k: v for k, v in best_result.items() if k != "oof"},
            "window": window_rows,
            "learning": learning,
        }, fh, ensure_ascii=False, indent=2, default=_jsonable)
    print(f"  -> {config.EVAL_JSON.name}")

    with open(config.OOF_JSONL, "w", encoding="utf-8") as fh:
        for doc_id, real, pred in zip(dataset.doc_ids, y, best_result["oof"]):
            fh.write(json.dumps({
                "doc_id": int(doc_id), "real": str(real), "pred": str(pred),
                "correct": bool(real == pred),
            }, ensure_ascii=False) + "\n")
    print(f"  -> {config.OOF_JSONL.name}")

    with open(config.TABLES_MD, "w", encoding="utf-8") as fh:
        fh.write(build_tables(payload))
    print(f"  -> {config.TABLES_MD.name}")

    print(f"\ntotal: {time.time() - t0:.1f}s")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recalcula features (5 min)")
    parser.add_argument("--skip-window", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force, skip_window=args.skip_window)
