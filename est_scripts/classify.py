"""Clasificación supervisada y validación cruzada agrupada.

Dos decisiones metodológicas que son el aporte del módulo:

1. **Grupos de casi-duplicados en la CV.** Los discursos repetidos se agrupan y
   caen siempre en el mismo fold (``StratifiedGroupKFold``). Sin esto, el mismo
   texto en train y test infla el F1 y el resultado deja de ser una estimación de
   generalización.
2. **Ablación de bloques.** Se entrena con estilo, función, caracteres, tema y
   fecha por separado. Si el mejor F1 viene de "tema" o "fecha", el clasificador
   está detectando *de qué habla* el líder o *cuándo* gobernó, no *cómo escribe*.
   Esa comparación es la que convierte el confundido país = persona en un número.

Métrica principal: **macro-F1** (media no ponderada por clase), porque el corpus
está desbalanceado 2,293 : 30. Accuracy no se reporta como métrica principal.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import est_scripts.config as config

MODEL_LABEL = {"svm": "SVM lineal", "nb": "Naive Bayes", "lr": "Regresión logística"}


# Bloques de bolsa de n-gramas de alta dimensión: el solver dual es mucho más
# rápido (n_features >> n_samples y la matriz es muy dispersa). En el resto se
# usa el primal, que con 4.5k documentos converge mejor.
DUAL_BLOCKS = {"caracteres", "caracteres_enmascarados", "tema"}


def make_model(name: str, X, block: str = None) -> object:
    """Pipeline por familia y tipo de matriz.

    - ``svm`` y ``lr``: estandarización sin centrar en dispersas (centrar destruye
      la estructura CSR), ``class_weight='balanced'`` por el desbalance.
    - ``nb``: Naive Bayes complementario en bloques de conteo/TF-IDF (no negativos)
      y gaussiano en el bloque denso de rasgos agregados.
    """
    sparse = sp.issparse(X)
    if name == "svm":
        dual = block in DUAL_BLOCKS if block is not None else (
            sparse and X.shape[1] > 3 * X.shape[0])
        return Pipeline([
            ("scale", StandardScaler(with_mean=not sparse)),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=20000,
                              dual=dual, random_state=config.SEED)),
        ])
    if name == "lr":
        return Pipeline([
            ("scale", StandardScaler(with_mean=not sparse)),
            ("clf", LogisticRegression(max_iter=800, class_weight="balanced",
                                       solver="lbfgs", random_state=config.SEED)),
        ])
    if name == "nb":
        if sparse:
            return ComplementNB(alpha=0.5)
        return GaussianNB()
    raise ValueError(f"modelo desconocido: {name}")


def _slice(X, rows):
    return X[rows]


def cv_run(X, y, groups, model_name: str, n_splits: int = config.N_SPLITS,
           block: str = None) -> dict:
    """Validación cruzada agrupada y estratificada; devuelve OOF y métricas."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    oof = np.empty(len(y), dtype=object)
    fold_scores = []

    for train, test in splitter.split(X, y, groups):
        model = make_model(model_name, X, block=block)
        model.fit(_slice(X, train), y[train])
        pred = model.predict(_slice(X, test))
        oof[test] = pred
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[test], pred, average="macro", zero_division=0)
        fold_scores.append(f1)

    labels = sorted(set(y))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, oof, labels=labels, average=None, zero_division=0)
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y, oof, labels=labels)
    per_class = [{
        "label": label, "precision": float(p), "recall": float(r),
        "f1": float(f), "support": int(s),
    } for label, p, r, f, s in zip(labels, precision, recall, f1, support)]

    return {
        "model": model_name,
        "macro_f1": macro_f1,
        "fold_f1_mean": float(np.mean(fold_scores)),
        "fold_f1_std": float(np.std(fold_scores)),
        "fold_f1": [float(x) for x in fold_scores],
        "labels": labels,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "oof_accuracy": float(np.mean(oof == y)),
        "oof": oof,
    }


def run_ablation(blocks: dict, y, groups, models_by_block: dict,
                 verbose: bool = True) -> list:
    """Entrena cada bloque con sus modelos y devuelve una fila por combinación."""
    import time

    rows = []
    for block_name, model_names in models_by_block.items():
        X = blocks[block_name]
        for model_name in model_names:
            t0 = time.time()
            result = cv_run(X, y, groups, model_name, block=block_name)
            rows.append({
                "block": block_name,
                "model": model_name,
                "macro_f1": result["macro_f1"],
                "fold_mean": result["fold_f1_mean"],
                "fold_std": result["fold_f1_std"],
                "accuracy": result["oof_accuracy"],
                "seconds": round(time.time() - t0, 1),
                "per_class": result["per_class"],
                "confusion": result["confusion"],
                "labels": result["labels"],
            })
            if verbose:
                print(f"  {block_name:16s} {MODEL_LABEL[model_name]:20s} "
                      f"macro-F1={result['macro_f1']:.3f} "
                      f"(±{result['fold_f1_std']:.3f})  {time.time() - t0:5.1f}s")
    return rows


def learning_curve(X, y, groups, model_name: str, fractions=(0.10, 0.25, 0.50, 0.75, 1.00),
                   n_splits: int = 3, block: str = None) -> list:
    """Macro-F1 frente a fracción de entrenamiento (¿cuánta firma estilística hay?)."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    rows = []
    for fraction in fractions:
        scores = []
        for train, test in splitter.split(X, y, groups):
            rng = np.random.default_rng(config.SEED)
            train = rng.permutation(train)
            train = train[:max(20, int(len(train) * fraction))]
            model = make_model(model_name, X, block=block)
            model.fit(_slice(X, train), y[train])
            pred = model.predict(_slice(X, test))
            scores.append(precision_recall_fscore_support(
                y[test], pred, average="macro", zero_division=0)[2])
        rows.append({"fraction": fraction, "macro_f1": float(np.mean(scores)),
                     "f1_std": float(np.std(scores))})
    return rows


def confusion_table(result: dict, short=lambda s: s.split()[0]) -> list:
    """Matriz de confusión como filas ``{real, pred_1..k, total}``."""
    labels = result["labels"]
    rows = []
    for i, label in enumerate(labels):
        row = {"real": short(label)}
        for j, other in enumerate(labels):
            row[short(other)] = int(result["confusion"][i][j])
        row["total"] = int(sum(result["confusion"][i]))
        rows.append(row)
    return rows
