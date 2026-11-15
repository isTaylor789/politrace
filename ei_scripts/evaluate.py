"""Evaluación del extractor contra un gold standard manual.

Métricas por oración (¿es o no una promesa/anuncio?):

    P = TP / (TP + FP)          R = TP / (TP + FN)          F1 = 2PR / (P + R)

Con muestras pequeñas los intervalos importan más que el punto. Se reporta el
intervalo de Wilson al 95 %:

    (p̃ + z²/2n ± z·sqrt(p̃(1-p̃)/n + z²/4n²)) / (1 + z²/n),   p̃ = k/n,  z = 1.96

El gold se etiqueta a mano en tres estratos (aleatorio / aceptado / rechazado),
así que además de P y R se reporta la **prevalencia** estimada con el estrato
aleatorio, que es el único insesgado respecto de la población de oraciones.

Uso:

    python -m ei_scripts.evaluate                 # evalúa contra gold/gold_seed.csv
    python -m ei_scripts.evaluate --markdown      # además escribe tablas .md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

from ei_scripts import config
from ei_scripts.extract import Options, automaton, extract_sentence
from ei_scripts.segment import segment


# -----------------------------------------------------------------------------
# Utilidades estadísticas
# -----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson al 95 % para una proporción k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3),
            "recall": round(r, 3), "f1": round(f1, 3),
            "precision_ci": [round(x, 3) for x in wilson(tp, tp + fp)],
            "recall_ci": [round(x, 3) for x in wilson(tp, tp + fn)]}


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Kappa de Cohen entre dos etiquetados binarios (aquí: dos configuraciones)."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 0.0


# -----------------------------------------------------------------------------
# Configuraciones evaluadas (ablaciones)
# -----------------------------------------------------------------------------
# Escalera de ablaciones: se van encendiendo los filtros, sin umbral de confianza,
# para medir cada componente por separado.
CONFIGS: dict[str, Options] = {
    "A · autómata": Options(use_slots=False, use_cyk=False, use_negation=False,
                            use_gazetteer=False, min_conf=0.0),
    "B · + negación": Options(use_slots=False, use_cyk=False, use_negation=True,
                              use_gazetteer=False, min_conf=0.0),
    "C · + slots regex": Options(use_slots=True, use_cyk=False, use_negation=True,
                                 use_gazetteer=True, min_conf=0.0),
    "D · + CFG/CYK (completo)": Options(use_slots=True, use_cyk=True, use_negation=True,
                                        use_gazetteer=True, min_conf=0.0),
}

# El score sólo toma valores discretos (combinación de indicadores con pesos fijos),
# así que el "barrido de umbral" se hace sobre los niveles de evidencia reales.
LEVELS = [
    ("cualquier evidencia", 0.0),
    ("acción de contenido", 0.65),
    ("+ CYK", 0.71),
    ("+ lugar explícito", 0.81),
    ("+ fecha / monto / cantidad", 0.86),
]


def load_gold(path=config.GOLD_CSV) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip()
            if label not in {"0", "1"}:
                continue
            row["label"] = int(label)
            row["doc_id"] = int(row["doc_id"])
            row["sent_idx"] = int(row["sent_idx"])
            rows.append(row)
    return rows


def gold_sentences(gold: list[dict]) -> dict[tuple[int, int], object]:
    """Re-segmenta sólo los documentos que aparecen en el gold."""
    from ri_scripts.corpus import read_doc

    wanted = defaultdict(set)
    for row in gold:
        wanted[row["doc_id"]].add(row["sent_idx"])
    out = {}
    for doc_id, idxs in wanted.items():
        doc = read_doc(doc_id)
        for sentence in segment(doc["text"]):
            if sentence.idx in idxs:
                out[(doc_id, sentence.idx)] = sentence
    return out


def sweep(gold: list[dict], sentences: dict) -> list[dict]:
    """P/R/F1 en cada nivel de evidencia del sistema completo."""
    ac = automaton()
    out = []
    for nombre, thr in LEVELS:
        opts = Options(min_conf=thr)
        tp = fp = fn = 0
        for row in gold:
            sentence = sentences.get((row["doc_id"], row["sent_idx"]))
            pred = 0
            if sentence is not None:
                pred = 1 if extract_sentence(sentence, opts, ac, {"doc_id": row["doc_id"]}) else 0
            if pred and row["label"]:
                tp += 1
            elif pred:
                fp += 1
            elif row["label"]:
                fn += 1
        out.append({"nivel": nombre, "umbral": thr, **prf(tp, fp, fn)})
    return out


def by_stratum(gold: list[dict], sentences: dict, opts: Options | None = None) -> dict:
    """P/R por estrato de muestreo (el aleatorio es el único insesgado)."""
    ac = automaton()
    opts = opts or Options()
    cells: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in gold:
        sentence = sentences.get((row["doc_id"], row["sent_idx"]))
        pred = 0
        if sentence is not None:
            pred = 1 if extract_sentence(sentence, opts, ac, {"doc_id": row["doc_id"]}) else 0
        cell = cells[row.get("estrato") or "?"]
        if pred and row["label"]:
            cell[0] += 1
        elif pred:
            cell[1] += 1
        elif row["label"]:
            cell[2] += 1
    return {k: prf(*v) for k, v in cells.items()}


def evaluate(gold: list[dict], sentences: dict, configs=None) -> dict:
    ac = automaton()
    configs = configs or CONFIGS
    results: dict[str, dict] = {}
    per_speaker: dict[str, dict] = {}
    predictions: dict[str, list[int]] = {}

    for name, opts in configs.items():
        tp = fp = fn = tn = 0
        preds: list[int] = []
        by_spk: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for row in gold:
            key = (row["doc_id"], row["sent_idx"])
            sentence = sentences.get(key)
            pred = 0
            if sentence is not None:
                ev = extract_sentence(sentence, opts, ac, {"doc_id": row["doc_id"]})
                pred = 1 if ev is not None else 0
            preds.append(pred)
            gold_label = row["label"]
            cell = by_spk[row["speaker"]]
            if pred and gold_label:
                tp += 1; cell[0] += 1
            elif pred and not gold_label:
                fp += 1; cell[1] += 1
            elif not pred and gold_label:
                fn += 1; cell[2] += 1
            else:
                tn += 1; cell[3] += 1
        predictions[name] = preds
        results[name] = prf(tp, fp, fn) | {"tn": tn}
        per_speaker[name] = {sp: prf(c[0], c[1], c[2]) | {"tn": c[3]} for sp, c in by_spk.items()}

    # -- prevalencia en el estrato aleatorio ---------------------------------
    rand = [r for r in gold if r.get("estrato") == "aleatorio"]
    pos = sum(r["label"] for r in rand)
    prevalence = {
        "n": len(rand),
        "positivas": pos,
        "prevalencia": round(pos / len(rand), 4) if rand else 0.0,
        "prevalencia_ci": [round(x, 4) for x in wilson(pos, len(rand))],
    }

    names = list(results)
    kappas = {
        f"{a} vs {b}": cohen_kappa(predictions[a], predictions[b])
        for a, b in [(names[0], names[1]), (names[0], names[-1])]
    }
    return {
        "barrido_umbral": sweep(gold, sentences),
        "por_estrato": by_stratum(gold, sentences),
        "config_evaluada": names[-1],
        "gold": {"n": len(gold), "positivas": sum(r["label"] for r in gold),
                 "por_estrato": dict(_stratum_counts(gold)),
                 "por_hablante": dict(_speaker_counts(gold))},
        "prevalencia_estrato_aleatorio": prevalence,
        "configuraciones": results,
        "por_hablante": per_speaker,
        "kappa": kappas,
    }


def _stratum_counts(gold):
    counts = defaultdict(int)
    for row in gold:
        counts[row.get("estrato") or "?"] += 1
    return counts


def _speaker_counts(gold):
    counts = defaultdict(int)
    for row in gold:
        counts[row["speaker"]] += 1
    return counts


# -----------------------------------------------------------------------------
# Tablas markdown
# -----------------------------------------------------------------------------
def markdown_tables(result: dict, stats: dict | None = None) -> str:
    lines = ["### Ablaciones (gold manual)", "",
             "| Configuración | TP | FP | FN | P | R | F1 | IC95 P | IC95 R |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in result["configuraciones"].items():
        lines.append(
            f"| {name} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | "
            f"[{r['precision_ci'][0]:.2f}, {r['precision_ci'][1]:.2f}] | "
            f"[{r['recall_ci'][0]:.2f}, {r['recall_ci'][1]:.2f}] |"
        )
    best = result["config_evaluada"]
    lines += ["", "### Niveles de evidencia (sistema completo)", "",
              "| Nivel | Umbral | TP | FP | FN | P | R | F1 |",
              "|---|---|---|---|---|---|---|---|"]
    for row in result["barrido_umbral"]:
        lines.append(f"| {row['nivel']} | {row['umbral']:.2f} | {row['tp']} | {row['fp']} | "
                     f"{row['fn']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |")

    lines += ["", "### Por estrato de muestreo (sistema completo)", "",
              "| Estrato | TP | FP | FN | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for sp, r in result["por_estrato"].items():
        lines.append(f"| {sp} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} |")

    lines += ["", f"### Por hablante ({best})", "",
              "| Hablante | TP | FP | FN | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for sp, r in result["por_hablante"][best].items():
        lines.append(f"| {sp} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} |")
    prev = result["prevalencia_estrato_aleatorio"]
    lines += ["", f"Prevalencia en el estrato aleatorio: {prev['prevalencia']:.3f} "
                  f"({prev['positivas']}/{prev['n']}, IC95 [{prev['prevalencia_ci'][0]:.3f}, "
                  f"{prev['prevalencia_ci'][1]:.3f}]).",
              "", "Kappa entre configuraciones: " +
              ", ".join(f"{k} = {v}" for k, v in result["kappa"].items())]
    if stats:
        lines += ["", "### Volumen del corpus", "", "| Métrica | Valor |", "|---|---|"]
        for key, value in stats.get("volumen", {}).items():
            lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluación del extractor de promesas")
    ap.add_argument("--gold", default=str(config.GOLD_CSV))
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", default=str(config.EVAL_JSON))
    args = ap.parse_args(argv)

    gold = load_gold(args.gold)
    if not gold:
        raise SystemExit(f"gold vacío o sin etiquetas: {args.gold}")
    sentences = gold_sentences(gold)
    result = evaluate(gold, sentences)

    print(f"gold: {result['gold']['n']} oraciones "
          f"({result['gold']['positivas']} positivas) · {result['gold']['por_estrato']}")
    for name, r in result["configuraciones"].items():
        print(f"  {name:22s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"(TP={r['tp']} FP={r['fp']} FN={r['fn']})")
    print(f"  prevalencia (aleatorio) = {result['prevalencia_estrato_aleatorio']['prevalencia']:.3f}")
    print(f"  kappa = {result['kappa']}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")

    if args.markdown:
        stats = None
        if config.STATS_JSON.exists():
            stats = json.loads(config.STATS_JSON.read_text(encoding="utf-8"))
        md = markdown_tables(result, stats)
        path = config.ARTIFACTS / "tables.md"
        path.write_text(md, encoding="utf-8")
        print(f"-> {path}")


if __name__ == "__main__":
    main()
