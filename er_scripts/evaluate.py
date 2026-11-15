"""Evaluación del módulo de entidades y relaciones.

Dos evaluaciones distintas, porque miden cosas distintas:

**Menciones** (gold por oración). Se etiquetan a mano las entidades de una muestra
de oraciones (`gold_sentences.csv`, columna `entidades_gold` con el formato
`PER:Nombre|ORG:Organización|LOC:Lugar`). Para cada tipo:

    P = TP/(TP+FP)   R = TP/(TP+FN)   F1 = 2PR/(P+R)

con emparejamiento **estricto** (etiqueta canónica idéntica) y **laxo** (forma
plegada contenida en la superficie de la mención). Se reportan por tipo y en
micro-promedio, con intervalo de Wilson al 95 %.

**Relaciones** (muestra etiquetada). Se muestrea un conjunto de aristas con su
oración de evidencia y se marca si la relación es plausible. Se reporta la
precisión por tipo de relación (no hay recall: no existe gold exhaustivo de
relaciones, y construir uno excede el alcance del módulo).

Uso:

    python -m er_scripts.evaluate                 # evalúa contra gold_sentences.csv
    python -m er_scripts.evaluate --sample-relations 30   # genera muestra de relaciones
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict

from ri_scripts.preprocess import fold

from er_scripts import config
from er_scripts.coref import resolve
from er_scripts.mentions import EntityDetector, canonical_label


# -----------------------------------------------------------------------------
# Estadística
# -----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3),
            "recall": round(r, 3), "f1": round(f1, 3),
            "precision_ci": [round(x, 3) for x in wilson(tp, tp + fp)],
            "recall_ci": [round(x, 3) for x in wilson(tp, tp + fn)]}


# -----------------------------------------------------------------------------
# Gold de menciones
# -----------------------------------------------------------------------------
def load_gold(path=None) -> list[dict]:
    path = path or config.GOLD_SENTENCES_CSV
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            spec = (row.get("entidades_gold") or "").strip()
            if not spec:
                continue
            if spec == "-":                 # oración etiquetada como "sin entidades"
                row["gold"] = []
                row["doc_id"] = int(row["doc_id"])
                row["sent_idx"] = int(row["sent_idx"])
                rows.append(row)
                continue
            gold = []
            for chunk in spec.split("|"):
                chunk = chunk.strip()
                if not chunk or ":" not in chunk:
                    continue
                etype, label = chunk.split(":", 1)
                gold.append((etype.strip().upper(), label.strip()))
            row["gold"] = gold
            row["doc_id"] = int(row["doc_id"])
            row["sent_idx"] = int(row["sent_idx"])
            rows.append(row)
    return rows


def gold_sentences(gold: list[dict]) -> dict:
    from ei_scripts.segment import segment
    from ri_scripts.corpus import read_doc

    wanted: dict[int, set] = defaultdict(set)
    for row in gold:
        wanted[row["doc_id"]].add(row["sent_idx"])
    out = {}
    for doc_id, idxs in wanted.items():
        doc = read_doc(doc_id)
        for sentence in segment(doc["text"]):
            if sentence.idx in idxs:
                out[(doc_id, sentence.idx)] = (sentence, doc)
    return out


def _matches(gold_type: str, gold_label: str, mention, strict: bool) -> bool:
    if mention.type != gold_type:
        # un CARGO resuelto a persona cuenta como PER
        if not (gold_type == "PER" and mention.source.startswith("cargo")):
            return False
    label = canonical_label(mention.resolved or mention.canonical)
    if strict:
        return fold(label) == fold(gold_label)
    return fold(gold_label) in fold(mention.surface) or fold(label) == fold(gold_label)


def evaluate_mentions(gold: list[dict], sentences: dict, use_coref: bool = True,
                      strict: bool = True) -> dict:
    detector = EntityDetector()
    per_type = defaultdict(lambda: [0, 0, 0])          # tipo -> [tp, fp, fn]
    detail = []
    for row in gold:
        entry = sentences.get((row["doc_id"], row["sent_idx"]))
        if entry is None:
            continue
        sentence, doc = entry
        mentions = detector.detect(sentence, doc.get("doc_id"))
        if use_coref:
            mentions = resolve(sentence, mentions, doc)
        used = set()
        tp = {t: 0 for t in ("PER", "ORG", "LOC")}
        for gtype, glabel in row["gold"]:
            # una entidad de gold puede tener varias menciones en la oración: se
            # marcan todas para que no cuenten como falsos positivos
            hits = [i for i, m in enumerate(mentions)
                    if i not in used and _matches(gtype, glabel, m, strict)]
            if hits:
                used.update(hits)
                per_type[gtype][0] += 1
                tp[gtype] = tp.get(gtype, 0) + 1
            else:
                per_type[gtype][2] += 1
        for i, m in enumerate(mentions):
            if i in used:
                continue
            if m.type in per_type or m.type in {"PER", "ORG", "LOC"}:
                per_type[m.type][1] += 1

    micro = [sum(v[i] for v in per_type.values()) for i in range(3)]
    return {
        "n_oraciones": len(gold),
        "estricto": strict,
        "coref": use_coref,
        "por_tipo": {t: prf(*v) for t, v in per_type.items()},
        "micro": prf(*micro),
    }


# -----------------------------------------------------------------------------
# Muestra de relaciones para etiquetar
# -----------------------------------------------------------------------------
def make_relation_sample(n: int = 30, seed: int = 20240, path=None) -> None:
    path = path or config.RELATIONS_JSONL
    edges = [json.loads(line) for line in open(path, encoding="utf-8")]
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in edges:
        by_type[e["type"]].append(e)
    rows = []
    for rtype, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        quota = max(1, round(n * len(items) / len(edges)))
        rng.shuffle(items)
        for e in items[:quota]:
            ev = e["evidence"][0] if e["evidence"] else {}
            rows.append({
                "gold_id": f"r{len(rows):03d}",
                "tipo": rtype,
                "source": e["source"],
                "target": e["target"],
                "peso": e["weight"],
                "oracion": (ev.get("sentence") or "").replace("\n", " ")[:300],
                "label": "",
            })
    out = config.PKG_DIR / "gold" / "gold_relations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"muestra de relaciones -> {out} ({len(rows)} filas)")


def evaluate_relations(path=None) -> dict:
    path = path or (config.PKG_DIR / "gold" / "gold_relations.csv")
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    n = 0
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip()
            if label not in {"0", "1"}:
                continue
            n += 1
            cell = per_type[row["tipo"]]
            cell[0 if label == "1" else 1] += 1
    if not n:
        return {}
    out = {}
    groups = {"verbales (todos los tipos)": [0, 0], "cargo (inferido/explícito)": [0, 0]}
    for rtype, (ok, bad) in per_type.items():
        out[rtype] = {"n": ok + bad, "aciertos": ok, "precision": round(ok / (ok + bad), 3)}
        if rtype.startswith("cargo") or rtype == "ocupa_cargo":
            groups["cargo (inferido/explícito)"][0] += ok
            groups["cargo (inferido/explícito)"][1] += bad
        elif rtype != "coocurre":
            groups["verbales (todos los tipos)"][0] += ok
            groups["verbales (todos los tipos)"][1] += bad
    total_ok = sum(v[0] for v in per_type.values())
    return {
        "n": n, "aciertos": total_ok, "precision": round(total_ok / n, 3),
        "por_tipo": out,
        "grupos": {
            name: {"n": ok + bad, "aciertos": ok,
                   "precision": round(ok / (ok + bad), 3) if ok + bad else 0.0}
            for name, (ok, bad) in groups.items()
        },
    }


# -----------------------------------------------------------------------------
# Tablas
# -----------------------------------------------------------------------------
def markdown_tables(result: dict) -> str:
    lines = ["### Menciones (gold manual, emparejamiento estricto)", "",
             "| Tipo | TP | FP | FN | P | R | F1 | IC95 P |", "|---|---|---|---|---|---|---|---|"]
    for etype, r in result["menciones"]["por_tipo"].items():
        lines.append(f"| {etype} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} | "
                     f"[{r['precision_ci'][0]:.2f}, {r['precision_ci'][1]:.2f}] |")
    micro = result["menciones"]["micro"]
    lines.append(f"| **micro** | {micro['tp']} | {micro['fp']} | {micro['fn']} | "
                 f"{micro['precision']:.3f} | {micro['recall']:.3f} | {micro['f1']:.3f} | "
                 f"[{micro['precision_ci'][0]:.2f}, {micro['precision_ci'][1]:.2f}] |")

    lines += ["", "### Ablación: coreferencia y criterio de emparejamiento", "",
              "| Configuración | P | R | F1 |", "|---|---|---|---|"]
    for name, r in result["ablaciones"].items():
        m = r["micro"]
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")

    rel = result.get("relaciones") or {}
    if rel:
        lines += ["", "### Relaciones (muestra etiquetada)", "",
                  "| Tipo | n | aciertos | precisión |", "|---|---|---|---|"]
        for rtype, r in rel["por_tipo"].items():
            if r["n"] >= 3:
                lines.append(f"| {rtype} | {r['n']} | {r['aciertos']} | {r['precision']:.3f} |")
        for name, r in rel.get("grupos", {}).items():
            lines.append(f"| {name} | {r['n']} | {r['aciertos']} | {r['precision']:.3f} |")
        lines.append(f"| **total** | {rel['n']} | {rel['aciertos']} | {rel['precision']:.3f} |")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluación de entidades y relaciones")
    ap.add_argument("--gold", default=str(config.GOLD_SENTENCES_CSV))
    ap.add_argument("--sample-relations", type=int, default=0)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", default=str(config.EVAL_JSON))
    args = ap.parse_args(argv)

    if args.sample_relations:
        make_relation_sample(args.sample_relations)
        return

    gold = load_gold(args.gold)
    if not gold:
        raise SystemExit(f"gold vacío o sin etiquetas: {args.gold}")
    sentences = gold_sentences(gold)

    ablaciones = {
        "gazetteer + patrones + coref (estricto)": evaluate_mentions(gold, sentences, True, True),
        "sin coreferencia (estricto)": evaluate_mentions(gold, sentences, False, True),
        "con coreferencia (laxo)": evaluate_mentions(gold, sentences, True, False),
    }
    principal = ablaciones["gazetteer + patrones + coref (estricto)"]

    result = {
        "gold": {
            "oraciones": len(gold),
            "entidades": sum(len(r["gold"]) for r in gold),
            "por_tipo": dict(Counter(t for r in gold for t, _ in r["gold"])),
        },
        "menciones": principal,
        "ablaciones": ablaciones,
        "relaciones": evaluate_relations(),
    }

    print(f"gold: {result['gold']['oraciones']} oraciones, {result['gold']['entidades']} entidades "
          f"{result['gold']['por_tipo']}")
    for etype, r in principal["por_tipo"].items():
        print(f"  {etype:5s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"(TP={r['tp']} FP={r['fp']} FN={r['fn']})")
    m = principal["micro"]
    print(f"  micro P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    if result["relaciones"]:
        print(f"  relaciones (muestra): precisión={result['relaciones']['precision']} "
              f"n={result['relaciones']['n']}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")

    if args.markdown:
        config.TABLES_MD.write_text(markdown_tables(result), encoding="utf-8")
        print(f"-> {config.TABLES_MD}")


if __name__ == "__main__":
    main()
