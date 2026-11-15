"""CLI: corpus unificado -> eventos de promesa (artefactos del módulo de EI).

    python -m ei_scripts.build_events                      # corrida completa
    python -m ei_scripts.build_events --limit 200          # prueba rápida
    python -m ei_scripts.build_events --no-cyk --no-slots  # ablación
    python -m ei_scripts.build_events --sample 30          # muestra para el gold

Escribe en `ei_scripts/artifacts/`:

    events.jsonl          un evento por línea (con árbol CYK en texto)
    events_by_doc.jsonl   {"doc_id": ..., "events": [...]} para el dashboard
    stats.json            embudo de filtros, distribuciones, deduplicación
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from ei_scripts import config
from ei_scripts.dedup import deduplicate, MinHasher
from ei_scripts.extract import Options, automaton, extract_sentence, json_ready
from ei_scripts.segment import segment


def _speaker_tokens(event: dict) -> list[str]:
    """Tokens plegados de la oración, para los shingles de MinHash."""
    from ei_scripts.segment import tokenize

    return [t.folded for t in tokenize(event["sentence"]) if t.is_word]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Extracción de anuncios/promesas (EI)")
    ap.add_argument("--limit", type=int, default=0, help="máximo de documentos (0 = todos)")
    ap.add_argument("--no-slots", action="store_true", help="desactiva los slots regex")
    ap.add_argument("--no-cyk", action="store_true", help="desactiva el filtro de la CFG")
    ap.add_argument("--no-negation", action="store_true", help="desactiva el autómata de negación")
    ap.add_argument("--no-gazetteer", action="store_true", help="desactiva lugares/organizaciones")
    ap.add_argument("--min-conf", type=float, default=config.MIN_CONFIDENCE)
    ap.add_argument("--no-dedup", action="store_true", help="no deduplicar")
    ap.add_argument("--sample", type=int, default=0, help="tamaño de la muestra por hablante")
    ap.add_argument("--tag", default="", help="sufijo de los archivos de salida")
    args = ap.parse_args(argv)

    from ri_scripts.corpus import iter_corpus

    opts = Options(
        use_slots=not args.no_slots,
        use_cyk=not args.no_cyk,
        use_negation=not args.no_negation,
        use_gazetteer=not args.no_gazetteer,
        min_conf=args.min_conf,
    )
    ac = automaton()
    stats: dict = {}
    events: list[dict] = []
    by_doc: dict[int, list[dict]] = defaultdict(list)
    sentences_by_doc: dict[int, list[dict]] = defaultdict(list)
    sentences_per_speaker = Counter()

    t0 = time.time()
    n_docs = 0
    for doc in iter_corpus():
        if args.limit and n_docs >= args.limit:
            break
        n_docs += 1
        meta = {k: doc.get(k) for k in ("doc_id", "speech_id", "speaker", "country", "date")}
        for sentence in segment(doc["text"]):
            sentences_per_speaker[meta["speaker"]] += 1
            if args.sample:
                sentences_by_doc[doc["doc_id"]].append(
                    {"doc_id": doc["doc_id"], "sent_idx": sentence.idx,
                     "speaker": meta["speaker"], "date": meta["date"],
                     "sentence": sentence.text.strip()}
                )
            ev = extract_sentence(sentence, opts, ac, meta, stats)
            if ev is not None:
                events.append(ev)
                by_doc[doc["doc_id"]].append(ev)
        if n_docs % 250 == 0:
            print(f"  {n_docs:5d} docs · {len(events):6d} eventos · {time.time()-t0:6.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"extracción: {n_docs} docs, {len(events)} eventos, {elapsed:.1f}s")

    # -- deduplicación --------------------------------------------------------
    ddup = {"grupos": 0, "pares_verificados": 0, "pares_colision": 0}
    if not args.no_dedup and events:
        hasher = MinHasher()
        ddup = deduplicate(events, _speaker_tokens, lambda e: e["speaker"], hasher, verbose=True)
        for i, ev in enumerate(events):
            ev["dup_group"] = ddup["grupo"].get(i)
            ev["dup_rep"] = ddup["representante"].get(ev["dup_group"]) == i if ev.get("dup_group") is not None else True

    # -- salida ---------------------------------------------------------------
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    events_path = config.EVENTS_JSONL.with_name(f"events{suffix}.jsonl")
    by_doc_path = config.EVENT_DOCS_JSONL.with_name(f"events_by_doc{suffix}.jsonl")
    stats_path = config.STATS_JSON.with_name(f"stats{suffix}.json")

    with open(events_path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(json_ready(ev), ensure_ascii=False) + "\n")
    with open(by_doc_path, "w", encoding="utf-8") as fh:
        for doc_id, evs in sorted(by_doc.items()):
            fh.write(json.dumps({"doc_id": doc_id, "n": len(evs),
                                 "events": [json_ready(e) for e in evs]}, ensure_ascii=False) + "\n")

    # -- estadísticas ---------------------------------------------------------
    per_speaker: dict[str, dict] = {}
    for ev in events:
        s = per_speaker.setdefault(ev["speaker"], {"eventos": 0, "confianza": [], "tipos": Counter()})
        s["eventos"] += 1
        s["confianza"].append(ev["confidence"])
        s["tipos"][ev["event_type"]] += 1

    stats_out = {
        "opciones": {
            "slots": opts.use_slots, "cyk": opts.use_cyk, "negacion": opts.use_negation,
            "gazetteer": opts.use_gazetteer, "min_conf": opts.min_conf, "limit": args.limit,
        },
        "volumen": {
            "documentos": n_docs,
            "eventos": len(events),
            "segundos": round(elapsed, 1),
            "autómata": ac.stats(),
        },
        "embudo": dict(sorted(stats.items(), key=lambda kv: -kv[1])),
        "por_hablante": {
            sp: {
                "eventos": d["eventos"],
                "oraciones": sentences_per_speaker.get(sp, 0),
                "eventos_por_1000_oraciones": round(
                    1000 * d["eventos"] / max(1, sentences_per_speaker.get(sp, 0)), 2),
                "confianza_media": round(sum(d["confianza"]) / len(d["confianza"]), 3),
                "tipos": dict(d["tipos"].most_common()),
            }
            for sp, d in sorted(per_speaker.items(), key=lambda kv: -kv[1]["eventos"])
        },
        "por_ano": dict(sorted(Counter(e["date"][:4] for e in events if e["date"]).items())),
        "por_tipo": dict(Counter(e["event_type"] for e in events).most_common()),
        "disparadores": [
            {"forma": f, "n": n}
            for f, n in Counter(e["trigger"] for e in events).most_common(25)
        ],
        "polos": dict(Counter(e["polarity"] for e in events)),
        "cobertura_slots": {
            "fecha": sum(1 for e in events if e["frame"]["fecha"]),
            "monto": sum(1 for e in events if e["frame"]["monto"]),
            "cantidad": sum(1 for e in events if e["frame"]["cantidad"]),
            "lugar": sum(1 for e in events if e["frame"]["lugar"]),
            "beneficiario": sum(1 for e in events if e["frame"]["beneficiario"]),
            "organizacion": sum(1 for e in events if e["frame"]["organizacion"]),
        },
        "lugares_top": Counter(e["frame"]["lugar"] for e in events if e["frame"]["lugar"]).most_common(20),
        "dedup": ddup,
        "ejemplos_por_hablante": _examples(events),
    }
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats_out, fh, ensure_ascii=False, indent=2)

    print(f"eventos   -> {events_path}")
    print(f"por doc   -> {by_doc_path}")
    print(f"stats     -> {stats_path}")
    print(f"funnel    -> {stats_out['embudo']}")

    # -- muestra para etiquetado manual --------------------------------------
    if args.sample:
        _write_sample(sentences_by_doc, args.sample, events, suffix)


def _examples(events: list[dict], n: int = 2) -> dict:
    out: dict[str, list] = {}
    for ev in events:
        bucket = out.setdefault(ev["speaker"], [])
        if len(bucket) < n:
            bucket.append({"fecha": ev["date"], "confianza": ev["confidence"],
                           "tipo": ev["event_type"], "oracion": ev["sentence"][:300]})
    return out


def _write_sample(sentences_by_doc, per_speaker: int, events: list[dict], suffix: str) -> None:
    """Muestra estratificada de oraciones para el gold standard.

    Tres estratos: aleatorio (mide prevalencia), aceptado por el sistema (mide
    precisión) y con disparador pero rechazado (mide recall perdido).
    """
    rng = random.Random(config.RANDOM_SEED)
    accepted = {(e["doc_id"], e["sent_idx"]) for e in events}
    pool: dict[str, list[dict]] = defaultdict(list)
    for doc_id, items in sentences_by_doc.items():
        for item in items:
            pool[item["speaker"]].append(item)

    rows = []
    for speaker, items in pool.items():
        rng.shuffle(items)
        n_rand = per_speaker
        n_acc = max(1, per_speaker // 3)
        acc = [s for s in items if (s["doc_id"], s["sent_idx"]) in accepted]
        rej = [s for s in items if (s["doc_id"], s["sent_idx"]) not in accepted]
        chosen = [(s, "aleatorio") for s in items[:n_rand]]
        chosen += [(s, "aceptado") for s in acc[:n_acc]]
        chosen += [(s, "rechazado") for s in rej[:n_acc]]
        for item, stratum in chosen:
            rows.append({
                "gold_id": f"g{len(rows):04d}",
                "doc_id": item["doc_id"],
                "sent_idx": item["sent_idx"],
                "speaker": speaker,
                "date": item["date"],
                "estrato": stratum,
                "label": "",
                "sentence": item["sentence"].replace("\n", " ")[:400],
            })

    path = config.GOLD_CANDIDATES_CSV.with_name(f"gold_candidates{suffix}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"muestra   -> {path} ({len(rows)} oraciones)")


if __name__ == "__main__":
    main()
