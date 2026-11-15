"""CLI: corpus -> menciones, relaciones y grafo de entidades.

    python -m er_scripts.build_graph                  # corrida completa
    python -m er_scripts.build_graph --limit 200      # prueba rápida
    python -m er_scripts.build_graph --no-coref       # ablación sin coreferencia
    python -m er_scripts.build_graph --sample 8       # muestra de oraciones para el gold

Escribe en `er_scripts/artifacts/`:

    mentions_by_doc.jsonl  entidades por documento (conteo + superficies)
    relations.jsonl        aristas agregadas con evidencia
    graph.json             nodos/aristas para visualizar (d3/pyvis)
    graph.graphml          grafo para Gephi/NetworkX
    stats.json             embudo, distribuciones y métricas del grafo
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict

from er_scripts import config
from er_scripts.coref import resolve
from er_scripts.graph import EntityGraph
from er_scripts.mentions import EntityDetector, canonical_label, node_type
from er_scripts.relations import RelationAggregator, relations_of


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Grafo de entidades y relaciones (EI 3.3/3.5)")
    ap.add_argument("--limit", type=int, default=0, help="máximo de documentos (0 = todos)")
    ap.add_argument("--no-patterns", action="store_true", help="sólo gazetteer (sin nombres por patrón)")
    ap.add_argument("--no-coref", action="store_true", help="sin reglas de coreferencia")
    ap.add_argument("--min-weight", type=int, default=config.MIN_RELATION_WEIGHT)
    ap.add_argument("--sample", type=int, default=0, help="oraciones por hablante para el gold")
    ap.add_argument("--dump-mentions", action="store_true", help="volcar menciones crudas (grande)")
    ap.add_argument("--tag", default="", help="sufijo de los archivos de salida")
    args = ap.parse_args(argv)

    from ei_scripts.segment import segment
    from ri_scripts.corpus import iter_corpus

    detector = EntityDetector(use_patterns=not args.no_patterns)
    aggregator = RelationAggregator()
    graph = EntityGraph()

    stats: dict = Counter()
    per_speaker_entities: dict[str, Counter] = defaultdict(Counter)
    per_speaker_relations: dict[str, Counter] = defaultdict(Counter)
    by_doc: list[dict] = []
    sample_pool: dict[str, list[dict]] = defaultdict(list)
    dumped_mentions = []

    t0 = time.time()
    n_docs = 0
    for doc in iter_corpus():
        if args.limit and n_docs >= args.limit:
            break
        n_docs += 1
        meta = {k: doc.get(k) for k in ("doc_id", "speech_id", "speaker", "country", "date")}
        doc_entities: Counter = Counter()
        doc_surfaces: dict[str, Counter] = defaultdict(Counter)

        for sentence in segment(doc["text"]):
            stats["oraciones"] += 1
            mentions = detector.detect(sentence, doc.get("doc_id"))
            if not mentions:
                stats["oraciones_sin_entidades"] += 1
                continue
            stats["oraciones_con_entidades"] += 1
            if not args.no_coref:
                mentions = resolve(sentence, mentions, meta)
                stats["menciones_coref"] += sum(1 for m in mentions if m.source == "coref")

            for m in mentions:
                node = m.resolved or m.canonical
                doc_entities[node] += 1
                doc_surfaces[node][m.surface] += 1
                graph.add_mention(node, doc.get("doc_id"))
                stats["menciones"] += 1
                stats["tipo_" + node_type(node)] += 1
                stats["fuente_" + m.source] += 1
                per_speaker_entities[meta["speaker"]][node] += 1
                if args.dump_mentions:
                    dumped_mentions.append(m.as_dict())

            if args.sample:
                sample_pool[meta["speaker"]].append({
                    "doc_id": doc.get("doc_id"), "sent_idx": sentence.idx,
                    "speaker": meta["speaker"], "date": meta["date"],
                    "sentence": sentence.text.strip().replace("\n", " ")[:400],
                })

            occs = relations_of(sentence, mentions, meta)
            if occs:
                stats["oraciones_con_relaciones"] += 1
                stats["relaciones"] += len(occs)
                for occ in occs:
                    per_speaker_relations[meta["speaker"]][occ.type] += 1
                aggregator.add_all(occs)

        if doc_entities:
            by_doc.append({
                "doc_id": doc.get("doc_id"), "speaker": meta["speaker"], "date": meta["date"],
                "n_entidades": len(doc_entities),
                "entidades": [
                    {"nodo": node, "label": canonical_label(node), "tipo": node_type(node),
                     "menciones": count, "superficies": dict(doc_surfaces[node].most_common(3))}
                    for node, count in doc_entities.most_common()
                ],
            })
        if n_docs % 250 == 0:
            print(f"  {n_docs:5d} docs · {stats['menciones']:8d} menciones · "
                  f"{stats['relaciones']:9d} relaciones · {time.time()-t0:6.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"detección: {n_docs} docs, {stats['menciones']} menciones, "
          f"{stats['relaciones']} relaciones, {elapsed:.1f}s")

    # -- grafo ----------------------------------------------------------------
    edges = aggregator.edges(min_weight=args.min_weight)
    graph.add_edges(edges)
    graph._ensure_nodes()
    print(f"grafo: {graph.typed.number_of_nodes()} nodos, {len(edges)} aristas "
          f"(peso >= {args.min_weight})")
    metrics = graph.metrics()
    print(f"métricas: densidad={metrics.get('densidad')} comunidades={metrics.get('comunidades')} "
          f"componente_gigante={metrics.get('componente_gigante')}")

    # -- salida ---------------------------------------------------------------
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    path_docs = config.ARTIFACTS / f"mentions_by_doc{suffix}.jsonl"
    with open(path_docs, "w", encoding="utf-8") as fh:
        for row in by_doc:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    path_rel = config.ARTIFACTS / f"relations{suffix}.jsonl"
    with open(path_rel, "w", encoding="utf-8") as fh:
        for e in edges:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    graph_json = graph.to_json()
    path_graph = config.ARTIFACTS / f"graph{suffix}.json"
    path_graph.write_text(json.dumps(graph_json, ensure_ascii=False), encoding="utf-8")

    path_graphml = config.ARTIFACTS / f"graph{suffix}.graphml"
    graph.save_graphml(path_graphml)

    if args.dump_mentions:
        path_m = config.ARTIFACTS / f"mentions{suffix}.jsonl"
        with open(path_m, "w", encoding="utf-8") as fh:
            for m in dumped_mentions:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"menciones -> {path_m}")

    stats_out = {
        "opciones": {"patterns": not args.no_patterns, "coref": not args.no_coref,
                     "min_weight": args.min_weight, "limit": args.limit},
        "volumen": {
            "documentos": n_docs,
            "oraciones": stats["oraciones"],
            "menciones": stats["menciones"],
            "relaciones_ocurrencias": stats["relaciones"],
            "segundos": round(elapsed, 1),
        },
        "cobertura": {
            "oraciones_con_entidades": stats["oraciones_con_entidades"],
            "oraciones_sin_entidades": stats["oraciones_sin_entidades"],
            "oraciones_con_relaciones": stats["oraciones_con_relaciones"],
            "pct_oraciones_con_entidades": round(
                100 * stats["oraciones_con_entidades"] / max(1, stats["oraciones"]), 2),
        },
        "menciones": {
            "por_tipo": {k[5:]: v for k, v in sorted(stats.items(), key=lambda kv: -kv[1])
                         if k.startswith("tipo_")},
            "por_fuente": {k[7:]: v for k, v in sorted(stats.items(), key=lambda kv: -kv[1])
                           if k.startswith("fuente_")},
            "elípticas_por_coref": stats["menciones_coref"],
        },
        "relaciones": {
            "por_tipo": aggregator.type_counts(),
            "aristas_publicadas": len(edges),
        },
        "entidades": graph.node_stats(),
        "grafo": metrics,
        "por_hablante": {
            sp: {
                "menciones": sum(per_speaker_entities[sp].values()),
                "entidades_top": [{"label": canonical_label(n), "tipo": node_type(n), "n": v}
                                  for n, v in per_speaker_entities[sp].most_common(8)],
                "relaciones_top": dict(per_speaker_relations[sp].most_common(6)),
            }
            for sp in per_speaker_entities
        },
        "ejemplos_relaciones": edges[:15],
    }
    path_stats = config.ARTIFACTS / f"stats{suffix}.json"
    path_stats.write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"menciones/doc -> {path_docs}")
    print(f"relaciones    -> {path_rel}")
    print(f"grafo         -> {path_graph} · {path_graphml}")
    print(f"stats         -> {path_stats}")
    print(f"tipos         -> {stats_out['menciones']['por_tipo']}")
    print(f"fuentes       -> {stats_out['menciones']['por_fuente']}")

    if args.sample:
        _write_sample(sample_pool, args.sample, suffix)


def _write_sample(pool: dict, per_speaker: int, suffix: str) -> None:
    """Muestra estratificada de oraciones para el gold de menciones."""
    rng = random.Random(config.__dict__.get("RANDOM_SEED", 20240))
    rows = []
    for speaker, items in pool.items():
        rng.shuffle(items)
        for item in items[:per_speaker]:
            item["gold_id"] = f"g{len(rows):04d}"
            item["entidades_gold"] = ""     # se llena a mano: "PER:Pedro Sánchez|ORG:Congreso"
            rows.append(item)
    path = config.GOLD_SENTENCES_CSV.with_name(f"gold_sentences{suffix}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gold_id", "doc_id", "sent_idx", "speaker",
                                                "date", "sentence", "entidades_gold"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"muestra gold  -> {path} ({len(rows)} oraciones)")


if __name__ == "__main__":
    main()
