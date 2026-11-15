"""Módulo de Entidades y Relaciones (ER) — módulo 3 del proyecto.

Grafo de entidades sobre el corpus de discursos presidenciales: detección de
menciones (gazetteer + patrones + siglas), coreferencia por reglas, extracción de
relaciones (coocurrencia en ventana y patrones verbales) y análisis de grafo con
NetworkX.

    from er_scripts import EntityDetector, EntityGraph, resolve, relations_of
    from er_scripts.mentions import EntityDetector
    from ri_scripts.corpus import read_doc
    from ei_scripts.segment import segment

    doc = read_doc(2769)
    det = EntityDetector()
    for sent in segment(doc["text"])[:5]:
        for m in resolve(sent, det.detect(sent, doc["doc_id"]), doc):
            print(m.type, m.surface, "->", m.resolved)

CLI:

    python -m er_scripts.build_graph          # corpus -> menciones, relaciones, grafo
    python -m er_scripts.evaluate             # métricas contra el gold manual
"""

from er_scripts.coref import resolve
from er_scripts.graph import EntityGraph
from er_scripts.mentions import EntityDetector, Mention, automaton
from er_scripts.relations import Occurrence, RelationAggregator, relations_of

__all__ = [
    "EntityDetector",
    "EntityGraph",
    "Mention",
    "Occurrence",
    "RelationAggregator",
    "automaton",
    "relations_of",
    "resolve",
]
