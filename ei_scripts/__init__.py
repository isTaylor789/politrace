"""Módulo de Extracción de Información (EI) — módulo 2 del proyecto.

Extractor de **anuncios y promesas** en discursos presidenciales, sin nada
aprendido: autómata Aho-Corasick de disparadores, expresiones regulares para los
slots, gramática independiente del contexto con parser CYK para el marco del
evento, y MinHash/LSH para deduplicar promesas repetidas.

Uso típico:

    from ei_scripts import Options, extract_doc, automaton
    from ri_scripts.corpus import read_doc

    eventos = extract_doc(read_doc(0), Options())
    eventos[0]["frame"], eventos[0]["confidence"]

CLI:

    python -m ei_scripts.build_events           # corpus -> artifacts/events.jsonl
    python -m ei_scripts.evaluate --markdown    # métricas contra el gold manual
"""

from ei_scripts.automaton import AhoCorasick, NegationAutomaton, build_trigger_automaton, resolve_triggers
from ei_scripts.dedup import MinHasher, deduplicate
from ei_scripts.extract import Options, extract_doc, extract_sentence, frame_to_columns, iter_events
from ei_scripts.grammar import build_chunks, cyk, event_type, render_tree
from ei_scripts.segment import Sentence, Token, segment, tokenize
from ei_scripts.slots import Slot, extract_slots

__all__ = [
    "AhoCorasick",
    "MinHasher",
    "NegationAutomaton",
    "Options",
    "Sentence",
    "Slot",
    "Token",
    "automaton",
    "build_chunks",
    "build_trigger_automaton",
    "cyk",
    "deduplicate",
    "event_type",
    "extract_doc",
    "extract_sentence",
    "extract_slots",
    "frame_to_columns",
    "iter_events",
    "render_tree",
    "resolve_triggers",
    "segment",
    "tokenize",
]

from ei_scripts.extract import automaton  # noqa: E402  (evita import circular)
