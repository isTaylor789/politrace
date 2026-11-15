"""Módulo de Recuperación de Información (RI) — parte 1 del proyecto.

Uso típico:

    from ri_scripts import build_all, load_index, search_tfidf, search_bm25, search_boolean

    build_all.main()                 # genera corpus.jsonl + índice + w_global
    ix = load_index()                # carga todo en memoria
    search_tfidf(ix, "inflación", k=5)
"""

from ri_scripts.bm25_search import search_bm25
from ri_scripts.boolean_search import parse_boolean, search_boolean
from ri_scripts.index import Index, artifacts_exist, build_index, load_index, save_index
from ri_scripts.preprocess import STOPWORDS, analyze, fold, stem
from ri_scripts.tfidf_search import search_tfidf

__all__ = [
    "Index",
    "STOPWORDS",
    "analyze",
    "artifacts_exist",
    "build_index",
    "fold",
    "load_index",
    "parse_boolean",
    "save_index",
    "search_bm25",
    "search_boolean",
    "search_tfidf",
    "stem",
]
