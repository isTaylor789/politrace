"""Módulo de Resumen Automático de Textos (RAT) — parte 4 del proyecto.

Uso típico::

    from rat_scripts import summarize_text, summarize_doc, summarize_docs

    resumen = summarize_text(mi_texto, title="...")     # texto cualquiera
    resumen = summarize_doc(1065)                       # discurso del corpus
    resumen = summarize_docs(doc_ids, k_docs=5)         # fusión simple (4.6)
    anual = summarize_year(ix, 2023, "Pedro Sánchez")    # módulo 5: año completo
    print(resumen.text)

Métodos: ``hybrid`` (por omisión), ``centroid``, ``tfidf``, ``luhn``,
``textrank``, ``lexrank``, ``lsa``, ``discourse`` y la línea base ``lead``.
"""

from rat_scripts.evaluate import evaluate_summary, rouge_l, rouge_n
from rat_scripts.multi import (FusedDocs, ThemeGroup, cluster_documents,
                               docs_for_year, fuse_year, prepare_year,
                               summarize_year, year_availability)
from rat_scripts.representation import Representation, build_representation
from rat_scripts.scorers import all_scores, hybrid_score
from rat_scripts.summarize import (METHODS, Summary, select_representative_docs,
                                   summarize_doc, summarize_docs, summarize_text)
from rat_scripts.units import Unit, build_units

__all__ = [
    "METHODS",
    "FusedDocs",
    "Representation",
    "Summary",
    "ThemeGroup",
    "Unit",
    "all_scores",
    "build_representation",
    "build_units",
    "cluster_documents",
    "docs_for_year",
    "evaluate_summary",
    "fuse_year",
    "prepare_year",
    "hybrid_score",
    "rouge_l",
    "rouge_n",
    "select_representative_docs",
    "summarize_doc",
    "summarize_docs",
    "summarize_text",
    "summarize_year",
    "year_availability",
]
