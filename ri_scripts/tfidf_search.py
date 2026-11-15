"""Búsqueda vectorial por similitud coseno sobre `w_global` (TF-IDF)."""

from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from ri_scripts.preprocess import analyze
from ri_scripts.results import make_results, speaker_candidates, top_k


def query_weights(ix, query: str, use_idf: bool = True) -> sp.csr_matrix | None:
    """Convierte la consulta en un vector disperso sobre el vocabulario del índice."""
    terms = analyze(query)
    if not terms:
        return None
    counts = Counter(terms)
    cols, vals = [], []
    for term, freq in counts.items():
        j = ix.term_id(term)
        if j is None:
            continue
        cols.append(j)
        vals.append(float(freq) * (ix.idf[j] if use_idf else 1.0))
    if not cols:
        return None
    q = sp.csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, ix.vocab_size))
    return normalize(q, norm="l2", copy=False)


def search_tfidf(ix, query: str, k: int = 10, speaker: str | None = None, threshold: float = 0.0) -> list[dict]:
    """Devuelve los k documentos más similares a la consulta.

    Los vectores de documento están normalizados L2, así que el producto punto
    con la consulta normalizada es exactamente el coseno en [0, 1].
    """
    q = query_weights(ix, query)
    if q is None:
        return []

    scores = np.asarray((ix.w_global @ q.T).todense()).ravel()
    candidates = speaker_candidates(ix, speaker)
    doc_ids, top_scores = top_k(scores, k, candidates)

    keep = top_scores > threshold
    return make_results(ix, doc_ids[keep], top_scores[keep], query)
