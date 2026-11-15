"""Búsqueda probabilística BM25 sobre el índice invertido.

BM25 recorre solo las listas de postings de los términos de la consulta, así que
su coste depende de la longitud de esas listas y no del número de documentos.
"""

import numpy as np

from ri_scripts.preprocess import analyze
from ri_scripts.results import make_results, speaker_candidates, top_k


def scores_bm25(ix, query: str, k1: float | None = None, b: float | None = None,
                speaker: str | None = None) -> np.ndarray:
    """Vector de scores BM25 para todos los documentos (0 si no comparten términos)."""
    k1 = ix.k1 if k1 is None else k1
    b = ix.b if b is None else b

    scores = np.zeros(ix.n_docs, dtype=np.float64)

    # avgdl local si se filtra por hablante (sus discursos tienen otra longitud media)
    if speaker is not None:
        block = ix.docs_of_speaker(speaker)
        avgdl = float(ix.dl[block].mean())
    else:
        avgdl = ix.avgdl

    length_norm = k1 * (1.0 - b + b * ix.dl / avgdl)

    for term in analyze(query):
        j = ix.term_id(term)
        if j is None:
            continue
        docs, tf = ix.postings(j)
        tf = tf.astype(np.float64)
        scores[docs] += ix.bm25_idf[j] * (tf * (k1 + 1.0)) / (tf + length_norm[docs])
    return scores


def search_bm25(ix, query: str, k: int = 10, speaker: str | None = None,
                k1: float | None = None, b: float | None = None,
                threshold: float = 0.0) -> list[dict]:
    scores = scores_bm25(ix, query, k1=k1, b=b, speaker=speaker)
    candidates = speaker_candidates(ix, speaker)
    doc_ids, top_scores = top_k(scores, k, candidates)

    keep = top_scores > threshold
    return make_results(ix, doc_ids[keep], top_scores[keep], query)
