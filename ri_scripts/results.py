"""Utilidades compartidas por los buscadores: fragmentos y formato de resultados."""

import re

from ri_scripts import config
from ri_scripts.corpus import read_doc
from ri_scripts.preprocess import fold, surface_terms

_WORD_RE = re.compile(r"[a-zñ0-9]+")


def snippet(text: str, query: str, width: int = 180) -> str:
    """Ventana de texto alrededor de la primera aparición de un término de consulta."""
    clean = re.sub(r"\s+", " ", text).strip()
    folded = fold(clean)

    best = -1
    for term in surface_terms(query):
        pos = folded.find(term)
        if pos != -1 and (best == -1 or pos < best):
            best = pos

    if best == -1:
        return clean[:width] + ("…" if len(clean) > width else "")

    start = max(0, best - width // 3)
    end = min(len(clean), start + width)
    # cortar en frontera de palabra
    if start > 0:
        space = clean.find(" ", start)
        start = space + 1 if 0 <= space < best else start
    if end < len(clean):
        space = clean.rfind(" ", 0, end)
        end = space if space > start else end
    return ("…" if start > 0 else "") + clean[start:end] + ("…" if end < len(clean) else "")


def make_results(ix, doc_ids, scores, query: str, limit: int | None = None) -> list[dict]:
    """Empaqueta doc_id/score en registros listos para tabular o imprimir."""
    results = []
    for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
        doc = read_doc(int(doc_id), ix.corpus_path, ix.offsets)
        results.append(
            {
                "rank": rank,
                "doc_id": int(doc_id),
                "score": float(score),
                "speaker": ix.speakers[doc_id],
                "date": ix.dates[doc_id],
                "title": ix.titles[doc_id],
                "snippet": snippet(doc["text"], query),
            }
        )
        if limit is not None and rank >= limit:
            break
    return results


def top_k(scores, k: int, candidates=None) -> tuple:
    """Índices de los k mejores scores (opcionalmente restringido a candidatos)."""
    import numpy as np

    if candidates is None:
        candidates = np.arange(len(scores))
    candidates = np.asarray(candidates)
    if candidates.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    cand_scores = scores[candidates]
    k = min(k, candidates.size)
    if k < candidates.size:
        part = np.argpartition(-cand_scores, k - 1)[:k]
    else:
        part = np.arange(candidates.size)
    order = part[np.argsort(-cand_scores[part], kind="mergesort")]
    return candidates[order], cand_scores[order]


def speaker_candidates(ix, speaker: str | None):
    if speaker is None:
        return None
    if speaker not in ix.speaker_blocks:
        raise KeyError(f"hablante desconocido: {speaker!r}")
    return ix.docs_of_speaker(speaker)
