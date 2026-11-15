"""Selección de oraciones con MMR y generación del resumen (temario 4.3.3/4.3.4).

MMR (Maximal Marginal Relevance, Carbonell & Goldstein 1998) selecciona de forma
greedy la oración que maximiza relevancia y minimiza redundancia:

    MMR(i) = λ · rel(i) - (1 - λ) · max_{j ∈ S} sim(i, j)

con ``rel`` normalizada a [0,1], ``S`` el conjunto ya seleccionado y λ = 0.7.
Se añaden dos salvaguardas para que el resumen sea legible:
un filtro de casi-duplicados (Jaccard de términos > 0.8) y un presupuesto de
palabras. Al final las oraciones se reordenan según el discurso original.
"""

import numpy as np

from rat_scripts import config

_END = (".", "!", "?", "…", "”", '"')


def target_sentences(n_units: int, *, ratio: float = None, min_sentences: int = None,
                     max_sentences: int = None) -> int:
    """Número de oraciones objetivo: ceil(ratio·n), acotado por min y max."""
    ratio = config.SUMMARY_RATIO if ratio is None else ratio
    min_sentences = config.MIN_SENTENCES if min_sentences is None else min_sentences
    max_sentences = config.MAX_SENTENCES if max_sentences is None else max_sentences
    k = int(np.ceil(ratio * n_units))
    return int(max(min_sentences, min(max_sentences, k)))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_mmr(relevance: np.ndarray, sim: np.ndarray, k: int, units: list, *,
               lam: float = None, word_budget: int = None,
               dup_jaccard: float = None, min_sentences: int = None,
               groups: list = None, max_per_group: int = None) -> list:
    """Devuelve los índices (orden original) de las oraciones seleccionadas."""
    lam = config.MMR_LAMBDA if lam is None else lam
    word_budget = config.WORD_BUDGET if word_budget is None else word_budget
    dup_jaccard = config.DUP_JACCARD if dup_jaccard is None else dup_jaccard
    min_sentences = config.MIN_SENTENCES if min_sentences is None else min_sentences

    n = len(units)
    if n == 0:
        return []
    rel = np.asarray(relevance, dtype=float)
    if rel.max() > rel.min():
        rel = (rel - rel.min()) / (rel.max() - rel.min())
    else:
        rel = np.ones(n)

    term_sets = [set(u.terms) for u in units]
    chosen: list = []
    blocked = np.zeros(n, dtype=bool)

    while len(chosen) < min(k, n):
        remaining = np.where(~blocked)[0]
        if remaining.size == 0:
            break

        if chosen:
            redundancy = sim[np.ix_(remaining, chosen)].max(axis=1)
        else:
            redundancy = np.zeros(remaining.size)

        mmr = lam * rel[remaining] - (1.0 - lam) * redundancy

        # cuota por grupo/documento: un discurso o un tema no debe acaparar
        # todas las oraciones. `max_per_group` puede ser un entero (misma cuota
        # para todos) o un dict {grupo: cuota} (cuota proporcional al tamaño).
        if groups is not None and max_per_group:
            def _cap(i):
                if isinstance(max_per_group, dict):
                    return max_per_group.get(groups[i], 0)
                return max_per_group

            allowed = np.array([
                groups[i] is None
                or sum(1 for j in chosen if groups[j] == groups[i]) < _cap(i)
                for i in remaining
            ])
            if allowed.any():
                mmr = np.where(allowed, mmr, -np.inf)
            else:
                break

        best = int(remaining[int(np.argmax(mmr))])

        # casi-duplicado de algo ya elegido: se descarta y se sigue
        if any(jaccard(term_sets[best], term_sets[j]) > dup_jaccard for j in chosen):
            blocked[best] = True
            continue

        words = sum(units[j].n_words for j in chosen) + units[best].n_words
        if chosen and len(chosen) >= min_sentences and words > word_budget:
            # no cabe: se descarta este candidato y se prueba el siguiente
            blocked[best] = True
            continue

        chosen.append(best)
        blocked[best] = True

    return sorted(chosen)


def render(units: list, chosen: list) -> str:
    """Une las oraciones elegidas en orden de discurso, con puntuación limpia."""
    parts = []
    for i in chosen:
        text = units[i].text.strip()
        if not text:
            continue
        if not text.endswith(_END):
            text += "."
        parts.append(text)
    summary = " ".join(parts)
    return " ".join(summary.split())


def compression_stats(units: list, chosen: list, original_text: str = "") -> dict:
    """Métricas de tamaño: palabras originales, del resumen y compresión."""
    words_summary = sum(units[i].n_words for i in chosen)
    words_original = len(original_text.split()) if original_text else sum(u.n_words for u in units)
    return {
        "n_units": len(units),
        "n_selected": len(chosen),
        "words_original": words_original,
        "words_summary": words_summary,
        "compression": words_summary / words_original if words_original else 0.0,
        "reduction": 1.0 - words_summary / words_original if words_original else 0.0,
    }
