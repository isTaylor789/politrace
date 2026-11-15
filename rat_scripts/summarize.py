"""API del resumidor: de un texto (o de un discurso del corpus) a un resumen.

Uso rápido::

    from rat_scripts import summarize_text, summarize_doc

    s = summarize_text(texto, title="Inauguración del hospital")
    print(s.text)

    s = summarize_doc(doc_id=1065)          # discurso ya indexado en RI
    print(s.text)

El pipeline es el del temario 4.3: identificación de unidades (oraciones),
puntuación (estadística, de grafo, latente y discursiva), selección con MMR y
generación del resumen.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from rat_scripts import config
from rat_scripts.features import position_score
from rat_scripts.representation import Representation, build_representation
from rat_scripts.scorers import all_scores, hybrid_score
from rat_scripts.select import (compression_stats, render, select_mmr,
                                target_sentences)
from rat_scripts.units import build_units

# Un método = uno (o varios) componentes con su peso. "lead" es la línea base
# trivial: las primeras k oraciones del discurso.
METHOD_WEIGHTS = {
    "hybrid": config.WEIGHTS,
    "centroid": {"centroid": 1.0},
    "tfidf": {"tfidf": 1.0},
    "luhn": {"luhn": 1.0},
    "textrank": {"textrank": 1.0},
    "lexrank": {"lexrank": 1.0},
    "lsa": {"lsa": 1.0},
    "discourse": {"discourse": 1.0},
}
METHODS = ["lead", "tfidf", "centroid", "luhn", "textrank", "lexrank", "lsa", "hybrid"]


@dataclass
class Summary:
    """Resultado del resumen: texto, oraciones elegidas y métricas."""

    text: str
    method: str
    sentences: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    component_scores: dict = field(default_factory=dict)
    title: str = ""
    speaker: str = ""
    date: str = ""
    doc_id: int | None = None
    rep: Representation | None = None
    common_terms: list = field(default_factory=list)
    groups: list = field(default_factory=list)      # grupos temáticos (multi-doc por año)
    idf_mode: str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "speaker": self.speaker,
            "date": self.date,
            "title": self.title,
            "method": self.method,
            "summary": self.text,
            "sentences": self.sentences,
            "stats": self.stats,
            "groups": self.groups,
            "common_terms": self.common_terms,
            "idf_mode": self.idf_mode,
        }

    def pretty(self, width: int = 88) -> str:
        """Resumen en párrafo con sangría, para imprimir en consola."""
        import textwrap

        head = " ".join(x for x in (self.speaker, self.date, f"· {self.method}") if x)
        body = "\n".join(textwrap.wrap(self.text, width=width)) if self.text else "(sin resumen)"
        return f"{head}\n{body}"


def _coverage(rep: Representation, chosen: list, top_k: int = None) -> float:
    """Fracción del peso TF-IDF de los `top_k` conceptos del documento que
    aparece en el resumen (cobertura léxica del tópico principal)."""
    top_k = config.COVERAGE_TOP_TERMS if top_k is None else top_k
    weights = rep.term_weights()
    if weights.sum() <= 0:
        return 0.0
    top = np.argsort(-weights, kind="mergesort")[:top_k]
    chosen_terms = set()
    for i in chosen:
        chosen_terms.update(rep.units[i].terms)
    hit = sum(weights[j] for j in top if rep.terms[j] in chosen_terms)
    return float(hit / weights[top].sum())


def _redundancy(rep: Representation, chosen: list) -> float:
    if len(chosen) < 2:
        return 0.0
    block = rep.sim[np.ix_(chosen, chosen)]
    mask = ~np.eye(len(chosen), dtype=bool)
    return float(block[mask].mean())


def summarize_text(
    text: str,
    *,
    title: str = "",
    method: str = "hybrid",
    ratio: float = None,
    min_sentences: int = None,
    max_sentences: int = None,
    word_budget: int = None,
    lam: float = None,
    use_global_idf: bool = False,
    weights: dict = None,
    speaker: str = "",
    date: str = "",
    doc_id: int | None = None,
    rep: Representation | None = None,
) -> Summary:
    """Resume un texto cualquiera.

    `use_global_idf=True` reutiliza el IDF del corpus (recomendado para discursos
    del corpus); si no, se calcula un IDF local sobre las oraciones del texto.
    """
    if method not in METHOD_WEIGHTS and method != "lead":
        raise ValueError(f"método desconocido: {method}: usa uno de {METHODS}")

    units = build_units(text, title=title)
    stats = compression_stats(units, [], text)
    if not units:
        return Summary(text="", method=method, stats=stats, title=title,
                       speaker=speaker, date=date, doc_id=doc_id)

    rep = rep or build_representation(units, use_global_idf=use_global_idf)
    for unit in units:
        if unit.doc_label == "":
            unit.doc_label = " ".join(x for x in (speaker, date) if x)

    scores = all_scores(rep, title)

    if method == "lead":
        # línea base: posición pura, sin grafo ni estadística
        relevance = np.array([position_score(u.position, lead=1.0, tail=0.0) for u in units])
    else:
        relevance = hybrid_score(scores, weights or METHOD_WEIGHTS[method])

    # en textos muy cortos no se pueden exigir 3 oraciones sin vaciar el texto
    base_min = config.MIN_SENTENCES if min_sentences is None else min_sentences
    eff_min = max(1, min(base_min, len(units) // 4 or 1))
    k = target_sentences(len(units), ratio=ratio, min_sentences=eff_min,
                         max_sentences=max_sentences)
    chosen = select_mmr(
        relevance, rep.sim, k, units,
        lam=lam, word_budget=word_budget, min_sentences=eff_min,
    )

    body = render(units, chosen)
    stats = compression_stats(units, chosen, text)
    stats["coverage"] = _coverage(rep, chosen)
    stats["redundancy"] = _redundancy(rep, chosen)
    stats["mean_sent_words"] = float(np.mean([units[i].n_words for i in chosen])) if chosen else 0.0

    order = sorted(range(len(chosen)), key=lambda r: -relevance[chosen[r]])
    sentences = []
    for rank, pos in enumerate(order, start=1):
        i = chosen[pos]
        unit = units[i]
        sentences.append({
            "rank": rank,
            "idx": i,
            "sent_idx": unit.sent_idx,
            "char_start": unit.start,
            "char_end": unit.end,
            "text": unit.text,
            "words": unit.n_words,
            "position": round(unit.position, 4),
            "score": float(relevance[i]),
            "components": {
                name: float(values[i]) for name, values in scores.items()
                if name not in ("cue_hits", "penalty")
            },
            "doc_label": unit.doc_label,
        })

    return Summary(
        text=body,
        method=method,
        sentences=sentences,
        stats=stats,
        component_scores={name: [float(v) for v in values]
                          for name, values in scores.items()
                          if name not in ("cue_hits", "penalty")},
        title=title,
        speaker=speaker,
        date=date,
        doc_id=doc_id,
        rep=rep,
    )


def _load_index_cached():
    """Índice de RI en memoria (se carga una sola vez por proceso)."""
    global _INDEX
    if _INDEX is None:
        from ri_scripts.index import load_index

        _INDEX = load_index()
    return _INDEX


_INDEX = None


def summarize_doc(doc_id: int, *, ix=None, **kwargs) -> Summary:
    """Resume un discurso del corpus unificado por su `doc_id` (mismo de RI)."""
    from ri_scripts.corpus import read_doc

    ix = ix or _load_index_cached()
    doc = read_doc(int(doc_id), ix.corpus_path, ix.offsets)
    summary = summarize_text(
        doc["text"],
        title=doc["title"],
        use_global_idf=True,
        speaker=doc["speaker"],
        date=doc["date"],
        doc_id=int(doc_id),
        **kwargs,
    )
    return summary


def _doc_vectors(ix, doc_ids):
    """Filas de `w_global` (ya L2) para los documentos pedidos."""
    return ix.w_global[np.asarray(doc_ids)]


def select_representative_docs(doc_ids, k: int = 5, ix=None) -> list:
    """Los `k` documentos más centrales del conjunto (medoides por coseno).

    Sirve para el resumen multi-documento: en vez de promediar 200 discursos de
    un año, se eligen los que mejor representan el centroide del conjunto.
    """
    ix = ix or _load_index_cached()
    doc_ids = list(doc_ids)
    if len(doc_ids) <= k:
        return doc_ids
    M = _doc_vectors(ix, doc_ids)
    centroid = np.asarray(M.mean(axis=0)).ravel()
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return doc_ids[:k]
    sims = np.asarray(M @ (centroid / norm)).ravel()
    order = np.argsort(-sims, kind="mergesort")[:k]
    return [doc_ids[i] for i in order]


def summarize_docs(doc_ids, *, ix=None, k_docs: int = 5, per_doc_sentences: int = 3,
                   method: str = "hybrid", max_sentences: int = None,
                   word_budget: int = None, lam: float = None) -> Summary:
    """Resumen multi-documento (temario 4.6).

    Estrategia de **fusión de resúmenes** (summarize-then-fuse), que es la
    habitual cuando el conjunto es heterogéneo:

    1. se eligen los `k_docs` discursos más centrales del conjunto (medoides),
    2. cada discurso se resume por separado (`per_doc_sentences` oraciones),
    3. las oraciones de esos resúmenes se vuelven a puntuar juntas y se eligen
       con MMR, con cuota por discurso para que ninguno acapare el resumen,
    4. se reportan los tópicos comunes (términos presentes en ≥ 60 % de los
       discursos).
    """
    from ri_scripts.corpus import read_doc
    from ri_scripts.preprocess import analyze

    from rat_scripts.units import make_unit

    ix = ix or _load_index_cached()
    doc_ids = list(doc_ids)
    if not doc_ids:
        return Summary(text="", method=method, stats={"n_docs": 0})

    chosen_docs = (select_representative_docs(doc_ids, k=k_docs, ix=ix)
                   if k_docs else doc_ids)

    # 1) resumen individual de cada discurso (unidades de la fusión)
    per_doc, docs_meta, words_original = [], [], 0
    for doc_id in chosen_docs:
        doc = read_doc(int(doc_id), ix.corpus_path, ix.offsets)
        words_original += len(doc["text"].split())
        docs_meta.append({"doc_id": int(doc_id), "speaker": doc["speaker"],
                          "date": doc["date"], "title": doc["title"]})
        per_doc.append(summarize_doc(
            int(doc_id), ix=ix, method=method, ratio=1.0, min_sentences=1,
            max_sentences=per_doc_sentences, word_budget=10 ** 9,
        ))

    # 2) fusión: las oraciones ya elegidas se vuelven a puntuar en conjunto
    units = []
    for summary in per_doc:
        label = " · ".join(x for x in (summary.speaker, summary.date) if x)
        for sent in summary.sentences:
            unit = make_unit(sent["text"], idx=len(units),
                             position=0.5,   # neutraliza el prior de posición
                             doc_id=summary.doc_id, doc_label=label)
            if unit is not None:
                units.append(unit)
    if not units:
        return Summary(text="", method=method, stats={"n_docs": len(chosen_docs)})

    rep = build_representation(units, use_global_idf=True)
    scores = all_scores(rep, "")
    relevance = hybrid_score(scores, METHOD_WEIGHTS.get(method, config.WEIGHTS))

    n_docs = len(chosen_docs)
    budget = int(config.WORD_BUDGET * 1.25) if word_budget is None else word_budget
    cap = config.MAX_SENTENCES if max_sentences is None else max_sentences
    # se conserva poco más de la mitad del material fusionado
    k = int(min(cap, max(5, np.ceil(0.55 * len(units)))))
    groups = [u.doc_id for u in units]
    quota = max(1, math.ceil(k / n_docs))
    chosen = select_mmr(relevance, rep.sim, k, units, lam=lam,
                        word_budget=budget, groups=groups, max_per_group=quota)

    # 3) tópicos comunes sobre los documentos completos
    per_doc_counts = []
    for meta in docs_meta:
        doc = read_doc(meta["doc_id"], ix.corpus_path, ix.offsets)
        counts = {}
        for term in analyze(doc["text"]):
            counts[term] = counts.get(term, 0) + 1
        per_doc_counts.append(counts)
    term_docs, total_tf = {}, {}
    for counts in per_doc_counts:
        for term, count in counts.items():
            term_docs[term] = term_docs.get(term, 0) + 1
            total_tf[term] = total_tf.get(term, 0) + count
    min_docs = max(2, int(np.ceil(0.6 * n_docs)))
    common = sorted(
        ((term, docs, total_tf[term]) for term, docs in term_docs.items()
         if docs >= min_docs),
        key=lambda row: (-row[1], -row[2]),
    )[:12]

    stats = compression_stats(units, chosen)
    stats.update({
        "coverage": _coverage(rep, chosen),
        "redundancy": _redundancy(rep, chosen),
        "n_docs": n_docs,
        "doc_ids": [meta["doc_id"] for meta in docs_meta],
        "words_original": words_original,
        "compression": stats["words_summary"] / max(1, words_original),
        "reduction": 1.0 - stats["words_summary"] / max(1, words_original),
        "mean_sent_words": (float(np.mean([units[i].n_words for i in chosen]))
                            if chosen else 0.0),
        "per_doc_selected": {
            meta["doc_id"]: sum(1 for i in chosen if units[i].doc_id == meta["doc_id"])
            for meta in docs_meta
        },
    })

    sentences = [{
        "idx": i,
        "doc_id": units[i].doc_id,
        "doc_label": units[i].doc_label,
        "text": units[i].text,
        "score": float(relevance[i]),
    } for i in sorted(chosen)]

    return Summary(text=render(units, chosen), method=method, sentences=sentences,
                   stats=stats, component_scores={}, common_terms=common, rep=rep)
