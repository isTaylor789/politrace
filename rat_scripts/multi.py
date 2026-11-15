"""Resumen multi-documento por año (módulo 5 del plan; temario 4.6).

Pipeline (fusión con agrupamiento temático):

1. **Subconjunto**: discursos de un año (y opcionalmente de un hablante).
2. **Agrupamiento** de documentos por similitud coseno sobre las filas de
   `w_global` (k-means sobre vectores L2, equivalente a k-means esférico).
   Se etiqueta cada grupo con sus términos más pesados, excluyendo los términos
   genéricos del año (aparecen en ≥ 40 % de los documentos).
3. **Resúmenes individuales**: de cada grupo se toman hasta 3 documentos
   medoides (los más cercanos al centroide del grupo) y se resume cada uno con
   2 oraciones.
4. **Fusión**: las oraciones de esos resúmenes se puntúan juntas (híbrido del
   módulo 4) y se seleccionan con MMR, con cuota por grupo temático.
5. **Tópicos comunes**: términos presentes en varios grupos, ordenados por peso
   TF-IDF en el conjunto del año.

La posición dentro del discurso original se neutraliza (`p = 0.5`): el inicio de
un discurso medoide no es más importante que su desarrollo.
"""

import collections
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

from rat_scripts import config
from rat_scripts.representation import build_representation
from rat_scripts.scorers import all_scores, hybrid_score
from rat_scripts.select import render, select_mmr
from rat_scripts.summarize import (Summary, _load_index_cached,
                                   select_representative_docs, summarize_doc)
from rat_scripts.units import make_unit


@dataclass
class ThemeGroup:
    """Grupo temático de documentos (cluster por similitud coseno)."""

    group_id: int
    doc_ids: list
    size: int
    label: str = ""
    top_terms: list = field(default_factory=list)
    terms: set = field(default_factory=set)   # tema: top-30 términos no genéricos
    centroid: np.ndarray | None = None


# --- Subconjuntos ------------------------------------------------------------
def docs_for_year(ix, year: int | str, speaker: str = None) -> list:
    """`doc_id`s del año (opcionalmente de un hablante), en orden cronológico."""
    year = str(year)
    return [
        i for i in range(ix.n_docs)
        if ix.dates[i].startswith(year) and (speaker is None or ix.speakers[i] == speaker)
    ]


def year_availability(ix) -> list:
    """(hablante, año, nº de discursos) para todos los pares con documentos."""
    counts = collections.Counter(
        (ix.speakers[i], ix.dates[i][:4]) for i in range(ix.n_docs)
    )
    rows = [{"speaker": s, "year": y, "docs": n} for (s, y), n in counts.items()]
    rows.sort(key=lambda r: (r["speaker"], r["year"]))
    return rows


# --- Agrupamiento ------------------------------------------------------------
def term_doc_df(ix, doc_ids) -> np.ndarray:
    """df por término restringido a un conjunto de documentos (desde postings).

    Se calcula con una máscara booleana sobre `postings_indices`, así que no hay
    que re-tokenizar ni reconstruir la matriz documento-término.
    """
    mask = np.zeros(ix.n_docs, dtype=bool)
    mask[np.asarray(doc_ids, dtype=int)] = True
    df = np.zeros(ix.vocab_size, dtype=np.int64)
    for j in range(ix.vocab_size):
        s, e = ix.postings_indptr[j], ix.postings_indptr[j + 1]
        if s == e:
            continue
        df[j] = int(mask[ix.postings_indices[s:e]].sum())
    return df


def term_stats_by_group(ix, doc_ids, labels, n_groups: int):
    """tf y df de cada término por grupo (y total), desde los postings.

    Devuelve `(tf_grupos, df_grupos, tf_total, df_total)`, con los dos primeros
    de forma `(n_groups, V)` y los otros dos `(V,)`. Se recorre el índice
    invertido una sola vez y se agrega con `bincount`.
    """
    mask = np.zeros(ix.n_docs, dtype=bool)
    mask[np.asarray(doc_ids, dtype=int)] = True
    lab = np.full(ix.n_docs, -1, dtype=np.int64)
    lab[np.asarray(doc_ids, dtype=int)] = np.asarray(labels, dtype=np.int64)

    tf_groups = np.zeros((n_groups, ix.vocab_size), dtype=np.float64)
    df_groups = np.zeros((n_groups, ix.vocab_size), dtype=np.float64)
    for j in range(ix.vocab_size):
        s, e = ix.postings_indptr[j], ix.postings_indptr[j + 1]
        if s == e:
            continue
        docs = ix.postings_indices[s:e]
        keep = mask[docs]
        if not keep.any():
            continue
        group_of = lab[docs[keep]]
        tf_groups[:, j] = np.bincount(
            group_of, weights=ix.postings_tf[s:e][keep].astype(float),
            minlength=n_groups,
        )
        df_groups[:, j] = np.bincount(group_of, minlength=n_groups)
    return tf_groups, df_groups, tf_groups.sum(axis=0), df_groups.sum(axis=0)


def _is_clean_term(term: str) -> bool:
    """Término utilizable como etiqueta: sin dígitos y con algo de cuerpo."""
    return len(term) >= 2 and not any(ch.isdigit() for ch in term)


def _default_k(n_docs: int) -> int:
    k = int(round(math.sqrt(n_docs / 4.0)))
    return int(np.clip(k, config.CLUSTER_MIN, config.CLUSTER_MAX))


def cluster_documents(ix, doc_ids, k: int = None, *, seed: int = None,
                      df_docs: np.ndarray = None, return_stats: bool = False,
                      verbose: bool = False):
    """Agrupa documentos por coseno y devuelve grupos etiquetados por tema.

    k-means sobre las filas L2 de `w_global`: como `‖u−v‖² = 2 − 2cos(u,v)` para
    vectores unitarios, minimizar la distancia euclidiana es maximizar el coseno
    (k-means esférico). Los grupos de un solo documento se reasignan al centroide
    más cercano para no resumir discursos sueltos.
    """
    doc_ids = list(doc_ids)
    n = len(doc_ids)
    if n == 0:
        return []
    if n < config.CLUSTER_MIN:
        weights = np.asarray(ix.w_global[doc_ids].sum(0)).ravel()
        groups = [ThemeGroup(0, doc_ids, n,
                             top_terms=[ix.terms[j] for j in np.argsort(-weights)[:6]])]
        if return_stats:
            return groups, {"year_weights": weights}
        return groups

    k = _default_k(n) if k is None else k
    k = max(1, min(k, n // 2 if n >= 4 else 1))
    seed = config.CLUSTER_SEED if seed is None else seed

    X = ix.w_global[doc_ids].tocsr()
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed).fit(X)

    groups = collections.defaultdict(list)
    for doc_id, label in zip(doc_ids, km.labels_):
        groups[int(label)].append(int(doc_id))

    # --- reasignar clusters de un solo documento -----------------------------
    def centroid(ids):
        c = np.asarray(ix.w_global[ids].mean(axis=0)).ravel()
        norm = np.linalg.norm(c)
        return c / norm if norm > 0 else c

    while len(groups) > 1 and min(len(v) for v in groups.values()) == 1:
        small = min((gid for gid, v in groups.items() if len(v) == 1), key=lambda g: g)
        doc = groups.pop(small)[0]
        w = np.asarray(ix.w_global[doc].todense()).ravel()
        best, best_sim = None, -1.0
        for gid, ids in groups.items():
            sim = float(w @ centroid(ids))
            if sim > best_sim:
                best, best_sim = gid, sim
        groups[best].append(doc)

    # --- etiquetas: tf·idf dentro del grupo, sin términos genéricos ----------
    # tf (frecuencia real) × idf = "frecuente y distintivo", que es lo que
    # describe un tema; sumar sólo pesos TF-IDF de documentos largos daba
    # términos raros y anécdoticos.
    if df_docs is None:
        df_docs = term_doc_df(ix, doc_ids)
    generic = df_docs >= config.GENERIC_DF_RATIO * n

    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    labels_vec = np.zeros(len(doc_ids), dtype=np.int64)
    position = {doc: i for i, doc in enumerate(doc_ids)}
    ids_order = []
    for new_id, (gid, ids) in enumerate(ordered):
        ids_order.append(ids)
        for doc in ids:
            labels_vec[position[doc]] = new_id
    _, df_groups, _, df_total = term_stats_by_group(
        ix, doc_ids, labels_vec, len(ordered))
    # "tema del año" = término presente en muchos documentos (df), ponderado por
    # su rareza en el corpus (idf). Usar tf total dejaba que un discurso largo
    # repitiendo una palabra dominara el perfil del año.
    year_weights = df_total * ix.idf
    year_weights[generic] = 0.0

    out = []
    for gid, ids in enumerate(ids_order):
        weights = df_groups[gid] * ix.idf
        weights[generic] = 0.0
        order = [j for j in np.argsort(-weights, kind="mergesort")
                 if weights[j] > 0 and _is_clean_term(ix.terms[j])]
        top = [ix.terms[j] for j in order[:6]]
        theme = {ix.terms[j] for j in order[:30]}
        out.append(ThemeGroup(
            group_id=len(out), doc_ids=sorted(int(d) for d in ids), size=len(ids),
            label=" · ".join(top[:4]), top_terms=top, terms=theme,
            centroid=centroid(ids),
        ))

    if verbose:
        for g in out:
            print(f"  grupo {g.group_id} ({g.size:3d} docs): {', '.join(g.top_terms)}")
    if return_stats:
        return out, {"year_weights": year_weights}
    return out


# --- Utilidades del resumen anual -------------------------------------------
_DOC_WORDS: dict = {}


def _doc_words(ix, doc_id: int) -> int:
    """Nº de palabras del texto completo de un documento (cacheado)."""
    from ri_scripts.corpus import read_doc

    key = int(doc_id)
    if key not in _DOC_WORDS:
        _DOC_WORDS[key] = len(read_doc(key, ix.corpus_path, ix.offsets)["text"].split())
    return _DOC_WORDS[key]


def _year_space(ix, fused, units):
    """Vectores de las oraciones y centroide del año, en el espacio global.

    Se anulan los términos genéricos del año (≥ 40 % de los documentos) en ambos
    lados: el centroide no queda dominado por la plantilla ("presidente",
    "gobierno", el nombre del hablante) y la comparación mide tema, no estilo.
    Devuelve `(X_oraciones, centroide)` con filas L2 = 1.
    """
    import scipy.sparse as sp

    from rat_scripts.representation import global_idf

    generic = fused.generic_mask
    if generic is None:
        df_docs = term_doc_df(ix, fused.doc_ids)
        generic = df_docs >= config.GENERIC_DF_RATIO * len(fused.doc_ids)

    M = ix.w_global[fused.doc_ids].tocsr(copy=True)
    M.data[generic[M.indices]] = 0.0
    M.eliminate_zeros()
    centroid = np.asarray(M.mean(axis=0)).ravel()
    norm = np.linalg.norm(centroid)
    centroid = centroid / norm if norm > 0 else centroid

    table, default = global_idf()
    rows, cols, data = [], [], []
    for i, unit in enumerate(units):
        for term in unit.terms:
            j = ix.vocabulary.get(term)
            if j is None or generic[j]:
                continue
            rows.append(i)
            cols.append(j)
            data.append(table.get(term, default))
    if not rows:
        return sp.csr_matrix((len(units), ix.vocab_size)), centroid
    X = sp.csr_matrix((data, (rows, cols)), shape=(len(units), ix.vocab_size))
    X.sum_duplicates()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ X, centroid


def year_affinity(ix, fused, units) -> np.ndarray:
    """Coseno entre cada oración y el centroide del año (sin términos genéricos).

    Señal específica del multi-documento: mide si la oración habla de lo que
    domina en el conjunto del año, no sólo en su propio discurso. Por defecto su
    peso es 0 (`MULTI_YEAR_WEIGHT`): en las pruebas no mejoró la cobertura y
    favorecía en exceso al grupo temático mayoritario.
    """
    X, centroid = _year_space(ix, fused, units)
    if X.shape[1] != centroid.shape[0]:
        return np.zeros(len(units))
    return np.asarray(X @ centroid).ravel()


def _year_coverage(ix, fused, units, chosen) -> float:
    """Cobertura del año: coseno entre el centroide del resumen y el del año.

    Es una medida de "de qué habla el año" que no depende de listas de términos:
    los dos vectores se construyen en el mismo espacio (TF-IDF global, sin
    términos genéricos) y el resumen se promedia sobre sus oraciones elegidas.
    """
    if not chosen:
        return 0.0
    X, centroid = _year_space(ix, fused, units)
    if X.shape[1] != centroid.shape[0]:
        return 0.0
    block = X[chosen]
    summary = np.asarray(block.mean(axis=0)).ravel()
    norm = np.linalg.norm(summary)
    if norm <= 0:
        return 0.0
    return float(summary @ centroid / norm)


def _allocate_quotas(groups, k: int, min_share: float = None,
                     min_raw: float = None) -> dict:
    """Reparte `k` oraciones entre los grupos en proporción a su tamaño.

    - floor(k · n_g / n) como base;
    - los grupos que son ≥ 5 % de los documentos tienen garantizada 1 oración
      (un tema significativo no debe desaparecer por redondeo);
    - los grupos con menos de 0,5 oraciones esperadas quedan en 0 (un tema
      anecdótico de 3-11 discursos no ocupa un lugar fijo);
    - los lugares restantes se reparten por mayor resto (método de Hare).
    """
    min_share = config.MULTI_MIN_SHARE if min_share is None else min_share
    min_raw = config.MULTI_MIN_RAW if min_raw is None else min_raw
    total = max(1, sum(g.size for g in groups))
    raw = np.array([k * g.size / total for g in groups], dtype=float)
    base = np.floor(raw).astype(int)

    significant = np.array([g.size / total >= min_share for g in groups])
    base = np.where(significant, np.maximum(base, 1), base)
    base = np.where(raw <= min_raw, 0, base)

    # nunca más de lo esperado + 1 (evita que un grupo se lleve de más)
    base = np.minimum(base, np.maximum(1, np.ceil(raw)))

    remaining = k - int(base.sum())
    if remaining > 0:
        order = np.argsort(-(raw - base), kind="mergesort")
        for i in order:
            if remaining <= 0:
                break
            if base[i] == 0:
                continue
            base[i] += 1
            remaining -= 1
    elif remaining < 0:
        order = np.argsort(-base, kind="mergesort")
        for i in order:
            if remaining >= 0:
                break
            if base[i] > (1 if significant[i] else 0):
                base[i] -= 1
                remaining += 1

    return {g.group_id: int(base[i]) for i, g in enumerate(groups)}


def _redundancy(rep, chosen) -> float:
    """Coseno medio entre pares de oraciones seleccionadas."""
    if len(chosen) < 2:
        return 0.0
    block = rep.sim[np.ix_(chosen, chosen)]
    mask = ~np.eye(len(chosen), dtype=bool)
    return float(block[mask].mean())


def _common_terms(ix, doc_ids, groups, year_weights=None, min_groups: int = 2) -> list:
    """Tópicos comunes: términos compartidos por varios grupos.

    Se cuenta en cuántos grupos aparece cada término (usando el conjunto de
    tema de cada grupo) y se ordena por nº de grupos y luego por peso TF-IDF en
    el conjunto del año. Es la "detección de tópico común" de 4.6 basada en
    frecuencia/cobertura, no en similitud.
    """
    weights = (year_weights if year_weights is not None
               else np.asarray(ix.w_global[doc_ids].sum(axis=0)).ravel())
    df_docs = term_doc_df(ix, doc_ids)
    generic = df_docs >= config.GENERIC_DF_RATIO * len(doc_ids)
    counts = collections.Counter()
    for group in groups:
        for term in group.terms:          # top-30 no genéricos del grupo
            counts[term] += 1
    candidates = [
        (term, counts[term], float(weights[ix.vocabulary[term]]))
        for term in counts
        if counts[term] >= min_groups and term in ix.vocabulary
        and not generic[ix.vocabulary[term]] and not term.isdigit()
    ]
    candidates.sort(key=lambda row: (-row[1], -row[2]))
    return candidates[:12]


@dataclass
class FusedDocs:
    """Salida de la fase de fusión: unidades, grupos y tamaños del año."""

    doc_ids: list
    groups: list
    units: list
    words_original: int
    year: str
    speaker: str = ""
    generic_mask: np.ndarray | None = None
    year_weights: np.ndarray | None = None


def prepare_year(ix, year, speaker: str = None, *, docs_per_group: int = None,
                 per_doc_sentences: int = None, n_clusters: int = None,
                 method: str = "hybrid", verbose: bool = False) -> FusedDocs:
    """Agrupa el año y fusiona los resúmenes individuales (fase costosa).

    Separar esta fase permite evaluar varias configuraciones de IDF/selección
    sobre el mismo material sin repetir el agrupamiento ni los resúmenes.
    """
    from ri_scripts.corpus import read_doc

    ix = ix or _load_index_cached()
    doc_ids = docs_for_year(ix, year, speaker)
    if len(doc_ids) < config.YEAR_MIN_DOCS:
        raise ValueError(
            f"{speaker or 'todos'} {year}: solo {len(doc_ids)} discursos "
            f"(mínimo {config.YEAR_MIN_DOCS})"
        )

    docs_per_group = config.DOCS_PER_GROUP if docs_per_group is None else docs_per_group
    per_doc_sentences = (config.PER_DOC_SENTENCES if per_doc_sentences is None
                         else per_doc_sentences)

    df_docs = term_doc_df(ix, doc_ids)
    groups, stats = cluster_documents(
        ix, doc_ids, k=n_clusters, df_docs=df_docs, return_stats=True, verbose=verbose)

    # sin agrupamiento (1 grupo) hay que tomar más documentos para tener
    # material comparable; si no, la ablación mide el tamaño del pool, no el
    # efecto de agrupar
    if len(groups) == 1:
        docs_per_group = max(docs_per_group, min(10, len(groups[0].doc_ids)))

    units = []
    for group in groups:
        medoids = select_representative_docs(group.doc_ids, k=docs_per_group, ix=ix)
        for doc_id in medoids:
            doc = read_doc(int(doc_id), ix.corpus_path, ix.offsets)
            summary = summarize_doc(
                int(doc_id), ix=ix, method=method, ratio=1.0, min_sentences=1,
                max_sentences=per_doc_sentences, word_budget=10 ** 9,
            )
            for sent in summary.sentences:
                unit = make_unit(
                    sent["text"], idx=len(units),
                    position=0.5,          # neutraliza el prior de posición
                    doc_id=int(doc_id),
                    doc_label=doc["date"],
                    group_id=group.group_id,
                )
                if unit is not None:
                    units.append(unit)

    words_original = sum(_doc_words(ix, d) for d in doc_ids)
    generic_mask = df_docs >= config.GENERIC_DF_RATIO * len(doc_ids)
    return FusedDocs(doc_ids=doc_ids, groups=groups, units=units,
                     words_original=words_original, year=str(year),
                     speaker=speaker or "", generic_mask=generic_mask,
                     year_weights=stats["year_weights"])


def fuse_year(ix, fused: FusedDocs, *, idf_mode: str = None,
              max_sentences: int = None, word_budget: int = None,
              lam: float = None, year_weight: float = None,
              use_quotas: bool = True, quota_min_share: float = None,
              quota_min_raw: float = None, method: str = "hybrid") -> Summary:
    """Puntúa y selecciona sobre las unidades fusionadas (fase barata)."""
    idf_mode = config.MULTI_IDF_MODE if idf_mode is None else idf_mode
    max_sentences = config.MULTI_MAX_SENTENCES if max_sentences is None else max_sentences
    word_budget = config.MULTI_WORD_BUDGET if word_budget is None else word_budget

    groups, units, doc_ids = fused.groups, fused.units, fused.doc_ids
    if not units:
        return Summary(text="", method=method, stats={
            "n_docs": len(doc_ids), "n_groups": len(groups)})

    rep = build_representation(units, idf_mode=idf_mode)
    scores = all_scores(rep, "", extra_penalties=True)
    scores["year_affinity"] = year_affinity(ix, fused, units)
    weights = dict(config.WEIGHTS)
    weights["year_affinity"] = (config.MULTI_YEAR_WEIGHT
                                if year_weight is None else year_weight)
    relevance = hybrid_score(scores, weights)

    n_groups = len(groups)
    k = int(min(max_sentences, max(n_groups, np.ceil(0.45 * len(units)))))

    # cuota proporcional al tamaño del grupo: los temas grandes aportan más
    # oraciones, los anecdóticos no ocupan un lugar fijo
    quotas = _allocate_quotas(groups, k, min_share=quota_min_share,
                              min_raw=quota_min_raw)
    # cobertura temática garantizada sólo para grupos significativos: un tema
    # marginal (3-11 discursos) no debe desplazar a los dominantes
    chosen = select_mmr(
        relevance, rep.sim, k, units, lam=lam, word_budget=word_budget,
        groups=[u.group_id for u in units],
        max_per_group=quotas if use_quotas else None,
    )

    group_counts = collections.Counter(units[i].group_id for i in chosen)
    words_summary = sum(units[i].n_words for i in chosen)
    stats = {
        "n_docs": len(doc_ids),
        "n_groups": n_groups,
        "n_units": len(units),
        "n_selected": len(chosen),
        "words_original": fused.words_original,
        "words_summary": words_summary,
        "compression": words_summary / max(1, fused.words_original),
        "reduction": 1.0 - words_summary / max(1, fused.words_original),
        "coverage": _year_coverage(ix, fused, units, chosen),
        "theme_coverage": (sum(g.size for g in groups
                              if group_counts.get(g.group_id, 0) > 0)
                           / max(1, len(doc_ids))),
        "redundancy": _redundancy(rep, chosen),
        "mean_sent_words": float(np.mean([units[i].n_words for i in chosen])) if chosen else 0.0,
        "docs_represented": len({units[i].doc_id for i in chosen}),
        "per_group_selected": {
            g.group_id: int(group_counts.get(g.group_id, 0)) for g in groups
        },
        "idf_mode": rep.idf_source,
        "doc_ids": [int(d) for d in doc_ids],
    }

    sentences = [{
        "idx": i,
        "group_id": units[i].group_id,
        "group_label": groups[units[i].group_id].label,
        "doc_id": units[i].doc_id,
        "date": units[i].doc_label,
        "text": units[i].text,
        "score": float(relevance[i]),
    } for i in sorted(chosen)]

    return Summary(
        text=render(units, chosen),
        method=method,
        sentences=sentences,
        stats=stats,
        component_scores={name: [float(v) for v in values]
                          for name, values in scores.items()
                          if name not in ("cue_hits", "penalty")},
        common_terms=_common_terms(ix, doc_ids, groups, fused.year_weights),
        rep=rep,
        groups=[{
            "group_id": g.group_id,
            "size": g.size,
            "label": g.label,
            "top_terms": g.top_terms,
            "doc_ids": g.doc_ids,
            "selected": int(group_counts.get(g.group_id, 0)),
        } for g in groups],
        idf_mode=rep.idf_source,
        title=f"{fused.speaker or 'todos los hablantes'} · {fused.year}",
        speaker=fused.speaker,
        date=fused.year,
    )


def summarize_year(ix, year, speaker: str = None, *, idf_mode: str = None,
                   n_clusters: int = None, docs_per_group: int = None,
                   per_doc_sentences: int = None, max_sentences: int = None,
                   word_budget: int = None, lam: float = None,
                   method: str = "hybrid", verbose: bool = False) -> Summary:
    """Resumen multi-documento de un año (o de un hablante en un año)."""
    fused = prepare_year(
        ix, year, speaker, docs_per_group=docs_per_group,
        per_doc_sentences=per_doc_sentences, n_clusters=n_clusters,
        method=method, verbose=verbose,
    )
    return fuse_year(ix, fused, idf_mode=idf_mode, max_sentences=max_sentences,
                     word_budget=word_budget, lam=lam, method=method)
