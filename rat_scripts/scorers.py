"""Puntuadores de oración (temario 4.4.1 estadísticos, 4.4.2 selección,
4.4.4 discursivos).

Cada función devuelve un vector de longitud `n_units` con la importancia de cada
oración según un criterio. Todas son deterministas y sólo usan álgebra lineal
clásica: TF-IDF, IDF, PageRank sobre un grafo de similitud y SVD truncada.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD

from rat_scripts import config


# --- Normalización -----------------------------------------------------------
def standardize(x: np.ndarray) -> np.ndarray:
    """z-score robusto (se recorta a ±3σ) seguido de min-max a [0,1].

    El z-score evita que un componente con valores muy grandes domine la suma
    ponderada; el recorte impide que un único valor atípico aplaste al resto.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    std = x.std()
    if std <= 1e-12:
        return np.zeros_like(x)
    z = np.clip((x - x.mean()) / std, -3.0, 3.0)
    lo, hi = z.min(), z.max()
    return (z - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(z)


# --- 4.4.1 Estadísticos ------------------------------------------------------
def score_tfidf(rep, top_n: int = 5) -> np.ndarray:
    """Media de los `top_n` pesos TF-IDF más altos de la oración.

        s_tfidf(i) = (1/k) Σ_{j=1..k} w_ij       (pesos ordenados desc.)

    Es la variante clásica "suma de pesos" pero promediada en la cabeza de la
    distribución, para que una oración larga no gane sólo por ser larga.
    """
    W = rep.W.tocsr()
    scores = np.zeros(rep.n_units, dtype=float)
    for i in range(rep.n_units):
        row = W.data[W.indptr[i]:W.indptr[i + 1]]
        if row.size == 0:
            continue
        k = min(top_n, row.size)
        scores[i] = np.sort(row)[-k:].mean()
    return scores


def score_centroid(rep) -> np.ndarray:
    """Coseno entre la oración y el centroide del documento.

        c = Σ_i w_i / ||Σ_i w_i|| ,      s_cent(i) = cos(w_i, c)

    Las filas de W ya son unitarias, así que el coseno es el producto punto con
    el centroide unitario.
    """
    c = rep.centroid()
    return np.asarray(rep.W @ c).ravel()


def score_luhn(rep, low: float = 0.05, high: float = 0.70) -> np.ndarray:
    """Método de Luhn: palabras significativas + densidad.

    Una palabra es significativa si su frecuencia documental (número de
    oraciones donde aparece) cae en la banda ``[low·n, high·n]``: por encima de
    ``high`` es una palabra ubicua (no distingue), por debajo de ``low`` es un
    hápax (ruido). El peso de cada palabra significativa es su IDF.

        s_luhn(i) = Σ_{t ∈ sig ∩ i} idf_t / sqrt(span_i)

    con ``span_i`` = distancia, en términos, entre la primera y la última palabra
    significativa de la oración. Dividir por el span premia las oraciones donde
    las palabras clave están concentradas.
    """
    n = rep.n_units
    if n == 0:
        return np.zeros(0)
    df = np.asarray((rep.X > 0).sum(axis=0)).ravel()
    lo = max(1, int(np.ceil(low * n)))
    hi = max(lo + 1, int(np.floor(high * n)))
    sig = (df >= lo) & (df <= hi)

    X, idf = rep.X.tocsr(), rep.idf
    out = np.zeros(n, dtype=float)
    for i, unit in enumerate(rep.units):
        positions, weight = [], 0.0
        for pos, term in enumerate(unit.terms):
            j = rep.vocab.get(term)
            if j is None or not sig[j]:
                continue
            positions.append(pos)
            weight += idf[j]
        if positions:
            span = positions[-1] - positions[0] + 1
            out[i] = weight / np.sqrt(span)
    return out


# --- 4.4.2 Selección basada en grafos ---------------------------------------
def pagerank(weights: np.ndarray, damping: float = None, tol: float = None,
             max_iter: int = None) -> np.ndarray:
    """PageRank por iteración de potencias sobre la matriz de adyacencia.

        p^{t+1} = (1-d)/n · 1 + d · Mᵀ p^t

    con ``M`` la matriz de transición por filas (filas sin aristas → uniforme,
    que es el tratamiento estándar de nodos colgantes).
    """
    damping = config.PAGERANK_DAMPING if damping is None else damping
    tol = config.PAGERANK_TOL if tol is None else tol
    max_iter = config.PAGERANK_MAX_ITER if max_iter is None else max_iter

    n = weights.shape[0]
    if n == 0:
        return np.zeros(0)
    M = np.array(weights, dtype=float, copy=True)
    np.fill_diagonal(M, 0.0)
    M[~np.isfinite(M)] = 0.0
    np.clip(M, 0.0, None, out=M)

    row = M.sum(axis=1)
    dangling = row <= 0
    M[~dangling] /= row[~dangling, None]
    M[dangling] = 1.0 / n

    p = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = (1.0 - damping) / n + damping * (M.T @ p)
        if np.abs(new - p).sum() < tol:
            return new
        p = new
    return p


def score_textrank(rep, window: int = None) -> np.ndarray:
    """TextRank: PageRank sobre el grafo de similitud coseno entre oraciones.

    Arista ``sim(i,j) = cos(w_i, w_j)`` (grafo completo o ventana de proximidad
    si ``window > 0``). Se usa la similitud como peso de la arista.
    """
    window = config.TEXTRANK_WINDOW if window is None else window
    sim = rep.sim
    if window and window > 0:
        n = sim.shape[0]
        idx = np.arange(n)
        mask = np.abs(idx[:, None] - idx[None, :]) > window
        sim = np.where(mask, 0.0, sim)
    return pagerank(sim)


def score_lexrank(rep, threshold: float = None) -> np.ndarray:
    """LexRank: centralidad de grado sobre el grafo de similitud binarizado.

        A_ij = 1[cos(w_i, w_j) ≥ θ] ,     s_lex(i) = deg(i) / (n - 1)

    θ = 0.10 (config). Es la versión "de grado" del LexRank original; la versión
    con PageRank sobre A es prácticamente idéntica aquí porque A es simétrica.
    """
    threshold = config.LEXRANK_THRESHOLD if threshold is None else threshold
    n = rep.n_units
    if n <= 1:
        return np.zeros(n)
    A = (rep.sim >= threshold).astype(float)
    np.fill_diagonal(A, 0.0)
    return A.sum(axis=1) / (n - 1)


# --- 4.4.6 Otros: LSA --------------------------------------------------------
def score_lsa(rep, k: int = None) -> np.ndarray:
    """Importancia por espacio latente (Steinberger & Ježek, variante σ).

    Sea ``W = U Σ Vᵀ`` la SVD truncada a ``k`` componentes. La puntuación de la
    oración ``j`` es su masa en los componentes, ponderada por el valor singular:

        s_lsa(j) = Σ_{i=1..k} σ_i · u_{ji}² / Σ_{i=1..k} σ_i

    Queda en [0,1] y mide cuánto participa la oración del/los tópicos
    dominantes del documento.
    """
    n = rep.n_units
    if n < 2 or rep.n_terms < 2:
        return np.zeros(n)
    k = config.LSA_K if k is None else k
    # con pocas oraciones una SVD casi completa reconstruye la matriz original y
    # la puntuación se vuelve ruido; se limita también de forma proporcional.
    k = max(1, min(k, int(np.ceil(0.30 * n)), n - 1, rep.n_terms - 1))

    svd = TruncatedSVD(n_components=k, n_iter=10, random_state=42)
    proj = svd.fit_transform(rep.W)          # proj = U Σ
    sigma = svd.singular_values_
    u = np.divide(proj, sigma, out=np.zeros_like(proj), where=sigma > 1e-12)
    denom = sigma.sum()
    if denom <= 0:
        return np.zeros(n)
    return (u ** 2 * sigma).sum(axis=1) / denom


# --- Orquestador -------------------------------------------------------------
def all_scores(rep, title: str = "", extra_penalties: bool = False) -> dict:
    """Todos los componentes, sin normalizar (la normalización es del híbrido)."""
    from rat_scripts.features import discourse_scores

    scores = {
        "tfidf": score_tfidf(rep),
        "centroid": score_centroid(rep),
        "luhn": score_luhn(rep),
        "textrank": score_textrank(rep),
        "lexrank": score_lexrank(rep),
        "lsa": score_lsa(rep),
    }
    scores.update(discourse_scores(rep.units, title, extra_penalties))
    return scores


def hybrid_score(scores: dict, weights: dict = None) -> np.ndarray:
    """Combinación lineal de componentes normalizados.

        S(i) = Σ_c λ_c · ẑ_c(i)

    donde ẑ_c es el componente c estandarizado y reescalado a [0,1]. Los pesos
    λ_c viven en `config.WEIGHTS` y suman 1.
    """
    weights = config.WEIGHTS if weights is None else weights
    total = None
    used = 0.0
    for name, weight in weights.items():
        if weight <= 0 or name not in scores:
            continue
        component = standardize(scores[name]) * weight
        total = component if total is None else total + component
        used += weight
    if total is None:
        return np.zeros(len(scores.get("centroid", [])))
    return total / used if used > 0 else total
