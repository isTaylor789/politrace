"""Coherencia de tópicos implementada desde cero: PMI/NPMI, c_v y UMass.

No se usa gensim ni ninguna caja negra: la métrica que decide el número de
tópicos es la misma que se explica en el reporte.

**Espacio de pares.** Para cada documento se toman los tokens consecutivos y se
cuenta todo par de tokens a distancia ``d < W`` (``W = 10``). El espacio de
muestreo es el conjunto de pares ordenados, de tamaño

    T = 2 · Σ_{d=1}^{W-1} (N - d)

donde ``N`` es el número de tokens del corpus. Con ``m_ij`` el número de pares
ordenados en los que el token de la izquierda es ``i`` y el de la derecha ``j``:

    p_ij = m_ij / T ,      p_i = Σ_j p_ij / T ,
    PMI(i,j) = log( p_ij / (p_i p_j) ) ,
    NPMI(i,j) = PMI(i,j) / ( -log p_ij )  ∈ [-1, 1] .

NPMI normaliza el PMI para que sea comparable entre términos de frecuencia muy
distinta; es la base de la medida ``c_v`` (Röder et al., 2015), que compara
tópicos por la **confirmación semántica** entre sus palabras: se toma el vector
de NPMI de cada término restringido a las palabras del tópico y se promedia el
coseno entre pares.

**UMass** (Mimno et al., 2011) usa co-ocurrencia a nivel de documento:

    UMass = (2 / (n(n-1))) Σ_{i<j} log( (D(i,j) + 1) / D(i) ) ,

con los términos ordenados por frecuencia documental decreciente, ``D(i)`` = df
del término y ``D(i,j)`` = nº de documentos que contienen ambos. Premia tópicos
cuyas palabras aparecen juntas en los mismos discursos (no sólo cerca).

Ambas métricas se calculan sobre el top-n del tópico (n = 10). Cuanto más altas,
mejor: c_v ≈ 0.4-0.7 es un rango típico en corpus de discursos; valores cercanos
a 0 indican tópicos que son sólo listas de palabras frecuentes.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp

import mt_scripts.config as config
from mt_scripts.matrices import build_tokens


# --- Co-ocurrencia por ventana ----------------------------------------------
def load_tokens(ix, force: bool = False) -> dict:
    """Tokens del corpus como ids de vocabulario (cacheado en ``tokens.npz``)."""
    return build_tokens(ix, force=force, verbose=False)


def cooccurrence_counts(
    ids: np.ndarray,
    offsets: np.ndarray,
    cand_ids: np.ndarray,
    window: int = config.COOC_WINDOW,
    global_vocab_size: int = None,
) -> np.ndarray:
    """Conteos de pares **no ordenados** dentro de la ventana para los candidatos.

    Devuelve ``U`` de forma ``(C, C)`` con ``U[i,j]`` = nº de pares (i,j) a
    distancia ``1..window-1`` (incluye la diagonal). En vez de recorrer pares en
    Python se desplaza el vector de índices locales de cada documento: para cada
    desplazamiento ``d`` se comparan ``loc[:-d]`` y ``loc[d:]`` con NumPy y se
    cuentan los pares válidos con un único `bincount` por corpus.
    """
    C = len(cand_ids)
    V = int(global_vocab_size or (int(ids.max()) + 1 if len(ids) else 0))
    loc_map = np.full(V, -1, dtype=np.int32)
    loc_map[np.asarray(cand_ids, dtype=np.int64)] = np.arange(C, dtype=np.int32)

    chunks = []
    for doc in range(len(offsets) - 1):
        doc_ids = ids[offsets[doc]:offsets[doc + 1]]
        loc = loc_map[doc_ids]
        if not (loc >= 0).any():
            continue
        for d in range(1, window):
            if d >= len(loc):
                break
            a, b = loc[:-d], loc[d:]
            valid = (a >= 0) & (b >= 0)
            if not valid.any():
                continue
            u = np.minimum(a[valid], b[valid]).astype(np.int64)
            v = np.maximum(a[valid], b[valid]).astype(np.int64)
            chunks.append(u * C + v)

    if not chunks:
        return np.zeros((C, C), dtype=np.int64)
    flat = np.concatenate(chunks)
    return np.bincount(flat, minlength=C * C).reshape(C, C).astype(np.int64)


def npmi_from_counts(U: np.ndarray) -> tuple:
    """``(NPMI, M, p_i)`` a partir de los conteos no ordenados ``U``.

    ``M = U + Uᵀ − diag(U)`` son los conteos ordenados (cada par ``i≠j`` cuenta
    dos veces, una por orden). Con ``T = Σ M`` se obtienen las probabilidades
    conjuntas y marginales del espacio de pares.
    """
    M = U + U.T - np.diag(np.diag(U))
    M = M.astype(np.float64)
    T = M.sum()
    if T <= 0:
        return np.zeros_like(M), M, np.zeros(M.shape[0])
    P = M / T
    p_i = M.sum(axis=1) / T

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = P / np.outer(p_i, p_i)
        pmi = np.log(np.where(ratio > 0, ratio, 1.0))
        denom = -np.log(np.where(P > 0, P, 1.0))
        npmi = np.where((P > 0) & (denom > 0), pmi / denom, 0.0)
    np.fill_diagonal(npmi, 0.0)
    npmi = np.clip(npmi, -1.0, 1.0)
    return npmi, M, p_i


def _cosine_rows(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    B = A / norms
    return B @ B.T


def c_v(top_ids: list, npmi: np.ndarray, cand_ids: np.ndarray) -> float:
    """Coherencia ``c_v`` de un tópico dado por sus ids globales de vocabulario.

    ``cos(NPMI_i, NPMI_j)`` promediado sobre todos los pares de las top-n
    palabras del tópico. Los términos que no están en el conjunto de candidatos
    se ignoran (no debería ocurrir: los candidatos son la unión de los top-n de
    todos los modelos evaluados).
    """
    pos = {int(t): i for i, t in enumerate(cand_ids)}
    idx = [pos[int(t)] for t in top_ids if int(t) in pos]
    if len(idx) < 3:
        return 0.0
    S = npmi[np.ix_(idx, idx)]
    Cm = _cosine_rows(S)
    n = len(idx)
    iu = np.triu_indices(n, k=1)
    return float(Cm[iu].mean())


# --- UMass (nivel documento) -------------------------------------------------
def doc_cooccurrence(ix, cand_ids: np.ndarray) -> tuple:
    """``(D, df)``: co-ocurrencia a nivel documento entre los candidatos.

    ``D[i,j]`` = nº de documentos con ambos términos; ``df[i]`` = nº de
    documentos con el término. Se calcula con un producto disperso booleano
    sobre las filas correspondientes del índice invertido.
    """
    B = sp.csr_matrix(
        (np.ones(len(ix.postings_indices), dtype=np.float64), ix.postings_indices,
         ix.postings_indptr),
        shape=(ix.vocab_size, ix.n_docs),
    )
    sub = B[np.asarray(cand_ids, dtype=np.int64)]
    df = np.asarray(sub.sum(axis=1)).ravel()
    D = np.asarray((sub @ sub.T).todense())
    return D, df


def umass(top_ids: list, D: np.ndarray, df: np.ndarray, cand_ids: np.ndarray) -> float:
    """Coherencia UMass de un tópico (términos ordenados por df decreciente)."""
    pos = {int(t): i for i, t in enumerate(cand_ids)}
    idx = [pos[int(t)] for t in top_ids if int(t) in pos]
    if len(idx) < 2:
        return 0.0
    idx = sorted(idx, key=lambda i: -df[i])
    total, pairs = 0.0, 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            total += np.log((D[i, j] + 1.0) / max(df[i], 1.0))
            pairs += 1
    return float(total / max(pairs, 1))


# --- Evaluación de un modelo -------------------------------------------------
def evaluate_model(
    model,
    cooc: dict,
    cand_ids: np.ndarray,
    n_terms: int = config.COOC_TOP_TERMS,
) -> dict:
    """c_v, UMass, diversidad, exclusividad y peso medio del top-n del modelo."""
    from mt_scripts.models import topic_diversity

    cv, um, mean_w = [], [], []
    for t in range(model.k):
        top = model.top_ids(t, n_terms)
        cv.append(c_v(top, cooc["npmi"], cand_ids))
        um.append(umass(top, cooc["D"], cooc["df"], cand_ids))
        mean_w.append(float(model.top_weights(t, n_terms).mean()))
    return {
        "family": model.family,
        "k": model.k,
        "c_v": float(np.mean(cv)),
        "umass": float(np.mean(um)),
        "diversity": float(topic_diversity(model, config.TOP_TERMS_DIVERSITY)),
        "exclusivity": float(model.exclusivity().mean()),
        "mean_top_weight": float(np.mean(mean_w)),
        "c_v_per_topic": [float(x) for x in cv],
    }


def build_cooc(ix, cand_ids, *, window: int = config.COOC_WINDOW, verbose: bool = True) -> dict:
    """Construye (y cachea) NPMI y co-ocurrencia documental para los candidatos."""
    tokens = load_tokens(ix)
    U = cooccurrence_counts(
        tokens["ids"], tokens["offsets"], np.asarray(cand_ids),
        window=window, global_vocab_size=ix.vocab_size,
    )
    npmi, M, p_i = npmi_from_counts(U)
    D, df = doc_cooccurrence(ix, np.asarray(cand_ids))

    if verbose:
        print(f"co-ocurrencia: {len(cand_ids)} candidatos, ventana={window}, "
              f"pares dentro de ventana={int(U.sum()):,}")
    return {
        "cand_ids": np.asarray(cand_ids),
        "U": U, "M": M, "npmi": npmi, "p_i": p_i, "D": D, "df": df,
        "window": window,
    }


def save_cooc(cooc: dict, path: Path = config.COOC_NPZ) -> None:
    """Guarda la matriz NPMI y la co-ocurrencia documental (reproducibilidad)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cand_ids=cooc["cand_ids"],
        npmi=cooc["npmi"].astype(np.float32),
        U=cooc["U"].astype(np.int32),
        D=cooc["D"].astype(np.int32),
        df=cooc["df"].astype(np.int32),
        window=np.array([cooc["window"]]),
    )


def strongest_pairs(cooc: dict, ix, n: int = 20) -> list:
    """Pares de términos con mayor NPMI (para la sección cualitativa del reporte)."""
    npmi, cand = cooc["npmi"], cooc["cand_ids"]
    iu = np.triu_indices(len(cand), k=1)
    vals = npmi[iu]
    order = np.argsort(-vals)[:n]
    rows = []
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        rows.append({
            "term_a": ix.terms[int(cand[i])],
            "term_b": ix.terms[int(cand[j])],
            "npmi": float(vals[k]),
            "pairs": int(cooc["M"][i, j]),
        })
    return rows
