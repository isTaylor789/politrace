"""Representación vectorial de las oraciones (bolsa de términos + TF-IDF).

Reutiliza el IDF **global** del índice de RI (`ri_scripts/artifacts/idf.npy` y
`vocabulary.json`) cuando se resume un discurso del corpus: así una palabra
común en todo el corpus presidencial pesa menos que una rara, igual que en la
búsqueda. Cuando se resume un texto cualquiera (fuera del corpus) se calcula un
IDF local sobre las propias oraciones del texto.
"""

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from ri_scripts import config as ri_config

from rat_scripts import config


@dataclass
class Representation:
    """Matrices y utilidades sobre las unidades de un documento."""

    units: list
    terms: list                 # id de columna -> término
    vocab: dict                 # término -> id de columna
    X: sp.csr_matrix            # conteos (n_oraciones x n_términos)
    idf: np.ndarray             # IDF usado por cada término
    W: sp.csr_matrix            # TF-IDF, filas L2 = 1
    sim: np.ndarray             # matriz de similitud coseno (densa, 1 = máx)
    idf_source: str             # "global" o "local"

    @property
    def n_units(self) -> int:
        return self.X.shape[0]

    @property
    def n_terms(self) -> int:
        return self.X.shape[1]

    def centroid(self) -> np.ndarray:
        """Centroide del documento (media de los vectores de oración, L2 = 1)."""
        c = np.asarray(self.W.mean(axis=0)).ravel()
        norm = np.linalg.norm(c)
        return c / norm if norm > 0 else c

    def term_weights(self) -> np.ndarray:
        """Peso total de cada término en el documento (para cobertura)."""
        return np.asarray(self.W.sum(axis=0)).ravel()

    def top_terms(self, k: int = 20) -> list:
        order = np.argsort(-self.term_weights(), kind="mergesort")[:k]
        weights = self.term_weights()
        return [(self.terms[i], float(weights[i])) for i in order if weights[i] > 0]


@lru_cache(maxsize=1)
def global_idf() -> tuple:
    """(dict término -> idf, idf por defecto para términos desconocidos)."""
    with open(ri_config.VOCABULARY_JSON, "r", encoding="utf-8") as fh:
        vocab = json.load(fh)
    idf = np.load(ri_config.IDF_NPY)
    table = {term: float(idf[j]) for term, j in vocab.items()}
    # Término fuera del vocabulario global (df < min_df): se le asigna el IDF
    # máximo observado, que es lo que le correspondería por ser "muy raro".
    return table, float(idf.max())


def build_representation(units: list, *, use_global_idf: bool = True,
                         idf_mode: str = None, blend_alpha: float = None) -> Representation:
    """Construye conteos, IDF, TF-IDF normalizado y la matriz de similitud."""
    n = len(units)
    vocab: dict = {}
    rows, cols, data = [], [], []
    for i, unit in enumerate(units):
        for term in unit.terms:
            j = vocab.get(term)
            if j is None:
                j = len(vocab)
                vocab[term] = j
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    V = len(vocab)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n, V), dtype=np.float64)
    # colapsa duplicados por si el analizador repite un término
    X.sum_duplicates()

    terms = [None] * V
    for term, j in vocab.items():
        terms[j] = term

    # IDF local (siempre se calcula: es barato y lo necesita el modo "blend")
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf_local = np.log((1.0 + n) / (1.0 + df)) + 1.0

    mode = idf_mode or ("global" if use_global_idf else "local")
    if mode not in ("global", "local", "blend"):
        raise ValueError(f"idf_mode desconocido: {mode}")

    if mode == "local" or n == 0:
        idf, source = idf_local, "local"
    else:
        table, default = global_idf()
        idf_global = np.array([table.get(t, default) for t in terms], dtype=float)
        if mode == "global":
            idf, source = idf_global, "global"
        else:
            alpha = config.IDF_BLEND_ALPHA if blend_alpha is None else blend_alpha
            idf = alpha * idf_global + (1.0 - alpha) * idf_local
            source = f"blend({alpha:.2f})"

    W = X.copy()
    W.data *= idf[W.indices]
    norms = np.sqrt(np.asarray(W.multiply(W).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    W = sp.diags(1.0 / norms) @ W
    W = W.tocsr()

    # coseno = producto punto de filas unitarias; con pocas oraciones (cientos)
    # la matriz densa es barata y simplifica TextRank/LexRank/MMR.
    sim = np.asarray((W @ W.T).todense())
    np.fill_diagonal(sim, 0.0)
    np.clip(sim, 0.0, 1.0, out=sim)

    return Representation(units, terms, vocab, X.tocsr(), idf, W, sim, source)
