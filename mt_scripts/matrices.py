"""Representaciones matriciales para MT, reconstruidas desde el índice de RI.

El índice de RI deja tres artefactos que bastan para no re-tokenizar nada:

- ``postings.npz``  forma término -> documentos con ``tf`` (matriz CSR por filas
  de término, ``shape = (V, N)``);
- ``idf.npy``       IDF global por término;
- ``vocabulary.json`` término -> columna.

De ahí salen **tres** matrices de trabajo, y esta distinción es la decisión
técnica central del módulo:

===========  ==============================================  =================
matriz       definición                                       uso
===========  ==============================================  =================
``X``        conteos crudos ``tf_ij``                         LDA, PMI, UMass
``W``        ``tf_ij * idf_j`` (sin normalizar por fila)      NMF, keyness
``W_l2``     ``W`` con filas L2 = 1                           coseno, SVD, ...
===========  ==============================================  =================

`w_global` (el artefacto del módulo 1) **no** se usa como entrada de los modelos
porque sus filas ya están normalizadas L2: eso borra la longitud del documento y
aplana los discursos largos (AMLO llega a 20k palabras). Se conserva, eso sí,
como verificación: `verify_reconstruction` comprueba que reconstruir desde
``postings + idf`` y normalizar reproduce `w_global` exactamente.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import mt_scripts.config as config
from ri_scripts.corpus import iter_corpus
from ri_scripts.preprocess import analyze, fold, stem

@dataclass
class Space:
    """Matriz de trabajo de un (sub)conjunto de documentos.

    ``terms_ids`` son columnas **globales** del vocabulario de RI: así todos los
    modelos viven en el mismo espacio y se pueden comparar/alinear sin remapear.
    """

    doc_ids: np.ndarray          # doc_id (posición en el corpus de RI)
    terms_ids: np.ndarray        # columna global de cada término conservado
    terms: list                  # stem de cada columna
    X: sp.csr_matrix             # conteos      (n_docs, n_terms)
    W: sp.csr_matrix             # tf*idf       (n_docs, n_terms)
    W_l2: sp.csr_matrix          # tf*idf L2    (n_docs, n_terms)
    df: np.ndarray               # df local de cada columna
    idf: np.ndarray              # idf global de cada columna
    global_vocab_size: int

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def n_terms(self) -> int:
        return len(self.terms_ids)

    def corpus_tf(self) -> np.ndarray:
        """tf total de cada columna en el subconjunto (para keyness y top terms)."""
        return np.asarray(self.X.sum(axis=0)).ravel()


# --- Construcción de las matrices -------------------------------------------
def global_counts(ix) -> sp.csr_matrix:
    """Matriz CSR ``(V, N)`` término -> documentos, directamente desde postings.

    `postings.npz` guarda exactamente el CSR transpuesto de la matriz de conteos:
    ``indptr`` por término, ``indices`` con los documentos y ``tf`` con el valor.
    No hay copia ni reconstrucción: es la misma estructura del índice invertido.
    """
    return sp.csr_matrix(
        (ix.postings_tf.astype(np.float64), ix.postings_indices, ix.postings_indptr),
        shape=(ix.vocab_size, ix.n_docs),
    )


def _clean_term(term: str) -> bool:
    """Término admisible como tópico: sin dígitos, con cuerpo suficiente."""
    if len(term) < config.MIN_TERM_LEN:
        return False
    if config.DROP_DIGITS and any(ch.isdigit() for ch in term):
        return False
    return True


def select_columns(
    df: np.ndarray,
    idf: np.ndarray,
    terms: list,
    n_docs: int,
    *,
    min_df: int,
    max_df_ratio: float,
    max_terms: int,
    boilerplate: set = frozenset(),
) -> np.ndarray:
    """Índices de las columnas que entran al modelo (ordenados).

    Reglas, en este orden:

    1. ``df >= min_df`` (un término de un solo documento no es un tema);
    2. ``df <= max_df_ratio * n_docs`` (si está en la mitad de los discursos es
       plantilla: "presidente", "gobierno", "pueblo");
    3. el término es limpio (sin dígitos, longitud mínima);
    4. el término no está en la lista de puesta en escena
       (`config.BOILERPLATE_TERMS`: cargos, gentilicios de autorreferencia,
       nombres de los hablantes, dícticos); si no se filtra, los tópicos se
       agrupan por hablante en vez de por tema;
    5. si quedan más de ``max_terms``, se conservan los de mayor ``df * idf``
       —frecuentes pero distintivos—, no los más raros ni los más comunes.
    """
    df = np.asarray(df)
    keep = df >= min_df
    keep &= df <= max(1.0, max_df_ratio * n_docs)
    clean = np.fromiter((_clean_term(t) for t in terms), dtype=bool, count=len(terms))
    keep &= clean
    if boilerplate:
        keep &= np.fromiter((t not in boilerplate for t in terms), dtype=bool, count=len(terms))
    idx = np.flatnonzero(keep)
    if len(idx) > max_terms:
        score = df[idx] * idf[idx]
        order = np.argsort(-score, kind="mergesort")[:max_terms]
        idx = np.sort(idx[order])
    return idx


def build_space(
    ix,
    doc_ids=None,
    *,
    min_df: int = config.MIN_DF,
    max_df_ratio: float = config.MAX_DF_RATIO,
    max_terms: int = config.MAX_TERMS,
    use_global_idf: bool = True,
    drop_boilerplate: bool = None,
) -> Space:
    """Construye ``X``, ``W`` y ``W_l2`` para un subconjunto de documentos.

    ``doc_ids=None`` usa todo el corpus. Para hablantes/años se pasa el bloque y
    se rebajan los umbrales con `subspace` (ver `build_subspace`).
    """
    Xt = global_counts(ix)
    docs = np.arange(ix.n_docs) if doc_ids is None else np.asarray(doc_ids, dtype=np.int64)

    # df restringido al subconjunto: se cuenta sobre la CSR término -> docs.
    local = Xt if doc_ids is None else Xt[:, docs]
    df = np.asarray((local > 0).sum(axis=1)).ravel().astype(np.int64)

    drop_boilerplate = config.DROP_BOILERPLATE if drop_boilerplate is None else drop_boilerplate
    cols = select_columns(
        df, ix.idf, ix.terms, len(docs),
        min_df=min_df, max_df_ratio=max_df_ratio, max_terms=max_terms,
        boilerplate=config.BOILERPLATE_TERMS if drop_boilerplate else frozenset(),
    )

    X = local[cols].T.tocsr().astype(np.float64)
    X.sort_indices()

    idf = ix.idf[cols].astype(np.float64) if use_global_idf else _local_idf(df[cols], len(docs))
    W = X.copy()
    W.data *= idf[W.indices]
    W_l2 = normalize(W, norm="l2", copy=True).tocsr()

    return Space(
        doc_ids=docs,
        terms_ids=cols,
        terms=[ix.terms[j] for j in cols],
        X=X,
        W=W,
        W_l2=W_l2,
        df=df[cols],
        idf=idf,
        global_vocab_size=ix.vocab_size,
    )


def _local_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """IDF local con el mismo suavizado que RI (para textos/corpus fuera de RI)."""
    return np.log((1 + n_docs) / (1 + df)) + 1.0


def build_subspace(ix, doc_ids, *, max_terms: int = config.SUB_MAX_TERMS) -> Space:
    """Espacio de un subconjunto pequeño (hablante/año/ventana).

    Los umbrales bajan con el tamaño: Maduro tiene 30 discursos y con
    ``min_df = 10`` se quedaría sin vocabulario. Se exige
    ``df >= max(2, ceil(0.02 * n_docs))`` y un tope de términos menor.
    """
    docs = np.asarray(doc_ids, dtype=np.int64)
    n = len(docs)
    min_df = max(config.SUB_MIN_DF_ABS, int(np.ceil(config.SUB_MIN_DF_RATIO * n)))
    return build_space(
        ix, docs,
        min_df=min_df,
        max_df_ratio=config.SUB_MAX_DF_RATIO,
        max_terms=max_terms,
        drop_boilerplate=config.SUB_DROP_BOILERPLATE,
    )


# --- Verificación de la reutilización del índice de RI -----------------------
def verify_reconstruction(ix, tol: float = 1e-8) -> dict:
    """Comprueba que ``normalize(X * idf)`` reproduce `w_global` del módulo 1.

    Es la prueba de que MT usa el mismo índice y no una versión paralela: si
    esta diferencia es ~0, el IDF, el vocabulario y los conteos coinciden.
    """
    X = global_counts(ix).T.tocsr()   # (N, V): mismas filas que `w_global`
    W = X.copy()
    W.data *= ix.idf[W.indices]
    W = normalize(W, norm="l2", copy=False).tocsr()
    diff = abs(W - ix.w_global.tocsr())
    return {
        "max_abs_diff": float(diff.max()),
        "nnz_mt": int(W.nnz),
        "nnz_ri": int(ix.w_global.nnz),
        "ok": bool(diff.max() <= tol and W.nnz == ix.w_global.nnz),
    }


# --- Streams de tokens y formas legibles -------------------------------------
def build_tokens(ix, *, force: bool = False, verbose: bool = True) -> dict:
    """Guarda el corpus como ids de vocabulario (int32) + offsets por documento.

    Se usa para la coherencia c_v (ventana deslizante). Los términos fuera del
    vocabulario de RI (``df = 1``) se descartan: no pueden ser términos de un
    tópico y su PMI es degenerado. Se documenta el efecto (dos palabras que
    quedan contiguas al borrar una tercera): con ventana 10 es despreciable.
    """
    path = Path(config.TOKENS_NPZ)
    if path.exists() and not force:
        data = np.load(path)
        return {"ids": data["ids"], "offsets": data["offsets"],
                "n_tokens": int(data["n_tokens"][0])}

    vocab = ix.vocabulary
    chunks = []
    offsets = np.zeros(ix.n_docs + 1, dtype=np.int64)
    for i, record in enumerate(iter_corpus(config.CORPUS_JSONL)):
        ids = np.fromiter(
            (vocab[t] for t in analyze(record["text"]) if t in vocab),
            dtype=np.int32,
        )
        chunks.append(ids)
        offsets[i + 1] = offsets[i] + len(ids)
    flat = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=flat, offsets=offsets, n_tokens=np.array([len(flat)]))
    if verbose:
        print(f"tokens: {len(flat):,} ({len(flat) / 1e6:.1f}M) -> {path}")
    return {"ids": flat, "offsets": offsets, "n_tokens": len(flat)}


def build_stem_surface(ix, *, force: bool = False, verbose: bool = True) -> dict:
    """Mapa ``stem -> forma superficial más frecuente`` del corpus.

    Los tópicos salen como stems ("inflacion", "segur", "trabaj"). Para que las
    tablas se lean se guarda, por cada stem, la forma real que más aparece
    ("inflación", "seguridad", "trabajadores"). Es un artefacto de presentación:
    no interviene en ningún cálculo.
    """
    path = Path(config.STEM_SURFACE_JSON)
    if path.exists() and not force:
        import json

        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    import re
    from collections import Counter, defaultdict

    surface_re = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]*")
    counts = defaultdict(Counter)
    for record in iter_corpus(config.CORPUS_JSONL):
        for raw in surface_re.findall(record["text"]):
            if len(raw) < 3:
                continue
            folded = fold(raw)
            if any(ch.isdigit() for ch in folded):
                continue
            counts[stem(folded)][folded] += 1

    mapping = {s: c.most_common(1)[0][0] for s, c in counts.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, sort_keys=True)
    if verbose:
        print(f"stem -> superficie: {len(mapping):,} entradas -> {path}")
    return mapping


def load_reference(ix) -> dict:
    """Metadatos por `doc_id` que usan todas las tablas (hablante, fecha, título)."""
    return {
        "speaker": ix.speakers,
        "date": ix.dates,
        "title": ix.titles,
        "doc_ids": ix.doc_ids,
    }
