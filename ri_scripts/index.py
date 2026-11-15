"""Índice invertido (postings) + matriz TF-IDF global (`w_global`) + stats BM25.

Artefactos que quedan en `ri_scripts/artifacts/`:

- ``corpus.jsonl``      un documento por línea (texto completo)
- ``corpus_offsets.npy``offset de byte de cada línea (acceso aleatorio)
- ``postings.npz``      índice invertido: término -> (docs, tf)
- ``w_global.npz``      matriz TF-IDF doc x término, filas normalizadas L2
- ``idf.npy``           IDF global (una entrada por término)
- ``bm25_stats.npz``    df, longitud de cada doc y longitud media
- ``vocabulary.json``   término -> id de columna
- ``index_meta.json``   metadatos de cada doc y parámetros del índice
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from ri_scripts import config
from ri_scripts.corpus import iter_corpus
from ri_scripts.preprocess import analyze


@dataclass
class Index:
    """Estructuras cargadas en memoria que usan los tres buscadores."""

    doc_ids: list          # speech_id de cada doc_id posicional
    speakers: list         # hablante de cada doc_id
    dates: list            # fecha ISO de cada doc_id
    titles: list           # título de cada doc_id
    vocabulary: dict       # término -> id de columna
    terms: list            # id de columna -> término
    postings_indptr: np.ndarray
    postings_indices: np.ndarray
    postings_tf: np.ndarray
    df: np.ndarray
    idf: np.ndarray
    bm25_idf: np.ndarray
    dl: np.ndarray
    avgdl: float
    w_global: sp.csr_matrix
    speaker_blocks: dict
    k1: float = config.BM25_K1
    b: float = config.BM25_B
    corpus_path: Path = config.CORPUS_JSONL
    offsets: np.ndarray | None = None

    # --- utilidades ----------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.terms)

    def term_id(self, term: str) -> int | None:
        return self.vocabulary.get(term)

    def postings(self, term_id: int) -> tuple[np.ndarray, np.ndarray]:
        """(doc_ids, tf) del término, en orden ascendente de doc_id."""
        start, end = self.postings_indptr[term_id], self.postings_indptr[term_id + 1]
        return self.postings_indices[start:end], self.postings_tf[start:end]

    def docs_of_speaker(self, speaker: str) -> np.ndarray:
        start, end = self.speaker_blocks[speaker]
        return np.arange(start, end)


# --- Construcción -------------------------------------------------------------
def build_index(verbose: bool = True) -> Index:
    """Recorre el corpus, construye postings, TF-IDF global y stats de BM25."""
    records = list(iter_corpus(config.CORPUS_JSONL))
    texts = [r["text"] for r in records]
    n_docs = len(texts)

    vectorizer = CountVectorizer(
        analyzer=analyze,      # mismo analizador que en consulta
        lowercase=False,       # ya normalizado dentro de `analyze`
        min_df=config.MIN_DF,
        dtype=np.int32,
    )
    X = vectorizer.fit_transform(texts)            # CSR (doc, término) = conteos
    terms = vectorizer.get_feature_names_out().tolist()
    vocabulary = {t: i for i, t in enumerate(terms)}
    V = len(terms)

    # df y IDF global (suavizado tipo sklearn: ln((1+N)/(1+df)) + 1)
    df = X.getnnz(axis=0).astype(np.int64)
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0

    # w_global = tf * idf, filas normalizadas L2 -> coseno = producto punto
    # En CSR los elementos se agrupan por fila, así que el IDF de cada valor
    # almacenado es idf[columna].
    W = X.astype(np.float64)
    W.data *= idf[W.indices]
    W = normalize(W, norm="l2", copy=False).tocsr()

    # Índice invertido: la transpuesta en CSR es exactamente término -> docs
    Xt = X.T.tocsr()
    Xt.sort_indices()

    dl = np.asarray(X.sum(axis=1)).ravel().astype(np.int64)
    avgdl = float(dl.mean())
    bm25_idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    # Bloques contiguos por hablante (el corpus se ordenó así)
    speakers = [r["speaker"] for r in records]
    speaker_blocks: dict[str, tuple[int, int]] = {}
    for i, spk in enumerate(speakers):
        if spk not in speaker_blocks:
            speaker_blocks[spk] = (i, i + 1)
        else:
            speaker_blocks[spk] = (speaker_blocks[spk][0], i + 1)

    ix = Index(
        doc_ids=[r["speech_id"] for r in records],
        speakers=speakers,
        dates=[r["date"] for r in records],
        titles=[r["title"] for r in records],
        vocabulary=vocabulary,
        terms=terms,
        postings_indptr=Xt.indptr.astype(np.int64),
        postings_indices=Xt.indices.astype(np.int64),
        postings_tf=Xt.data.astype(np.int64),
        df=df,
        idf=idf,
        bm25_idf=bm25_idf,
        dl=dl,
        avgdl=avgdl,
        w_global=W,
        speaker_blocks=speaker_blocks,
    )

    if verbose:
        print(f"docs={n_docs}  vocabulario={V}  nnz(tf-idf)={W.nnz}")
        print(f"avgdl={avgdl:.1f} tokens  min_df={config.MIN_DF}")
    return ix


# --- Persistencia -------------------------------------------------------------
def save_index(ix: Index, artifacts: Path = config.ARTIFACTS) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)

    sp.save_npz(config.W_GLOBAL_NPZ, ix.w_global.tocsr())
    np.savez(
        config.POSTINGS_NPZ,
        indptr=ix.postings_indptr,
        indices=ix.postings_indices,
        tf=ix.postings_tf,
    )
    np.savez(
        config.BM25_STATS_NPZ,
        df=ix.df,
        dl=ix.dl,
        avgdl=np.array([ix.avgdl]),
        bm25_idf=ix.bm25_idf,
    )
    np.save(config.IDF_NPY, ix.idf)

    with open(config.VOCABULARY_JSON, "w", encoding="utf-8") as fh:
        json.dump(ix.vocabulary, fh, ensure_ascii=False)

    meta = {
        "n_docs": ix.n_docs,
        "vocab_size": ix.vocab_size,
        "min_df": config.MIN_DF,
        "k1": ix.k1,
        "b": ix.b,
        "avgdl": ix.avgdl,
        "nnz": int(ix.w_global.nnz),
        "doc_ids": ix.doc_ids,
        "speakers": ix.speakers,
        "dates": ix.dates,
        "titles": ix.titles,
        "speaker_blocks": {k: list(v) for k, v in ix.speaker_blocks.items()},
    }
    with open(config.INDEX_META_JSON, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)


def load_index(artifacts: Path = config.ARTIFACTS, use_offsets: bool = True) -> Index:
    with open(config.INDEX_META_JSON, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(config.VOCABULARY_JSON, "r", encoding="utf-8") as fh:
        vocabulary = json.load(fh)

    post = np.load(config.POSTINGS_NPZ)
    stats = np.load(config.BM25_STATS_NPZ)

    terms = [None] * len(vocabulary)
    for term, j in vocabulary.items():
        terms[j] = term

    return Index(
        doc_ids=meta["doc_ids"],
        speakers=meta["speakers"],
        dates=meta["dates"],
        titles=meta["titles"],
        vocabulary=vocabulary,
        terms=terms,
        postings_indptr=post["indptr"],
        postings_indices=post["indices"],
        postings_tf=post["tf"],
        df=stats["df"],
        idf=np.load(config.IDF_NPY),
        bm25_idf=stats["bm25_idf"],
        dl=stats["dl"],
        avgdl=float(stats["avgdl"][0]),
        w_global=sp.load_npz(config.W_GLOBAL_NPZ).tocsr(),
        speaker_blocks={k: tuple(v) for k, v in meta["speaker_blocks"].items()},
        k1=meta["k1"],
        b=meta["b"],
        offsets=np.load(config.CORPUS_OFFSETS) if use_offsets else None,
    )


def artifacts_exist() -> bool:
    return all(
        p.exists()
        for p in (
            config.CORPUS_JSONL,
            config.CORPUS_OFFSETS,
            config.POSTINGS_NPZ,
            config.W_GLOBAL_NPZ,
            config.IDF_NPY,
            config.BM25_STATS_NPZ,
            config.VOCABULARY_JSON,
            config.INDEX_META_JSON,
        )
    )
