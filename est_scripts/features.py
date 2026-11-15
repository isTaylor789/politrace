"""Matriz de diseño del módulo 7-lite: cinco bloques de features y grupos de duplicados.

Los bloques se construyen **una vez** y se reutilizan en todas las ablaciones y folds:

| bloque | qué mide | representación |
|---|---|---|
| `estilo` | riqueza, oración, puntuación, deixis (≈50 rasgos) | densa, por documento |
| `funcion` | palabras de función (artículos, preposiciones, pronombres, auxiliares) | frecuencia por 1000 palabras |
| `caracteres` | estilo ortográfico y morfológico | TF-IDF de n-gramas de carácter 3-4 |
| `tema` | **contenido** (control) | TF-IDF de palabras del espacio de MT (8,000 términos) |
| `fecha` | **periodo** (control de fuga) | año, mes, trimestre |
| `estilo_completo` | estilo = agregado + función + caracteres | hstack |
| `todo` | estilo + tema + fecha | hstack |

Los grupos de casi-duplicados se usan para que un discurso repetido no esté a la vez en
train y en test. Se detectan con coseno TF-IDF (proxy rápido); el módulo 9 hará MinHash+LSH.
"""

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import est_scripts.config as config
from est_scripts.style import document_features
from ei_scripts.dedup import UnionFind
from mt_scripts.matrices import build_space
from ri_scripts.corpus import iter_corpus
from ri_scripts.preprocess import fold

# --- Palabras de función (plegadas, sin acentos; la ñ se conserva) -----------
FUNCTION_WORDS = [
    # artículos y determinantes
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel",
    "aquella", "aquellos", "aquellas", "mi", "mis", "tu", "tus", "su", "sus",
    "nuestro", "nuestra", "nuestros", "nuestras", "cada", "todo", "toda", "todos",
    "todas", "otro", "otra", "otros", "otras", "mismo", "misma",
    # preposiciones
    "a", "ante", "bajo", "con", "contra", "de", "desde", "durante", "en", "entre",
    "hacia", "hasta", "mediante", "para", "por", "segun", "sin", "sobre", "tras",
    # conjunciones y relativos
    "y", "e", "o", "u", "pero", "sino", "que", "quien", "quienes", "cual",
    "cuales", "como", "cuando", "donde", "mientras", "porque", "pues", "aunque",
    "si", "ya", "ni", "tambien", "tan", "tanto", "mas", "menos", "muy",
    # pronombres y clíticos
    "yo", "me", "mi", "nos", "nosotros", "nosotras", "tu", "te", "ti", "usted",
    "ustedes", "vosotros", "os", "el", "ella", "ellos", "ellas", "le", "les",
    "se", "se", "conmigo", "contigo",
    # auxiliares, modales y copulativos
    "es", "son", "era", "eran", "fue", "fueron", "ser", "estar", "esta", "estan",
    "estaba", "estamos", "hay", "ha", "han", "haber", "habia", "he", "hemos",
    "tiene", "tienen", "tenia", "tener", "puede", "pueden", "podemos", "debe",
    "deben", "debemos", "va", "van", "vamos", "hace", "hacen", "hacer",
    # adverbios y conectores frecuentes
    "no", "si", "ya", "aun", "asi", "aqui", "ahi", "ahora", "antes", "despues",
    "siempre", "nunca", "solo", "solo", "tambien", "entonces", "ademas", "incluso",
    "sobre", "bien", "mal", "muy", "mucho", "poco",
]


@dataclass
class Dataset:
    """Documentos utilizables, sus bloques de features y los grupos de duplicados."""

    doc_ids: np.ndarray
    speakers: np.ndarray
    dates: np.ndarray
    years: np.ndarray
    style: pd.DataFrame                    # rasgos estilométricos (denso)
    style_matrix: np.ndarray
    blocks: dict                           # nombre -> matriz (densa o CSR)
    groups: np.ndarray
    n_duplicate_groups: int = 0
    largest_group: int = 0
    texts: list = field(default_factory=list)

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)


# --- Rasgos de estilo --------------------------------------------------------
def build_style_frame(records: list) -> pd.DataFrame:
    rows = []
    for record in records:
        feats = document_features(record["text"])
        if not feats:
            continue
        rows.append({
            "doc_id": record["doc_id"], "speaker": record["speaker"],
            "date": record["date"], "year": record["date"][:4], **feats,
        })
    return pd.DataFrame(rows)


def _function_matrix(texts: list, n_words: list) -> np.ndarray:
    """Frecuencia por 1000 palabras de las palabras de función."""
    mat = np.zeros((len(texts), len(FUNCTION_WORDS)), dtype=np.float32)
    word_re = re.compile(r"[a-zñ]+")
    for i, text in enumerate(texts):
        words = word_re.findall(fold(text))
        counts = {}
        for word in words:
            if word in _FUNCTION_SET:
                counts[word] = counts.get(word, 0) + 1
        denom = max(n_words[i], 1)
        for j, word in enumerate(FUNCTION_WORDS):
            mat[i, j] = 1000.0 * counts.get(word, 0) / denom
    return mat


_FUNCTION_SET = set(FUNCTION_WORDS)


_MASK_RE = re.compile("|".join(config.MASK_PATTERNS), re.IGNORECASE)


def mask_identity(text: str) -> str:
    """Sustituye nombres propios y gentilicios por " X " (control de fuga)."""
    return _MASK_RE.sub(" X ", text)


def _char_matrix(texts: list, masked: bool = False) -> sp.csr_matrix:
    """TF-IDF de n-gramas de carácter.

    Con ``masked=True`` se enmascaran los nombres de los hablantes y los países
    de autorreferencia: el clasificador ya no puede usar "milei" o "chile" como
    atajo y los n-gramas miden más forma y menos contenido.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=config.CHAR_NGRAMS,
        min_df=config.CHAR_MIN_DF, max_features=config.CHAR_MAX_FEATURES,
        sublinear_tf=True, lowercase=True, strip_accents="unicode",
    )
    prepared = [text[:config.CHAR_MAX_CHARS] for text in texts]
    if masked:
        prepared = [mask_identity(text) for text in prepared]
    return vectorizer.fit_transform(prepared).astype(np.float32).tocsr()


def _date_matrix(years: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Periodo como features numéricas (control de fuga temporal)."""
    year = years.astype(np.float32)
    month = np.array([int(d[5:7]) for d in dates], dtype=np.float32)
    quarter = np.ceil(month / 3).astype(np.float32)
    return np.column_stack([year, month, quarter])


# --- Duplicados --------------------------------------------------------------
def duplicate_groups(ix, doc_ids: np.ndarray) -> tuple:
    """Grupos de casi-duplicados por coseno TF-IDF + coincidencia exacta.

    Se usa `w_global` (ya normalizada L2) y un vecindario de 8 documentos por
    fila; los pares con coseno ≥ 0.80 se unen. Además se agrupan los textos
    idénticos por hash del texto plegado. Es un proxy rápido: MinHash+LSH del
    módulo 9 sería más barato a escala, pero con 4.5k documentos el vecindario
    exacto es suficiente.
    """
    sub = ix.w_global[doc_ids]
    nn = NearestNeighbors(n_neighbors=min(config.DUP_NEIGHBORS, len(doc_ids)),
                          metric="cosine", algorithm="brute", n_jobs=-1).fit(sub)
    _dist, neigh = nn.kneighbors(sub)

    uf = UnionFind(len(doc_ids))
    for i in range(len(doc_ids)):
        for col, j in enumerate(neigh[i]):
            if i < int(j) and 1.0 - float(_dist[i, col]) >= config.DUP_COSINE:
                uf.union(int(i), int(j))

    # coincidencia exacta del texto plegado (posiciones, no doc_id)
    position = {int(d): k for k, d in enumerate(doc_ids)}
    seen = {}
    for record in iter_corpus(config.CORPUS_JSONL):
        k = position.get(int(record["doc_id"]))
        if k is None:
            continue
        digest = hashlib.blake2b(fold(record["text"]).encode("utf-8"), digest_size=16).hexdigest()
        if digest in seen:
            uf.union(seen[digest], k)
        else:
            seen[digest] = k

    roots = np.array([uf.find(i) for i in range(len(doc_ids))])
    unique, groups = np.unique(roots, return_inverse=True)
    sizes = np.bincount(groups)
    return groups, len(unique), int(sizes.max())


# --- Caché -------------------------------------------------------------------
def _cache_exists() -> bool:
    return all(p.exists() for p in (config.DATASET_NPZ, config.STYLE_CSV,
                                    config.CHARS_NPZ, config.CHARS_MASKED_NPZ,
                                    config.TOPIC_NPZ))


def save_dataset(dataset: Dataset) -> None:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dataset.style.to_csv(config.STYLE_CSV, index=False)
    np.savez_compressed(
        config.DATASET_NPZ,
        doc_ids=dataset.doc_ids, speakers=dataset.speakers, dates=dataset.dates,
        years=dataset.years, style_matrix=dataset.style_matrix,
        func=dataset.blocks["funcion"], fecha=dataset.blocks["fecha"],
        groups=dataset.groups,
        counts=np.array([dataset.n_duplicate_groups, dataset.largest_group]),
    )
    sp.save_npz(config.CHARS_NPZ, dataset.blocks["caracteres"])
    sp.save_npz(config.CHARS_MASKED_NPZ, dataset.blocks["caracteres_enmascarados"])
    sp.save_npz(config.TOPIC_NPZ, dataset.blocks["tema"])


def load_dataset(verbose: bool = True) -> Dataset:
    data = np.load(config.DATASET_NPZ)
    style = pd.read_csv(config.STYLE_CSV)
    chars = sp.load_npz(config.CHARS_NPZ).tocsr()
    chars_masked = sp.load_npz(config.CHARS_MASKED_NPZ).tocsr()
    topic = sp.load_npz(config.TOPIC_NPZ).tocsr()
    style_matrix = data["style_matrix"].astype(np.float32)
    func = data["func"].astype(np.float32)
    dates_mat = data["fecha"].astype(np.float32)

    blocks = {
        "estilo": style_matrix, "funcion": func, "caracteres": chars,
        "caracteres_enmascarados": chars_masked, "tema": topic, "fecha": dates_mat,
        "estilo_controlado": sp.hstack(
            [sp.csr_matrix(style_matrix), sp.csr_matrix(func)], format="csr"
        ).astype(np.float32),
        "todo": sp.hstack(
            [sp.csr_matrix(style_matrix), sp.csr_matrix(func), topic,
             sp.csr_matrix(dates_mat)], format="csr"
        ).astype(np.float32),
    }
    if verbose:
        print(f"dataset cacheado: {len(style)} documentos, {len(blocks)} bloques")
    return Dataset(
        doc_ids=data["doc_ids"], speakers=data["speakers"], dates=data["dates"],
        years=data["years"], style=style, style_matrix=style_matrix,
        blocks=blocks, groups=data["groups"],
        n_duplicate_groups=int(data["counts"][0]), largest_group=int(data["counts"][1]),
    )


# --- Construcción ------------------------------------------------------------
def build_dataset(verbose: bool = True, force: bool = False) -> Dataset:
    """Corpus filtrado por longitud + los cinco bloques + grupos de duplicados."""
    if _cache_exists() and not force:
        return load_dataset(verbose=verbose)

    from ri_scripts.index import load_index  # carga perezosa (2 s)

    ix = load_index()
    records, n_words = [], []
    for record in iter_corpus(config.CORPUS_JSONL):
        n = len(record["text"].split())
        if n < config.MIN_WORDS:
            continue
        records.append(record)
        n_words.append(n)

    doc_ids = np.array([r["doc_id"] for r in records], dtype=np.int64)
    speakers = np.array([r["speaker"] for r in records])
    dates = np.array([r["date"] for r in records])
    years = np.array([d[:4] for d in dates])
    texts = [r["text"] for r in records]

    if verbose:
        print(f"documentos con >= {config.MIN_WORDS} palabras: {len(doc_ids)} de {ix.n_docs}")

    style_df = build_style_frame(records)
    # el frame puede perder documentos sin palabras; alinear por doc_id
    assert len(style_df) == len(records), "algún documento no produjo features"
    feature_cols = [c for c in style_df.columns if c not in ("doc_id", "speaker", "date", "year")]
    style_matrix = style_df[feature_cols].to_numpy(dtype=np.float32)

    func = _function_matrix(texts, n_words)
    # columnas de función que aparecen en menos del X % de los documentos: fuera
    present = (func > 0).mean(axis=0)
    keep = present >= config.FUNCTION_MIN_DF
    func = func[:, keep]

    chars = _char_matrix(texts)
    chars_masked = _char_matrix(texts, masked=True)
    space = build_space(ix)
    position = {int(d): i for i, d in enumerate(space.doc_ids)}
    topic = space.W_l2[[position[int(d)] for d in doc_ids]].tocsr()

    dates_mat = _date_matrix(years, dates)

    blocks = {
        "estilo": style_matrix,
        "funcion": func,
        "caracteres": chars,
        "caracteres_enmascarados": chars_masked,
        "tema": topic,
        "fecha": dates_mat,
        # estilo "controlado": sólo rasgos de forma (sin n-gramas ni tema)
        "estilo_controlado": sp.hstack(
            [sp.csr_matrix(style_matrix), sp.csr_matrix(func)], format="csr"
        ).astype(np.float32),
        # cota superior: forma + contenido + periodo
        "todo": sp.hstack(
            [sp.csr_matrix(style_matrix), sp.csr_matrix(func), topic,
             sp.csr_matrix(dates_mat)], format="csr"
        ).astype(np.float32),
    }

    groups, n_groups, largest = duplicate_groups(ix, doc_ids)

    if verbose:
        print(f"grupos de duplicados: {n_groups} (mayor: {largest} documentos)")
        print(f"estilo: {style_matrix.shape[1]} rasgos | función: {func.shape[1]} "
              f"| caracteres: {chars.shape[1]} | tema: {topic.shape[1]}")

    dataset = Dataset(
        doc_ids=doc_ids, speakers=speakers, dates=dates, years=years,
        style=style_df, style_matrix=style_matrix, blocks=blocks,
        groups=groups, n_duplicate_groups=n_groups, largest_group=largest,
        texts=texts,
    )
    save_dataset(dataset)
    if verbose:
        print(f"dataset cacheado en {config.ARTIFACTS}")
    return dataset


def window_mask(dataset: Dataset) -> np.ndarray:
    """Documentos de la ventana limpia 2023-2024, sólo hablantes con ≥ 30 discursos."""
    in_window = np.isin(dataset.years, list(config.WINDOW))
    counts = pd.Series(dataset.speakers[in_window]).value_counts()
    speakers = set(counts[counts >= config.WINDOW_MIN_DOCS].index)
    return in_window & np.isin(dataset.speakers, list(speakers))
