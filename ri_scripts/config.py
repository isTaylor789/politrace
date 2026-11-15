"""Configuración y rutas del módulo de Recuperación de Información (RI)."""

from pathlib import Path

# --- Rutas -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
ARTIFACTS = PKG_DIR / "artifacts"

PARQUET = PROJECT_ROOT / "unified" / "presidentes_latam_es.parquet"

CORPUS_JSONL = ARTIFACTS / "corpus.jsonl"
CORPUS_OFFSETS = ARTIFACTS / "corpus_offsets.npy"
VOCABULARY_JSON = ARTIFACTS / "vocabulary.json"
W_GLOBAL_NPZ = ARTIFACTS / "w_global.npz"          # matriz TF-IDF (doc x term), filas L2
IDF_NPY = ARTIFACTS / "idf.npy"
POSTINGS_NPZ = ARTIFACTS / "postings.npz"          # índice invertido término -> docs
BM25_STATS_NPZ = ARTIFACTS / "bm25_stats.npz"      # df, dl, avgdl
INDEX_META_JSON = ARTIFACTS / "index_meta.json"    # doc_ids, hablantes, fechas, parámetros

# --- Preproceso --------------------------------------------------------------
# Términos que aportan frecuencia pero no contenido en discursos. Se pliegan
# igual que el texto, así que van sin acentos.
EXTRA_STOPWORDS = {
    "si", "mas", "entonces", "asi", "aqui", "solo", "aun",
    "eh", "mmm", "ah", "oh", "uh", "aja",
}

MIN_DF = 2          # se descartan términos que aparecen en un solo documento

# --- BM25 --------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
