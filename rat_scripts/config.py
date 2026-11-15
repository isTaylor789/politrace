"""Configuración, rutas y pesos del módulo de Resumen Automático (RAT).

Unidad del temario: 4.1 a 4.5 (resumen de un documento), 4.6 (multi-documento).
La configuración del resumen anual (módulo 5) está al final del archivo.

Todo lo que es "número ajustable" vive aquí para que el reporte pueda citarlo.
Los valores están calibrados para producir resúmenes *legibles* de ~15-20 % del
original: no se busca la máxima compresión, sino un resumen general legible.
"""

from pathlib import Path

from ri_scripts import config as ri_config

# --- Rutas -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
ARTIFACTS = PKG_DIR / "artifacts"

# Corpus e índice se comparten con RI: mismo `doc_id`, mismo IDF global y misma
# matriz TF-IDF (`w_global`), de modo que el resumidor no reindexa nada.
CORPUS_JSONL = ri_config.CORPUS_JSONL
CORPUS_OFFSETS = ri_config.CORPUS_OFFSETS
VOCABULARY_JSON = ri_config.VOCABULARY_JSON
IDF_NPY = ri_config.IDF_NPY

GOLD_JSON = PKG_DIR / "gold" / "summaries_gold.json"
EVAL_JSON = ARTIFACTS / "eval.json"
DEMO_JSONL = ARTIFACTS / "summaries_demo.jsonl"
TABLES_MD = ARTIFACTS / "tables.md"

# --- Segmentación y filtrado de unidades (4.3.2) -----------------------------
# Una oración del discurso es candidata si tiene al menos MIN_SENT_WORDS palabras
# y no supera MAX_SENT_WORDS (las transcripciones de AMLO traen tiradas largas:
# una "oración" de 90 palabras no se puede leer dentro de un resumen).
MIN_SENT_WORDS = 7
MAX_SENT_WORDS = 75
MIN_UNITS = 8          # por debajo de esto el texto es demasiado corto para resumir

# --- Selección (4.3.3) -------------------------------------------------------
SUMMARY_RATIO = 0.15   # fracción de oraciones objetivo
MIN_SENTENCES = 3
MAX_SENTENCES = 12
WORD_BUDGET = 180      # tope duro de palabras del resumen
MMR_LAMBDA = 0.70      # 0.7 relevancia - 0.3 redundancia
DUP_JACCARD = 0.80     # casi-duplicados: se descartan aunque MMR los puntúe alto

# --- Grafos (TextRank / LexRank) --------------------------------------------
PAGERANK_DAMPING = 0.85
PAGERANK_TOL = 1e-8
PAGERANK_MAX_ITER = 300
LEXRANK_THRESHOLD = 0.10   # similitud mínima para que exista arista en LexRank
TEXTRANK_WINDOW = 0        # 0 = grafo completo (sin ventana de proximidad)

# --- LSA ---------------------------------------------------------------------
LSA_K = 100

# --- Pesos del híbrido (suman 1.0) ------------------------------------------
WEIGHTS = {
    "centroid": 0.28,     # centralidad léxica respecto al discurso completo
    "textrank": 0.24,     # centralidad en el grafo de similitud
    "discourse": 0.26,    # posición + pistas discursivas + título
    "lsa": 0.10,          # tópico latente dominante (aporta poco, pero aporta)
    "luhn": 0.12,         # palabras significativas (banda de frecuencia)
}

# Reparto interno del bloque "discourse"
DISCOURSE_WEIGHTS = {
    "position": 0.50,
    "cue": 0.35,
    "title": 0.15,
}

# --- Evaluación (4.5) --------------------------------------------------------
COVERAGE_TOP_TERMS = 20   # conceptos del documento que debería cubrir el resumen

# --- Resumen multi-documento por año (módulo 5, temario 4.6) ----------------
YEAR_MIN_DOCS = 5          # por debajo de esto no se resume el año
CLUSTER_MIN = 3            # nº mínimo de grupos temáticos
CLUSTER_MAX = 8            # nº máximo
CLUSTER_SEED = 42
GENERIC_DF_RATIO = 0.40    # término "genérico" si aparece en ≥40 % de los docs del año
DOCS_PER_GROUP = 3         # documentos (medoides) que aportan oraciones por grupo
PER_DOC_SENTENCES = 2      # oraciones que aporta cada documento
MULTI_MAX_SENTENCES = 14   # tope de oraciones del resumen anual
MULTI_WORD_BUDGET = 400    # presupuesto de palabras del resumen anual
MULTI_IDF_MODE = "global"  # "global" | "blend" | "local" (global gana en el nivel anual)
IDF_BLEND_ALPHA = 0.5      # peso del IDF global en el modo "blend"
MULTI_YEAR_WEIGHT = 0.0    # afinidad con el centroide del año (probado: no ayuda por defecto)
MULTI_MIN_SHARE = 0.05     # grupos con ≥5 % de los docs tienen 1 oración garantizada
MULTI_MIN_RAW = 0.50       # grupos con <0.5 oraciones esperadas no ocupan cupo fijo
