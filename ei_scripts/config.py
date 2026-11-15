"""Configuración, rutas y pesos del módulo de Extracción de Información (EI).

Todo lo que es "número ajustable" vive aquí para que el reporte pueda citarlo.
"""

from pathlib import Path

from ri_scripts import config as ri_config

# --- Rutas -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
ARTIFACTS = PKG_DIR / "artifacts"

# Corpus y offsets se comparten con RI: el `doc_id` es exactamente el mismo,
# de modo que un evento se puede cruzar con el índice invertido sin remapear.
CORPUS_JSONL = ri_config.CORPUS_JSONL
CORPUS_OFFSETS = ri_config.CORPUS_OFFSETS

EVENTS_JSONL = ARTIFACTS / "events.jsonl"        # un evento por línea
EVENT_DOCS_JSONL = ARTIFACTS / "events_by_doc.jsonl"  # eventos agrupados por doc
STATS_JSON = ARTIFACTS / "stats.json"
EVAL_JSON = ARTIFACTS / "eval.json"
GOLD_CSV = PKG_DIR / "gold" / "gold_seed.csv"
GOLD_CANDIDATES_CSV = PKG_DIR / "gold" / "gold_candidates.csv"

# --- Segmentación ------------------------------------------------------------
# Una "oración" de más de MAX_SENT_TOKENS tokens se vuelve a partir por ; : y ,
# porque las transcripciones (sobre todo las mañaneras de AMLO) llegan sin
# puntuación fiable y producirían una sola oración gigante.
MAX_SENT_TOKENS = 90

# --- Disparadores ------------------------------------------------------------
# Fuerza de cada clase de disparador. Se usa en la confianza del evento.
STRENGTH = {
    "commit": 1.00,     # me comprometo, prometo, doy mi palabra
    "future_1p": 0.90,  # voy a / vamos a + verbo de acción
    "future_1p_syn": 0.90,  # construiré, entregaremos
    "future_3p_syn": 0.85,  # el gobierno entregará (exige sujeto atribuible)
    "announce": 0.80,   # anuncio que, doy a conocer
    "instruct": 0.70,   # he girado instrucciones
    "goal": 0.45,       # antes de que termine, en los próximos 100 días
    "deontic": 0.25,    # hay que, tenemos que
}

# Distancia máxima (en palabras) entre el disparador y el primer verbo de acción.
# Las perífrasis verbales son locales: "vamos a construir". Si el verbo aparece
# muy lejos, el disparador es otra cosa ("mañana vamos a Chiapas y ... informarles").
MAX_ACTION_DISTANCE = {
    "commit": 4,
    "future_1p": 4,
    "future_1p_syn": 0,   # el propio verbo es el disparador
    "future_3p_syn": 0,
    "announce": 6,
    "instruct": 8,
    "deontic": 2,
    "goal": 2,
}

# --- Pesos del score de confianza (suman 1.0 antes de penalizaciones) --------
W_TRIGGER = 0.40
W_ACTION = 0.20      # hay verbo de acción en la misma oración
W_SLOT = 0.15        # hay al menos un slot "duro" (fecha/monto/cantidad)
W_CYK = 0.15         # la oración parsea con la CFG
W_LUGAR = 0.10       # hay lugar explícito (ancla territorial de la promesa)

NEGATION_FACTOR = 0.15   # multiplicador si el disparador está negado
NEGATION_WINDOW = 6      # tokens hacia atrás que mira el autómata de negación

# Umbrales de confianza para publicar un evento
MIN_CONFIDENCE = 0.55
MIN_CONFIDENCE_STRICT = 0.75

# --- Deduplicación (MinHash + LSH) -------------------------------------------
SHINGLE_SIZE = 3
MINHASH_PERM = 96        # 12 bandas x 8 filas
LSH_BANDS = 12
LSH_ROWS = 8
DEDUP_JACCARD = 0.80
DEDUP_MIN_TOKENS = 8     # no deduplicar oraciones cortas
LSH_MAX_BUCKET = 64      # ignorar cubetas gigantes (ruido de discursos repetidos)

# --- Evaluación --------------------------------------------------------------
EVAL_KAPPA_BOOTSTRAP = 2000
RANDOM_SEED = 20240

# --- Tipos de evento ---------------------------------------------------------
EVENT_TYPES = (
    "infraestructura",
    "salud",
    "educacion",
    "social",
    "economia_empleo",
    "seguridad",
    "politica_institucional",
    "otro",
)
