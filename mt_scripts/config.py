"""Configuración, rutas y parámetros del módulo de Minería de Textos (MT).

Unidad del temario: 5.1 (introducción), 5.2 (técnicas básicas: frecuencias,
co-ocurrencia, PMI), 5.3 (algoritmos y modelos: LSA, NMF, LDA), 5.4-5.5
(herramientas y software), 5.6 (más allá: dinámica temporal de tópicos).

Todo número ajustable vive aquí para que el reporte pueda citarlo. Los valores
por defecto están calibrados para un corpus de 4,531 documentos y ~30k términos
de vocabulario global (el de RI, `min_df = 2`).
"""

from pathlib import Path

from ri_scripts import config as ri_config

# --- Rutas -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
ARTIFACTS = PKG_DIR / "artifacts"
GOLD_DIR = PKG_DIR / "gold"

# Artefactos compartidos con RI: no se reindexa nada.
CORPUS_JSONL = ri_config.CORPUS_JSONL
CORPUS_OFFSETS = ri_config.CORPUS_OFFSETS
VOCABULARY_JSON = ri_config.VOCABULARY_JSON
IDF_NPY = ri_config.IDF_NPY
POSTINGS_NPZ = ri_config.POSTINGS_NPZ
W_GLOBAL_NPZ = ri_config.W_GLOBAL_NPZ
INDEX_META_JSON = ri_config.INDEX_META_JSON

# Artefactos propios del módulo.
TOKENS_NPZ = ARTIFACTS / "tokens.npz"            # streams de stems (ids globales)
STEM_SURFACE_JSON = ARTIFACTS / "stem_surface.json"
COOC_NPZ = ARTIFACTS / "cooc_npz.npz"             # co-ocurrencia PMI (ventana)
TOPICS_JSON = ARTIFACTS / "topics_global.json"
DOC_TOPICS_JSONL = ARTIFACTS / "doc_topics.jsonl"
TIMELINE_JSON = ARTIFACTS / "timeline.json"
SPEAKER_TOPICS_JSON = ARTIFACTS / "topics_by_speaker.json"
WINDOW_TOPICS_JSON = ARTIFACTS / "topics_by_window.json"
EVAL_JSON = ARTIFACTS / "eval.json"
SWEEP_JSON = ARTIFACTS / "sweep.json"
TABLES_MD = ARTIFACTS / "tables.md"

GOLD_TOPICS_JSON = GOLD_DIR / "topics_gold.json"

# --- Vocabulario de trabajo para MT -----------------------------------------
# El vocabulario de RI (min_df=2, ~30k términos) es demasiado ruidoso: LDA
# densifica y los tópicos se llenan de hápax y de números. Se poda:
MIN_DF = 10              # el término debe aparecer en >= 10 documentos
MAX_DF_RATIO = 0.50      # y en <= 50 % (por encima es plantilla, no tema)
MIN_TERM_LEN = 3         # stems de 2 letras no son temas
MAX_TERMS = 8000         # tope por masa df*idf
DROP_DIGITS = True       # "2024", "30", ... no son tópicos

# Términos de la **puesta en escena** del discurso, no de su contenido: cargos,
# gentilicios de autorreferencia, nombres de los propios hablantes, dícticos y
# ruido de transcripción. Sin esta lista, los tópicos se organizan por quién
# habla y no por lo que se dice (el corpus confunde país = persona, restricción
# 1 del plan base): el tópico dominante de Piñera queda "chilenos · Chile ·
# compatriotas" en vez de su tema. Se documenta como decisión, no se esconde.
BOILERPLATE_TERMS = {
    # gentilicios y países de autorreferencia
    "chil", "chilen", "argentin", "españ", "español", "mexic", "mexican",
    "venezuel", "venezolan",
    # nombres de los cinco hablantes
    "obrador", "lopez", "madur", "sanchez", "piñer", "milei",
    # cargos y etiquetas de la transcripción
    "president", "señori", "señor", "interlocutor", "moder", "aplaus",
    "grabacion", "version", "estenograf", "transcripcion",
    # dícticos y comodines
    "ayer", "hoy", "ahor", "aqui", "ahi", "moment", "tem", "asunt", "cos",
}
DROP_BOILERPLATE = True

# Mismas reglas, más laxas, para subconjuntos (hablante / año / ventana).
SUB_MIN_DF_ABS = 2
SUB_MIN_DF_RATIO = 0.02  # df >= max(2, ceil(0.02 * n_docs))
SUB_MAX_DF_RATIO = 0.60
SUB_MAX_TERMS = 4000
SUB_DROP_BOILERPLATE = True

# --- Modelos -----------------------------------------------------------------
K_GRID = (6, 8, 10, 12, 15)   # barrido de nº de tópicos
SEED = 42
# Regla de selección de k: banda de indiferencia sobre la coherencia. Se elige el
# menor k cuya c_v esté a menos de esta tolerancia de la máxima. Se fija en 0.05
# porque es del orden del error estándar de la c_v entre tópicos (≈ 0.048 en el
# barrido): diferencias menores no son distinguibles con 4,531 documentos.
SELECTION_TOLERANCE = 0.05
MIN_DIVERSITY = 0.60

NMF_INIT = "nndsvda"
NMF_MAX_ITER = 500
NMF_TOL = 1e-5

LDA_MAX_ITER = 12
LDA_LEARNING_METHOD = "online"   # batch tarda ~6 s/iteración; online ~1,5 s con n_jobs=-1
LDA_BATCH_SIZE = 2048
LDA_JOBS = -1                     # paraleliza las actualizaciones por lote
LDA_DOC_TOPIC_PRIOR = None     # None = 1/k (asimetría fija de sklearn)
LDA_TOPIC_WORD_PRIOR = None

LSA_COMPONENTS = 50            # SVD truncada, independiente del k de tópicos

# --- Etiquetas e interpretación ---------------------------------------------
TOP_TERMS = 10           # términos que se muestran por tópico
TOP_TERMS_DIVERSITY = 10  # ventana para diversidad y exclusividad
TOP_TERMS_WINDOW = 30    # vocabulario del tópico para anclas y alineación
N_EVIDENCE_DOCS = 5      # documentos representativos que se guardan por tópico
SNIPPET_CHARS = 320      # longitud del extracto de evidencia

# --- Coherencia (implementada a mano, sin gensim) ---------------------------
COOC_WINDOW = 10         # ventana deslizante de tokens (c_v)
COOC_TOP_TERMS = 10      # top-n que entra en c_v / UMass

# --- Estabilidad -------------------------------------------------------------
STABILITY_RUNS = 5
STABILITY_SUBSAMPLE = 0.80
STABILITY_RUNS_LDA = 3   # LDA es más caro: menos remuestreos si gana el barrido
STABILITY_LDA_ITERS = 10  # y con menos iteraciones, para que el costo sea acotado

# --- Modelos por hablante ----------------------------------------------------
SPEAKER_K = 8
SPEAKER_MIN_DOCS = 30    # Maduro tiene exactamente 30: entra con nota de baja confianza
SPEAKER_LOW_CONFIDENCE = 60

# --- Modelos por ventana temporal (dinámica de tópicos) ----------------------
WINDOW_MIN_DOCS = 30     # años por debajo no se ajustan
WINDOW_K = 8
# Alineación entre ventanas: se comparan los **conjuntos** de los 50 términos de
# mayor peso (coseno de vectores binarios = solapamiento / sqrt(50·50)). El
# coseno de los pesos continuos es diminuto (0,01-0,05) porque las colas no se
# solapan y no distingue tópicos emparejados de tópicos ajenos. Con binarios, un
# mismo tema entre dos años comparte 20-29 de 50 términos (sim 0,4-0,58) y uno
# distinto comparte < 10 (sim < 0,2).
ALIGN_TERMS = 50
WINDOW_MIN_SIM = 0.25    # solapamiento de al menos ~12,5 de 50 términos

# --- Comparación entre líderes ----------------------------------------------
# El plan base fija 2023-2024 como única ventana limpia con 4 de los 5 líderes.
CLEAN_WINDOW = ("2023", "2024")
