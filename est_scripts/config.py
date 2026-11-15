"""Configuración del módulo 7-lite: estilometría y clasificación de hablante.

Alcance reducido y explícito:

- **perfil estilométrico** por líder (rasgos observables del texto);
- **clasificación de hablante** con validación cruzada agrupada (macro-F1);
- **ablación estilo / tema / fecha** para medir el confundido país = persona.

No se infiere personalidad, ideología ni estado emocional: un discurso permite medir
*forma de hablar*, no *quién es* la persona. El documento lo dice en la primera sección.
"""

from pathlib import Path

from ri_scripts import config as ri_config

# --- Rutas -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
ARTIFACTS = PKG_DIR / "artifacts"

CORPUS_JSONL = ri_config.CORPUS_JSONL
CORPUS_OFFSETS = ri_config.CORPUS_OFFSETS
PARQUET = ri_config.PARQUET

STYLE_PROFILES_CSV = ARTIFACTS / "style_profiles.csv"
DATASET_NPZ = ARTIFACTS / "dataset_npz.npz"     # matrices pequeñas + metadatos
STYLE_CSV = ARTIFACTS / "style_matrix.csv"      # rasgos por documento (cache)
CHARS_NPZ = ARTIFACTS / "chars_npz.npz"         # TF-IDF de caracteres (cache)
CHARS_MASKED_NPZ = ARTIFACTS / "chars_masked_npz.npz"  # caracteres con nombres enmascarados
TOPIC_NPZ = ARTIFACTS / "topic_npz.npz"         # TF-IDF de tema (cache)
EVAL_JSON = ARTIFACTS / "eval.json"
TABLES_MD = ARTIFACTS / "tables.md"
OOF_JSONL = ARTIFACTS / "oof_predictions.jsonl"
FEATURES_NPZ = ARTIFACTS / "features_npz.npz"

# --- Corpus ------------------------------------------------------------------
MIN_WORDS = 120          # discursos más cortos no entran (ruido de transcripción)
SEED = 42

# --- Rasgos estilométricos ---------------------------------------------------
# Los rasgos agregados se calculan sobre el texto crudo (no sobre stems): la
# estilometría mide forma, no contenido.
MTLD_THRESHOLD = 0.72     # umbral estándar de MTLD (0.72)
LONG_WORD_CHARS = 7       # "palabra larga" para el ratio de cultismo
SHORT_SENT_WORDS = 8      # oración corta
LONG_SENT_WORDS = 35      # oración larga

# --- N-gramas de caracteres (estilo) ----------------------------------------
CHAR_MAX_CHARS = 12000    # se recorta el documento: el estilo se estima con 2k palabras
CHAR_NGRAMS = (3, 4)
CHAR_MIN_DF = 20
CHAR_MAX_FEATURES = 25000

# --- Enmascarado de nombres (control de fuga en n-gramas de carácter) --------
# Los n-gramas de carácter capturan "milei", "obrador", "ucrania", "tren maya".
# Para separar estilo de contenido se sustituyen los nombres de los hablantes y
# los gentilicios/países de autorreferencia por " X " antes de vectorizar.
MASK_PATTERNS = [
    r"andres manuel lopez obrador", r"lopez obrador", r"obrador", r"amlo",
    r"javier milei", r"milei",
    r"nicolas maduro", r"maduro",
    r"pedro sanchez", r"sanchez",
    r"sebastian piñera", r"piñera", r"piñera echenique",
    r"argentina", r"argentino", r"argentinos", r"argentina",
    r"chile", r"chileno", r"chilenos", r"chilena", r"chilenas",
    r"españa", r"español", r"española", r"españoles", r"españolas",
    r"mexico", r"mexicano", r"mexicanos", r"mexicana", r"mexicanas",
    r"venezuela", r"venezolano", r"venezolanos", r"venezolana", r"venezolanas",
]

# --- Bloque de función gramatical -------------------------------------------
FUNCTION_MIN_DF = 0.20    # palabras de función presentes en >= 20 % de los discursos

# --- Validación --------------------------------------------------------------
N_SPLITS = 5
DUP_COSINE = 0.80         # casi-duplicado si el coseno TF-IDF >= 0.80 (proxy; el módulo 9 hará MinHash+LSH)
DUP_NEIGHBORS = 8

# --- Ventana limpia ----------------------------------------------------------
WINDOW = ("2023", "2024")
WINDOW_MIN_DOCS = 30      # hablante con menos discursos en la ventana no entra en el control
