"""Preproceso de texto en español: normalización, stopwords y stemming.

Un único analizador (`analyze`) se usa tanto al indexar como al consultar,
para que los términos de la consulta y del índice coincidan exactamente.
"""

import re

from nltk.stem import SnowballStemmer

from ri_scripts.config import EXTRA_STOPWORDS

# --- Normalización -----------------------------------------------------------
# Se pliegan las vocales acentuadas (para que "México"/"mexico" coincidan) pero
# se conserva la eñe, que sí distingue palabras en español.
_FOLD = str.maketrans(
    "áàäâãéèëêíìïîóòöôõúùüûçÁÀÄÂÃÉÈËÊÍÌÏÎÓÒÖÔÕÚÙÜÛÇ",
    "aaaaaeeeeiiiiooooouuuucAAAAAEEEEIIIIOOOOOUUUUC",
)

_TOKEN_RE = re.compile(r"[a-zñ0-9]+")

# "3,000" / "3.000" -> "3000": evita fragmentos numéricos basura en el vocabulario
_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))")

# Fallback mínimo por si no está disponible el corpus de stopwords de NLTK.
_FALLBACK_STOPWORDS = """
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellos
asi aun bajo cada como con contra cual cuales cuando de del desde donde dos el
ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estado
estan estar estas este esto estos estoy fue fueron ha habia han hasta hay la
las le les lo los mas me mi mientras mismo mucho muy nada ni no nos nosotros o
os otra otras otro otros para pero poco por porque que quien se sea segun ser
si sin sobre solo son su sus tambien tan tanto te tiene tienen todo todos tu
tus un una uno unos y ya yo
"""

# --- Stopwords y stemmer -----------------------------------------------------
try:
    from nltk.corpus import stopwords as _nltk_stopwords

    _BASE_STOPWORDS = set(_nltk_stopwords.words("spanish"))
except Exception:  # pragma: no cover - depende de nltk_data local
    _BASE_STOPWORDS = set(_FALLBACK_STOPWORDS.split())

# La lista de NLTK viene acentuada ("más", "sí"...), así que hay que plegarla
# para que coincida con los tokens ya normalizados.
STOPWORDS = {w.lower().translate(_FOLD) for w in _BASE_STOPWORDS}
STOPWORDS |= {w.lower().translate(_FOLD) for w in EXTRA_STOPWORDS}

_STEMMER = SnowballStemmer("spanish")
_STEM_CACHE: dict[str, str] = {}


def fold(text: str) -> str:
    """Minúsculas + plegado de acentos (la ñ se conserva).

    Conserva la longitud del texto, así que sirve para mapear posiciones en los
    fragmentos de resultados.
    """
    return text.lower().translate(_FOLD)


def _prepare(text: str) -> str:
    return _THOUSANDS.sub("", fold(text))


def stem(token: str) -> str:
    """Stemming Snowball con caché (el corpus repite mucho los mismos tipos)."""
    cached = _STEM_CACHE.get(token)
    if cached is not None:
        return cached
    result = token if token.isdigit() else _STEMMER.stem(token)
    _STEM_CACHE[token] = result
    return result


def analyze(text: str) -> list[str]:
    """Texto -> lista de términos normalizados (stopwords fuera, con stemming)."""
    tokens = []
    for tok in _TOKEN_RE.findall(_prepare(text)):
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        tokens.append(tok if tok.isdigit() else stem(tok))
    return tokens


def analyze_phrase(text: str) -> list[str]:
    """Secuencia de términos para frases exactas: conserva stopwords."""
    return [tok if tok.isdigit() else stem(tok) for tok in _TOKEN_RE.findall(_prepare(text))]


def normalize_query(query: str) -> list[str]:
    """Alias explícito para el analizador, usado por los buscadores."""
    return analyze(query)


def surface_terms(text: str) -> list[str]:
    """Formas superficiales (plegadas, sin stemming) para resaltar fragmentos."""
    return [tok for tok in _TOKEN_RE.findall(fold(text)) if len(tok) >= 3 and tok not in STOPWORDS]
