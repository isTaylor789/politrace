"""Estilometría por discurso: rasgos de forma, no de contenido.

Todo se calcula sobre el texto crudo (plegado sólo para contar palabras) y se
normaliza por 1,000 palabras cuando es una frecuencia, para que discursos de 400 y
de 20,000 palabras sean comparables.

Tres bloques, siguiendo la tradición de estilometría (Mosteller & Wallace,
"Federalist Papers", 1964; Burrows, 2002):

1. **riqueza y longitud** (TTR, MTLD, Yule's K, hápax, longitud de palabra y de
   oración, legibilidad): cuánto y cómo se despliega el léxico.
2. **puntuación y tipografía** (comas, puntos, interrogación, comillas, mayúsculas):
   el "ritmo" escrito/oral, que es muy difícil de imitar.
3. **deixis y marcadores** (yo/nosotros/tú/él, muletillas orales, vocativos,
   atenuadores, citas): posición del hablante respecto a su auditorio.

Fórmulas:

- TTR = tipos / tokens.
- MTLD (McCarthy, 2005): se recorre el texto acumulando TTR; cada vez que cae por
  debajo de ``t=0.72`` se cierra un "factor". Se hace en las dos direcciones y se
  promedian. Es robusto al tamaño, a diferencia de TTR.
- Yule's K (Yule, 1944): ``K = 10⁴ (Σ i²·V_i − N) / N²``, con ``V_i`` = nº de tipos
  con frecuencia ``i``; mide repetición (más alto = más repetitivo).
- Fernández-Huerta: ``206.84 − 0.60·P − 1.02·F`` con ``P`` = sílabas/100 palabras
  y ``F`` = oraciones/100 palabras. Szigriszt-Pazos: ``206.835 − 62.3·(S/Pal) −
  Pal/Orac``. Las sílabas se aproximan contando grupos vocálicos (documentado como
  aproximación: no hay silabificador léxico en el proyecto).
"""

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

import est_scripts.config as config
from ri_scripts.preprocess import fold

_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_UPPER_RE = re.compile(r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]{2,}\b")
_VOWELS_RE = re.compile(r"[aeiou]+")

# Listas plegadas (minúsculas, sin acentos; la ñ se conserva).
FIRST_SING = {"yo", "me", "mi", "mis", "mio", "mia", "mios", "mias", "conmigo"}
FIRST_PLUR = {"nosotros", "nosotras", "nos", "nuestro", "nuestra", "nuestros", "nuestras"}
SECOND = {"tu", "tus", "ti", "te", "usted", "ustedes", "vosotros", "vosotras", "os",
          "vuestro", "vuestra", "vuestros", "vuestras", "contigo"}
THIRD = {"el", "ella", "ellos", "ellas", "le", "les", "se", "su", "sus", "suyo",
         "suya", "consigo"}

ORALITY = {"bueno", "entonces", "pues", "miren", "digamos", "oye", "eh", "mmm",
           "claro", "evidentemente", "lamentablemente", "afortunadamente",
           "sinceramente", "francamente", "aqui", "ahi"}
VOCATIVES = {"amigas", "amigos", "compañeras", "compañeros", "señorias", "señores",
             "señoras", "compatriotas", "queridos", "queridas", "hermanos",
             "hermanas", "ciudadanos", "ciudadanas"}
HEDGES = {"creo", "pienso", "considero", "parece", "quizas", "quiza", "puede",
          "podria", "deberia", "probablemente", "posiblemente", "seguramente",
          "supuestamente"}

# Frases (se buscan sobre el texto plegado, con espacios)
PHRASES = {
    "o sea": "o sea",
    "por supuesto": "por supuesto",
    "sin embargo": "sin embargo",
    "no obstante": "no obstante",
    "en definitiva": "en definitiva",
    "tal vez": "tal vez",
    "es decir": "es decir",
    "por ciento": "por ciento",
}

PUNCT_KEYS = {
    "comas": ",", "puntos": ".", "punto_coma": ";", "dos_puntos": ":",
    "interrogacion": "?", "exclamacion": "!", "comillas": '"', "parentesis": "(",
    "guiones": "-", "puntos_suspensivos": "…",
}


def _syllables(word: str) -> int:
    """Aproximación silábica: nº de grupos vocálicos (mínimo 1)."""
    return max(1, len(_VOWELS_RE.findall(word)))


def _mtld(words: list, threshold: float = config.MTLD_THRESHOLD) -> float:
    """MTLD bidireccional."""

    def forward(seq: list) -> float:
        factors, types, count = 0, set(), 0
        for word in seq:
            count += 1
            if word not in types:
                types.add(word)
                if len(types) / count <= threshold:
                    factors += 1
                    types, count = set(), 0
        if count > 0:
            ttr = len(types) / count
            factors += (1.0 - ttr) / (1.0 - threshold) if ttr < 1.0 else 0.0
        return len(seq) / factors if factors > 0 else float(len(seq))

    if len(words) < 10:
        return float(len(words))
    return (forward(words) + forward(list(reversed(words)))) / 2.0


def _yules_k(words: list) -> float:
    """Yule's K: repetición léxica."""
    if not words:
        return 0.0
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    n = len(words)
    s = sum(freq * freq for freq in counts.values())
    return 1e4 * (s - n) / (n * n)


def _per_1000(value: float, n_words: int) -> float:
    return 1000.0 * value / max(n_words, 1)


def document_features(text: str) -> dict:
    """Vector de rasgos estilométricos de un discurso (≈50 dimensiones)."""
    folded = fold(text)
    words = _WORD_RE.findall(folded)
    n_words = len(words)
    if n_words == 0:
        return {}

    # --- oraciones: spans reales de punkt (sin recortes de segmentación) ------
    from ei_scripts.segment import _punkt

    spans = list(_punkt().span_tokenize(folded))
    sent_lengths = [len(_WORD_RE.findall(folded[s:e])) for s, e in spans]
    sent_lengths = [n for n in sent_lengths if n > 0] or [n_words]
    n_sents = len(sent_lengths)

    counts = Counter(words)
    types = counts.keys()
    hapax = sum(1 for freq in counts.values() if freq == 1)
    syllables = sum(_syllables(w) for w in words)
    long_words = sum(1 for w in words if len(w) >= config.LONG_WORD_CHARS)
    numbers = sum(1 for token in _NUM_RE.findall(folded))

    # mayúsculas "no iniciales": nombres propios y siglas (proxy)
    upper = 0
    for s, e in spans:
        body = text[s:e]
        upper += len(_UPPER_RE.findall(body)) - (1 if _UPPER_RE.match(body.strip()) else 0)
    upper = max(upper, 0)

    feats = {
        # riqueza y longitud
        "n_words": float(n_words),
        "log_words": float(np.log1p(n_words)),
        "n_sents": float(n_sents),
        "words_per_sent": n_words / n_sents,
        "sd_words_per_sent": float(np.std(sent_lengths)),
        "short_sent_ratio": sum(1 for n in sent_lengths if n <= config.SHORT_SENT_WORDS) / n_sents,
        "long_sent_ratio": sum(1 for n in sent_lengths if n >= config.LONG_SENT_WORDS) / n_sents,
        "ttr": len(types) / n_words,
        "mtld": _mtld(words),
        "yules_k": _yules_k(words),
        "hapax_ratio": hapax / max(len(types), 1),
        "mean_word_len": float(np.mean([len(w) for w in words])),
        "long_word_ratio": long_words / n_words,
        "syllables_per_word": syllables / n_words,
        # legibilidad (aproximada)
        "fernandez_huerta": 206.84 - 0.60 * (syllables / n_words * 100)
                           - 1.02 * (n_words / n_sents),
        "szigriszt": 206.835 - 62.3 * (syllables / n_words) - (n_words / n_sents),
        # puntuación y tipografía (por 1000 palabras)
        "uppercase_ratio": upper / n_words,
        "digit_ratio": numbers / n_words,
    }
    for name, char in PUNCT_KEYS.items():
        feats[name] = _per_1000(text.count(char), n_words)
    feats["punto_coma_elevado"] = _per_1000(text.count(";") + text.count(":"), n_words)
    feats["comillas_angulares"] = _per_1000(
        text.count("«") + text.count("»") + text.count("“") + text.count("”"), n_words)
    feats["puntos_suspensivos"] = _per_1000(text.count("…") + text.count("..."), n_words)
    feats["saltos_linea"] = _per_1000(text.count("\n"), n_words)

    # --- deixis y marcadores -------------------------------------------------
    first_sing = sum(1 for w in words if w in FIRST_SING)
    first_plur = sum(1 for w in words if w in FIRST_PLUR)
    second = sum(1 for w in words if w in SECOND)
    third = sum(1 for w in words if w in THIRD)
    feats["first_sing_1000"] = _per_1000(first_sing, n_words)
    feats["first_plur_1000"] = _per_1000(first_plur, n_words)
    feats["second_1000"] = _per_1000(second, n_words)
    feats["third_1000"] = _per_1000(third, n_words)
    feats["yo_nosotros"] = (first_sing - first_plur) / max(first_sing + first_plur, 1)
    feats["orality_1000"] = _per_1000(sum(1 for w in words if w in ORALITY), n_words)
    feats["vocatives_1000"] = _per_1000(sum(1 for w in words if w in VOCATIVES), n_words)
    feats["hedges_1000"] = _per_1000(sum(1 for w in words if w in HEDGES), n_words)

    for name, phrase in PHRASES.items():
        feats[f"frase_{name.replace(' ', '_')}_1000"] = _per_1000(folded.count(phrase), n_words)

    return feats


@dataclass
class StyleRecord:
    doc_id: int
    speaker: str
    date: str
    year: str
    n_words: int
    features: dict


def style_record(doc_id: int, speaker: str, date: str, text: str) -> StyleRecord:
    feats = document_features(text)
    return StyleRecord(doc_id=doc_id, speaker=speaker, date=date, year=str(date)[:4],
                       n_words=int(feats.get("n_words", 0)), features=feats)
