"""Pistas discursivas y heurísticas de posición (temario 4.4.4).

Tres señales baratas, deterministas y explicables:

* **posición** — en un discurso político el planteamiento está al inicio y el
  cierre al final; el centro suele ser digresión o agradecimientos.
* **pistas discursivas** — conectores de síntesis, fórmulas de compromiso,
  verbos de relevancia y densidad de datos (cifras, porcentajes).
* **solapamiento con el título** — el título del discurso ya declara el tópico
  (4.3.1), así que las oraciones que lo retoman son más representativas.
"""

import math
import re

import numpy as np

from ri_scripts.preprocess import analyze, fold

from rat_scripts import config

# Cada patrón se busca sobre el texto PLEGADO (minúsculas, sin acentos), así que
# se escriben sin tildes. Cada categoría aporta un peso fijo a la oración.
CUES = {
    "sintesis": (1.00, [
        r"\ben resumen\b", r"\ben sintesis\b", r"\ben conclusion\b",
        r"\bpara concluir\b", r"\ben definitiva\b", r"\bpor ultimo\b",
        r"\bfinalmente\b", r"\bpara terminar\b", r"\brecapitulando\b",
        r"\bquiero terminar\b", r"\bpara finalizar\b",
    ]),
    "compromiso": (0.95, [
        r"\bvamos a\b", r"\bvoy a\b", r"\bme comprometo\b",
        r"\bnos comprometemos\b", r"\banuncio que\b", r"\bproponemos\b",
        r"\bpropongo\b", r"\bnos proponemos\b", r"\btenemos que\b",
        r"\bdebemos\b", r"\bhay que\b", r"\bes necesario\b",
        r"\bel objetivo\b", r"\bla meta\b", r"\bmi compromiso\b",
    ]),
    "relevancia": (0.65, [
        r"\bes importante\b", r"\blo mas importante\b", r"\blo fundamental\b",
        r"\bla prioridad\b", r"\bel reto\b", r"\bquiero destacar\b",
        r"\bquiero subrayar\b", r"\bquiero decir\b", r"\bdestaco\b",
    ]),
    "causalidad": (0.60, [
        r"\bpor lo tanto\b", r"\ben consecuencia\b", r"\bpor eso\b",
        r"\bdebido a\b", r"\bcomo resultado\b", r"\bde manera que\b",
        r"\bgracias a\b", r"\ba pesar de\b",
    ]),
    "gobierno": (0.55, [
        r"\bmi gobierno\b", r"\bnuestro gobierno\b", r"\beste gobierno\b",
        r"\bhemos logrado\b", r"\bhemos hecho\b", r"\bestamos haciendo\b",
        r"\bhemos avanzado\b", r"\bnuestro pueblo\b", r"\bhemos construido\b",
    ]),
}

_NUMBER_RE = re.compile(r"\d|\bpor ciento\b|%")

# Una oración que empieza a media frase ("...que la vacuna...") se lee mal en un
# resumen: se penaliza, no se descarta (a veces no hay alternativa).
_CONTINUATION = re.compile(
    r"^(y|o|pero|porque|pues|aunque|que|ademas|sin embargo|es decir|esto|eso|"
    r"muestra que|se trata|de igual forma|asimismo|entonces)\b"
)
# Fórmulas huecas: muy frecuentes en el habla política y con cero contenido.
_GENERIC = re.compile(
    r"\b(es una cosa|le quiero decir|les quiero decir|como ustedes saben|"
    r"como todos saben|como bien saben|en este sentido|valga la redundancia|"
    r"hay que decir|cabe mencionar|cabe senalar|es importante senalar|"
    r"mi (reciente |nuevo |ultimo )?libro|presentar (mi|el) libro)\b"
)
# Trámite de la sesión: agenda, turnos, informes de rutina. Muy frecuente en las
# mañaneras y en las sesiones parlamentarias; no es contenido del resumen.
_PROCEDURAL = re.compile(
    r"\b(informe de seguridad|informe del|vamos a informar|les voy a informar|"
    r"me acompanan|en el presidium|tiene el uso de la palabra|tiene la palabra|"
    r"damos inicio|en su turno|toma la palabra|microfon\w*|replic\w*|"
    r"orden del dia|punto del orden del dia)\b"
)
# Actos ceremoniales, deporte y entretenimiento: en un resumen de gestión
# aportan poco (condecoraciones, homenajes, medallas).
_CEREMONY = re.compile(
    r"\b(en este acto|en este evento|honrar|homenaj\w*|condecor\w*|galardon\w*|"
    r"medall\w*|trayectori\w*|felicit\w*|enhorabuena|deport\w*|olimp\w*|"
    r"atleta\w*|brindis)\b"
)
# Cortesía y despedidas: aportan poco contenido y mucho ruido al cierre.
_COURTESY = [
    re.compile(r"\b(muchas|muchisimas) gracias\b"),
    re.compile(r"\b(feliz navidad|felices fiestas|felicidades|enhorabuena)\b"),
    re.compile(r"\bque dios\b|\bbendiga\b|\bbendiciones\b"),
    re.compile(r"\bbienvenid[oa]s\b|\bun fuerte abrazo\b"),
    re.compile(r"\bquiero agradecer\b|\bagradezco\b"),
    re.compile(r"\bquiero saludar\b|\bsaludar a\b|\bme da mucho gusto\b|\bes un honor para mi\b"),
]

# Se compilan una sola vez (búsqueda lineal sobre la oración plegada).
_CUES_C = {
    name: (weight, [re.compile(p) for p in pats])
    for name, (weight, pats) in CUES.items()
}


def cue_score(text: str):
    """(score en [0,1], lista de categorías activas) para una oración."""
    folded = fold(text)
    total, hits = 0.0, []
    for name, (weight, patterns) in _CUES_C.items():
        if any(p.search(folded) for p in patterns):
            total += weight
            hits.append(name)
    # saturación suave: 1 - exp(-Σw) evita que una oración con 5 conectores
    # domine sobre una con 1 (el efecto marginal decrece).
    return 1.0 - math.exp(-total), hits


def numeric_score(n_numbers: int, n_words: int) -> float:
    """Densidad de cifras, saturada: un discurso con datos concretos los cita."""
    density = n_numbers / max(1, n_words)
    return min(1.0, 4.0 * density)


def position_score(position: float, lead: float = 0.30, tail: float = 0.12) -> float:
    """Prior de posición: bonus al inicio (lead) y, menor, al cierre (tail).

        π(p) = 0.85 * max(0, 1 - p/lead) + 0.15 * max(0, 1 - (1-p)/tail)

    con p ∈ [0,1] la posición relativa de la oración. Vale 0.85 al inicio, 0.15
    al final y 0 en el cuerpo central del discurso. El peso del cierre es bajo a
    propósito: en los discursos el final suele ser agradecimientos y deseos.
    """
    lead_bonus = max(0.0, 1.0 - position / lead) if lead > 0 else 0.0
    tail_bonus = max(0.0, 1.0 - (1.0 - position) / tail) if tail > 0 else 0.0
    return 0.85 * lead_bonus + 0.15 * tail_bonus


def title_overlap(terms: list, title_terms: set) -> float:
    """Fracción de términos del título presentes en la oración."""
    if not title_terms:
        return 0.0
    present = sum(1 for t in title_terms if t in set(terms))
    return present / len(title_terms)


def title_terms(title: str) -> set:
    return set(analyze(title or "")) - {"discurso", "palabras", "presidente"}


def discourse_scores(units: list, title: str = "", extra_penalties: bool = False) -> dict:
    """Componentes del bloque discursivo, normalizadas a [0,1].

    `extra_penalties=True` activa las penalizaciones de trámite y ceremonia, que
    sólo tienen sentido en el resumen multi-documento (en un discurso de
    condecoraciones esos actos son el contenido).
    """
    tt = title_terms(title)
    position = np.array([position_score(u.position) for u in units], dtype=float)
    cue, cue_hits = [], []
    for u in units:
        value, hits = cue_score(u.text)
        cue.append(value)
        cue_hits.append(hits)
    cue = np.asarray(cue, dtype=float)
    numeric = np.array([numeric_score(u.n_numbers, u.n_words) for u in units], dtype=float)
    overlap = np.array([title_overlap(u.terms, tt) for u in units], dtype=float)

    w = config.DISCOURSE_WEIGHTS
    combined = w["position"] * position + w["cue"] * cue + w["title"] * overlap
    # la densidad de cifras es una pista débil adicional, no un componente propio
    combined = 0.88 * combined + 0.12 * numeric

    # penalizaciones de legibilidad/ruido (factores multiplicativos)
    penalty = np.ones(len(units))
    for i, u in enumerate(units):
        folded = fold(u.text)
        if _CONTINUATION.match(folded):
            penalty[i] *= 0.75
        if any(p.search(folded) for p in _COURTESY):
            penalty[i] *= 0.55
        if _GENERIC.search(folded):
            penalty[i] *= 0.80
        if extra_penalties and _PROCEDURAL.search(folded):
            penalty[i] *= 0.75
        if extra_penalties and _CEREMONY.search(folded):
            penalty[i] *= 0.55
    combined = combined * penalty

    return {
        "discourse": combined,
        "position": position,
        "cue": cue,
        "cue_hits": cue_hits,
        "numeric": numeric,
        "title": overlap,
        "penalty": penalty,
    }
