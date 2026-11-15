"""Segmentación, filtrado y anotación de las unidades de resumen (oraciones).

Reutiliza el segmentador del módulo de EI (`ei_scripts.segment.segment`): punkt
para español, corte extra por ``;`` ``:`` ``,`` cuando una oración supera 90
palabras, y offsets de carácter sobre el texto crudo.

Sobre ese segmentado se aplican los **criterios de extracción** del temario
(4.2): la unidad debe ser una oración completa, con contenido léxico suficiente,
sin marcas de transcripción y de longitud legible.
"""

import re
from dataclasses import dataclass, field

from ei_scripts.segment import segment as _segment
from ri_scripts.preprocess import analyze

from rat_scripts import config

_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")
_NUMBER_RE = re.compile(r"\d|\bpor\s?ciento\b|%")

# Etiquetas de hablante/transcripción al inicio de la oración ("PRESIDENTE. –",
# "El Presidente:", "MODERADOR:"). Se eliminan para que la oración se lea sola.
_SPEAKER_TAG = re.compile(
    r"^\s*(?:~+\s*)?(?:el\s+|la\s+)?"
    r"(?:presidente|presidenta|interlocutor[a]?|moderador[a]?|presentador[a]?|"
    r"periodista|pregunta|respuesta|voz)\s*[.:\-–—]+\s*",
    re.IGNORECASE,
)
_BULLETS = re.compile(r"^[\s\-–—•*·>]+")

# Cabecera de la transcripción ("Palabras del Presidente ... Javier Milei:").
# Se recorta hasta los dos puntos para que la oración empiece en el saludo.
_HEADER = re.compile(
    r"^(?:palabras|discurso|intervenci[oó]n|declaraci[oó]n|comparecencia|"
    r"mensaje|conferencia|alocuci[oó]n|comparece)\b[^.]{0,200}?:\s*",
    re.IGNORECASE,
)
# Cabecera en mayúsculas: "~~~PRESIDENTE ANDRÉS MANUEL LÓPEZ OBRADOR 2023:"
_UPPER_HEADER = re.compile(
    r"^(?:~+\s*)?(?:PRESIDENTE|PRESIDENTA|MODERADOR|MODERADORA|INTERLOCUTOR|"
    r"INTERLOCUTORA|GOBIERNO)\b[^.]{0,140}?:\s*"
)
# Nombre de otro participante en mayúsculas: "GRACIELA MÁRQUEZ COLÍN,
# SECRETARIA DE ECONOMÍA: ...".
_NAME_TAG = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s.,\-]{3,80}?:\s*")
# Marcas invisibles de dirección de texto que traen algunas transcripciones.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
# Transcripciones pegadas: "en el año 2024.Y en ese sentido" -> "2024. Y"
_GLUED = re.compile(r"(?<=\d)\.(?=[A-ZÁÉÍÓÚÑ])")
_WS = re.compile(r"\s+")

# Muletillas orales al inicio ("Este, señorías...", "Bueno, ..."): se recortan
# porque el resumen se lee fuera de la conversación.
_ORAL_FILLER = re.compile(r"^(este|esto|bueno|pues|o sea|mira|eh|digamos),\s+", re.I)
# Saludo inicial pegado a la primera oración ("Muy buenas tardes: Hoy ...").
_GREETING = re.compile(
    r"^(?:muy\s+)?(?:buenos|buenas)\s+(?:d[ií]as|tardes|noches)"
    r"(?:\s+a\s+(?:todas|todos))?\s*[:.,;!]*\s*",
    re.I,
)
# Vocativos: "Amigas, amigos de Querétaro:", "Excelencias, ...".
_VOCATIVE = re.compile(
    r"^(?:señorías|excelencias|amigas|amigos|compañeras|compañeros|hermanas|"
    r"hermanos|estimados|distinguidos)\b[^.!?]{0,60}?:\s*|"
    r"^(?:señorías|excelencias)\s*,\s*",
    re.I,
)

# Ruido de transcripción: aplausos, risas, cortes técnicos, cabeceras y fechas
# sueltas. Estas "oraciones" no aportan contenido al resumen.
_FILLER = [
    re.compile(r"^~+"),
    re.compile(r"\b(aplausos?|risas?|corte de list[oó]n|c[aá]mara|video|pantalla)\b", re.I),
    re.compile(r"^(muchas|much[ií]simas)\s+gracias\b[^.!?]{0,40}[.!]*$", re.I),
    re.compile(r"^(muy\s+)?(buenos|buenas)\s+(d[ií]as|tardes|noches)\b[^.!?]{0,30}[:.!]*$", re.I),
    re.compile(r"^\d{1,2}[:.]\d{2}(:\d{2})?$"),
    re.compile(r"^[\W\d]+$"),
    re.compile(r"^(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo),?\s+\d{1,2}\s+de\s+\w+\s+de\s+\d{4}[.!]*$", re.I),
    re.compile(r"^(agradezco|saludo)\b.{0,40}(aplauso|presencia)", re.I),
    # pie de página de las transcripciones de AMLO en el CSV
    re.compile(r"^publicado por\b", re.I),
    re.compile(r"^.{0,40}\bpublicado por\b.{0,80}de \d{4}$", re.I),
]


@dataclass
class Unit:
    """Una oración candidata a entrar al resumen."""

    idx: int                 # orden dentro del documento filtrado
    start: int               # offset en el texto original
    end: int
    text: str                # texto limpio, listo para mostrar
    terms: list              # términos analizados (stem, sin stopwords)
    n_words: int             # palabras superficiales
    n_numbers: int
    paragraph: int           # índice del párrafo de origen
    position: float          # posición relativa 0..1 dentro del documento
    sent_idx: int = 0        # índice en el segmentado original (trazabilidad)
    doc_id: int | None = None
    group_id: int | None = None   # grupo temático en el resumen multi-documento
    doc_label: str = ""      # "hablante · fecha" en resumen multi-documento
    extra: dict = field(default_factory=dict)


def clean_sentence(text: str) -> str:
    """Normaliza espacios, quita etiquetas de hablante y viñetas."""
    body = _INVISIBLE.sub("", text)
    body = _WS.sub(" ", body).strip()
    body = _SPEAKER_TAG.sub("", body)
    body = _UPPER_HEADER.sub("", body)
    body = _HEADER.sub("", body)
    body = _NAME_TAG.sub("", body)
    body = _ORAL_FILLER.sub("", body)
    body = _GREETING.sub("", body)
    body = _VOCATIVE.sub("", body)
    body = _BULLETS.sub("", body).strip()
    body = _GLUED.sub(". ", body)
    if not body:
        return ""
    first = body[0]
    if first.islower():
        body = first.upper() + body[1:]
    return body


def is_filler(text: str) -> bool:
    return any(pat.search(text) for pat in _FILLER)


def _paragraph_starts(text: str) -> list:
    """Offsets de inicio de cada párrafo (separados por línea en blanco)."""
    starts, cursor = [0], 0
    for match in re.finditer(r"\n\s*\n", text):
        cursor = match.end()
        if cursor < len(text):
            starts.append(cursor)
    return starts


def _paragraph_of(starts: list, offset: int) -> int:
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def count_numbers(text: str) -> int:
    return len(_NUMBER_RE.findall(text))


def make_unit(text: str, *, idx: int = 0, position: float = 0.0,
              doc_id: int | None = None, doc_label: str = "",
              group_id: int | None = None) -> "Unit | None":
    """Construye una `Unit` a partir de una oración ya limpia (sin re-segmentar).

    Se usa en el resumen multi-documento para reutilizar las oraciones que cada
    resumen individual ya eligió.
    """
    body = clean_sentence(text)
    if not body or is_filler(body):
        return None
    terms = analyze(body)
    if len(terms) < 2:
        return None
    return Unit(
        idx=idx, start=0, end=0, text=body, terms=terms,
        n_words=count_words(body), n_numbers=count_numbers(body),
        paragraph=0, position=position, doc_id=doc_id, doc_label=doc_label,
        group_id=group_id,
    )


def build_units(text: str, *, title: str = "", relax: bool | None = None) -> list:
    """Texto crudo -> lista de `Unit` filtradas y ordenadas.

    `relax=None` decide automáticamente: si el filtrado estricto deja menos de
    `MIN_UNITS` oraciones se reintenta con criterios laxos (textos cortos o mal
    puntuados).
    """
    text = text or ""
    starts = _paragraph_starts(text)
    raw = _segment(text)

    def _collect(min_words: int, max_words: int | None) -> list:
        units = []
        seen = set()
        for sent in raw:
            body = clean_sentence(sent.text)
            if not body or is_filler(body):
                continue
            n = count_words(body)
            if n < min_words or (max_words is not None and n > max_words):
                continue
            terms = analyze(body)
            if len(terms) < 2:
                continue
            key = " ".join(terms)
            if key in seen:            # oración repetida literalmente
                continue
            seen.add(key)
            units.append(
                Unit(
                    idx=len(units),
                    start=sent.start,
                    end=sent.end,
                    text=body,
                    terms=terms,
                    n_words=n,
                    n_numbers=len(_NUMBER_RE.findall(body)),
                    paragraph=_paragraph_of(starts, sent.start),
                    position=0.0,
                    sent_idx=sent.idx,
                )
            )
        return units

    units = _collect(config.MIN_SENT_WORDS, config.MAX_SENT_WORDS)
    if relax is None:
        relax = len(units) < config.MIN_UNITS
    if relax:
        units = _collect(4, None)

    # posición relativa y longitud mediana (la usan los scorers)
    total = max(1, len(units) - 1)
    for u in units:
        u.position = u.idx / total
    return units
