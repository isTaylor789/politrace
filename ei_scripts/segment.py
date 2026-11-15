"""Segmentación en oraciones y tokenización con offsets sobre el texto crudo.

Dos vistas del mismo texto, sincronizadas por offsets de carácter:

* **cruda**  — conserva mayúsculas y acentos; es la que se devuelve al usuario y
  la que usan los gazetteers (los nombres propios dependen de la mayúscula) y
  las expresiones regulares de slots (meses, montos, porcentajes).
* **plegada** — minúsculas sin acentos (`ri_scripts.preprocess.fold`), la que
  consume el autómata de disparadores. `fold` conserva la longitud, así que el
  offset de un token es el mismo en ambas vistas.
"""

import re
from dataclasses import dataclass, field

import nltk

from ri_scripts.preprocess import fold
from ei_scripts import config

# Palabras y números. Se incluyen los dígitos con separador de miles para que
# "1,500" y "3.5" queden como un solo token.
_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|\d+(?:[.,]\d+)*")
_PUNCT_RE = re.compile(r"[.,;:!?¡¿\"'()\[\]\-–—…“”‘’/]")
# Para partir oraciones gigantes: cortes "fuertes" primero, luego comas.
_SPLIT_STRONG = re.compile(r"[;:]\s")
_SPLIT_WEAK = re.compile(r",\s")

_PUNKT = None


def _punkt():
    global _PUNKT
    if _PUNKT is None:
        _PUNKT = nltk.data.load("tokenizers/punkt/spanish.pickle")
    return _PUNKT


@dataclass(slots=True)
class Token:
    surface: str     # forma original ("Vamos")
    folded: str      # forma plegada ("vamos")
    start: int       # offset en el texto completo del documento
    end: int
    is_word: bool    # False para puntuación

    @property
    def stem(self) -> str:
        """Stem perezoso (sólo se calcula si alguien lo pide)."""
        from ri_scripts.preprocess import stem

        return stem(self.folded)


@dataclass(slots=True)
class Sentence:
    idx: int
    start: int
    end: int
    text: str
    tokens: list[Token] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return sum(1 for t in self.tokens if t.is_word)


def tokenize(text: str, offset: int = 0) -> list[Token]:
    """Palabras y números (con su puntuación) en orden de aparición."""
    tokens: list[Token] = []
    events = [(m.start(), m.end(), True) for m in _TOKEN_RE.finditer(text)]
    events += [(m.start(), m.end(), False) for m in _PUNCT_RE.finditer(text)]
    events.sort()
    for start, end, is_word in events:
        surface = text[start:end]
        tokens.append(Token(surface, fold(surface), start + offset, end + offset, is_word))
    return tokens


def _split_span(text: str, start: int, end: int):
    """Parte un span largo por ; : y, si hace falta, por comas."""
    def _cut(pattern):
        out, cursor = [], start
        for m in pattern.finditer(text, start, end):
            if m.start() <= cursor:
                continue
            out.append((cursor, m.start()))
            cursor = m.end()
        out.append((cursor, end))
        return [(s, e) for s, e in out if text[s:e].strip()]

    rough = _cut(_SPLIT_STRONG)
    pieces = []
    for s, e in rough:
        if sum(1 for _ in _TOKEN_RE.finditer(text, s, e)) > config.MAX_SENT_TOKENS:
            pieces.extend(_split_within(text, s, e))
        else:
            pieces.append((s, e))
    return pieces or [(start, end)]


def _split_within(text: str, start: int, end: int):
    """Corte por comas dentro de [start, end] agrupando hasta MAX tokens."""
    spans, cursor, count = [], start, 0
    for m in _SPLIT_WEAK.finditer(text, start, end):
        chunk = text[cursor:m.start()]
        n = sum(1 for _ in _TOKEN_RE.finditer(chunk))
        if count + n > config.MAX_SENT_TOKENS and count > 0:
            spans.append((cursor, m.start()))
            cursor, count = m.end(), 0
        else:
            count += n
    spans.append((cursor, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def segment(text: str) -> list[Sentence]:
    """Texto del discurso -> oraciones con offsets absolutos."""
    sentences: list[Sentence] = []
    spans = list(_punkt().span_tokenize(text))
    idx = 0
    for start, end in spans:
        n_tokens = sum(1 for _ in _TOKEN_RE.finditer(text, start, end))
        pieces = [(start, end)] if n_tokens <= config.MAX_SENT_TOKENS else _split_span(text, start, end)
        for s, e in pieces:
            body = text[s:e]
            if not body.strip():
                continue
            sentences.append(
                Sentence(idx=idx, start=s, end=e, text=body, tokens=tokenize(body, s))
            )
            idx += 1
    return sentences
