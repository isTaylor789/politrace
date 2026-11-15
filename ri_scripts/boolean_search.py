"""Búsqueda booleana sobre el índice invertido.

Gramática (precedencia de mayor a menor):

    expr    := or_expr
    or_expr := and_expr (OR and_expr)*
    and_expr:= not_expr (AND? not_expr)*        # AND implícito entre términos
    not_expr:= NOT not_expr | atom
    atom    := '(' expr ')' | '"' frase '"' | término

Operadores admitidos: AND/OR/NOT (también `&`, `|`, `!`), paréntesis y frases
entre comillas. Los términos pasan por el mismo analizador que el índice, así que
acentos, mayúsculas y flexión verbal no afectan a la coincidencia.
"""

import re

import numpy as np

from ri_scripts.corpus import read_doc
from ri_scripts.preprocess import STOPWORDS, analyze, analyze_phrase

_LEXER = re.compile(r'"([^"]*)"|(\()|(\))|([^\s()"]+)')

_AND = {"AND", "&"}
_OR = {"OR", "|"}
_NOT = {"NOT", "!"}


class BooleanSyntaxError(ValueError):
    pass


def _tokenize(query: str) -> list[tuple[str, str]]:
    """(tipo, valor) con tipo en {phrase, lparen, rparen, word}."""
    tokens = []
    for phrase, lparen, rparen, word in _LEXER.findall(query):
        if phrase:
            tokens.append(("phrase", phrase))
        elif lparen:
            tokens.append(("lparen", "("))
        elif rparen:
            tokens.append(("rparen", ")"))
        elif word:
            tokens.append(("word", word))
    return tokens


class _Parser:
    def __init__(self, ix, tokens):
        self.ix = ix
        self.tokens = tokens
        self.pos = 0
        self.all_docs = np.arange(ix.n_docs, dtype=np.int64)
        self.ignored: list[str] = []    # stopwords: no están en el índice, no filtran
        self.unknown: list[str] = []    # fuera de vocabulario: no coinciden con nada

    # --- helpers -------------------------------------------------------------
    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def _next(self):
        tok = self._peek()
        self.pos += 1
        return tok

    def _docs_for_term(self, term: str) -> np.ndarray:
        tids = [self.ix.term_id(t) for t in analyze(term)]
        tids = [t for t in tids if t is not None]
        if tids:
            result = self.ix.postings(tids[0])[0].astype(np.int64)
            for tid in tids[1:]:
                result = np.intersect1d(result, self.ix.postings(tid)[0])
            return result

        if not analyze(term):
            # stopword puro (p. ej. "estado", "de"): no está indexado -> no filtra
            self.ignored.append(term)
            return self.all_docs

        self.unknown.append(term)
        return np.array([], dtype=np.int64)

    def _docs_for_phrase(self, phrase: str) -> np.ndarray:
        """Coincidencia exacta: intersección de términos + verificación de secuencia.

        La verificación usa las formas con stemming pero **sin** quitar
        stopwords, de modo que `"estado de emergencia"` exige ese orden exacto.
        """
        content_terms = analyze(phrase)          # términos de contenido (candidatos)
        sequence = analyze_phrase(phrase)        # secuencia literal a verificar
        if not content_terms or not sequence:
            return np.array([], dtype=np.int64)

        candidates = self._docs_for_term(content_terms[0])
        for term in content_terms[1:]:
            candidates = np.intersect1d(candidates, self._docs_for_term(term))
        if candidates.size == 0:
            return candidates

        verified = []
        for doc_id in candidates:
            doc_seq = analyze_phrase(read_doc(int(doc_id), self.ix.corpus_path, self.ix.offsets)["text"])
            if _contains_sequence(doc_seq, sequence):
                verified.append(doc_id)
        return np.asarray(verified, dtype=np.int64)

    # --- gramática -----------------------------------------------------------
    def parse(self) -> np.ndarray:
        result = self._parse_or()
        if self.pos != len(self.tokens):
            raise BooleanSyntaxError(f"sobra token en la posición {self.pos}: {self.tokens[self.pos]}")
        return result

    def _parse_or(self) -> np.ndarray:
        result = self._parse_and()
        while True:
            kind, value = self._peek()
            if kind == "word" and value.upper() in _OR:
                self._next()
                result = np.union1d(result, self._parse_and()).astype(np.int64)
            else:
                return result

    def _parse_and(self) -> np.ndarray:
        result = self._parse_not()
        while True:
            kind, value = self._peek()
            if kind == "word" and value.upper() in _OR:
                return result
            if kind == "word" and value.upper() in _AND:
                self._next()
                result = np.intersect1d(result, self._parse_not()).astype(np.int64)
            elif kind in ("word", "phrase", "lparen"):
                # AND implícito: "presupuesto educación" == "presupuesto AND educación"
                result = np.intersect1d(result, self._parse_not()).astype(np.int64)
            else:
                return result

    def _parse_not(self) -> np.ndarray:
        kind, value = self._peek()
        if kind == "word" and value.upper() in _NOT:
            self._next()
            return np.setdiff1d(self.all_docs, self._parse_not()).astype(np.int64)
        return self._parse_atom()

    def _parse_atom(self) -> np.ndarray:
        kind, value = self._next()
        if kind == "lparen":
            result = self._parse_or()
            closing = self._next()
            if closing[0] != "rparen":
                raise BooleanSyntaxError("falta ')' de cierre")
            return result
        if kind == "phrase":
            return self._docs_for_phrase(value)
        if kind == "word":
            if value.upper() in _AND | _OR | _NOT:
                raise BooleanSyntaxError(f"operador inesperado: {value}")
            return self._docs_for_term(value)
        raise BooleanSyntaxError("se esperaba un término, una frase o '('")


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    first = needle[0]
    for i, token in enumerate(haystack):
        if token == first and haystack[i : i + n] == needle:
            return True
    return False


def parse_boolean(ix, query: str, speaker: str | None = None, diagnostics: bool = False):
    """Devuelve los doc_id que satisfacen la expresión (sin ranking)."""
    parser = _Parser(ix, _tokenize(query))
    matches = parser.parse()
    if speaker is not None:
        matches = np.intersect1d(matches, ix.docs_of_speaker(speaker))
    if diagnostics:
        info = {
            "ignored_stopwords": list(dict.fromkeys(parser.ignored)),
            "unknown_terms": list(dict.fromkeys(parser.unknown)),
        }
        return matches, info
    return matches


def search_boolean(ix, query: str, limit: int = 20, speaker: str | None = None,
                   sort: str = "date", diagnostics: bool = False):
    """Coincidencia booleana. No hay score: el orden es por fecha (o por doc_id)."""
    matches, info = parse_boolean(ix, query, speaker=speaker, diagnostics=True)
    if matches.size == 0:
        return ([], info) if diagnostics else []

    if sort == "date":
        order = np.array(sorted(matches, key=lambda d: (ix.dates[d], d), reverse=True))
    else:
        order = np.sort(matches)
    order = order[:limit]

    from ri_scripts.results import make_results

    results = make_results(ix, order, np.ones(len(order)), query)
    return (results, info) if diagnostics else results
