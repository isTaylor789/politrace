"""Autómatas finitos del extractor (EI 3.2.2).

1. `AhoCorasick` — autómata de múltiples patrones sobre **secuencias de tokens**.
   Se construye explícitamente con sus tres funciones clásicas:

       goto(q, a)    transición por símbolo (diccionario de aristas)
       fail(q)       enlace de fallo (el sufijo propio más largo que es prefijo)
       out(q)        patrones que terminan en q

   Coste de construcción O(sum |P_i|), coste de búsqueda O(n + z) con n tokens y
   z coincidencias, independientemente del número de patrones. Ésa es la razón de
   usarlo: hay ~90 disparadores y 4.5k discursos (~22M tokens); con `str.find`
   por patrón serían 2 mil millones de comparaciones.

2. `NegationAutomaton` — DFA de 4 estados que decide si el disparador detectado
   está bajo el alcance de una negación, mirando la ventana previa de tokens.
   Se implementa como tabla de transición explícita para poder dibujarlo en el
   reporte.
"""

from dataclasses import dataclass, field

from ei_scripts import config, lexicon

# -----------------------------------------------------------------------------
# 1. Aho-Corasick
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class Match:
    """Coincidencia: patrón `pid` que termina en el token `end` (inclusive)."""

    pid: int
    end: int          # índice del último token del patrón
    start: int        # índice del primer token del patrón
    length: int


class AhoCorasick:
    """Autómata de patrones multi-token con enlaces de fallo.

    `add_pattern` recibe la secuencia de tokens plegados y una etiqueta opaca
    (`tag`), que puede ser cualquier cosa: aquí guardamos (clase, canónico) del
    disparador o el tipo del gazetteer.
    """

    def __init__(self):
        self.goto: list[dict[str, int]] = [{}]   # función goto
        self.fail: list[int] = [0]               # enlace de fallo
        self.out: list[list[tuple[int, int]]] = [[]]  # (pid, longitud) por estado
        self.tags: list = []                     # tag de cada pid
        self.patterns: list[tuple[str, ...]] = []
        self._alpha: set[str] = set()

    # -- construcción ---------------------------------------------------------
    def add_pattern(self, seq, tag=None) -> int:
        seq = tuple(seq)
        if not seq:
            raise ValueError("patrón vacío")
        state = 0
        for symbol in seq:
            nxt = self.goto[state].get(symbol)
            if nxt is None:
                nxt = len(self.goto)
                self.goto[state][symbol] = nxt
                self.goto.append({})
                self.fail.append(0)
                self.out.append([])
            state = nxt
            self._alpha.add(symbol)
        pid = len(self.patterns)
        self.patterns.append(seq)
        self.tags.append(tag)
        self.out[state].append((pid, len(seq)))
        return pid

    def build(self) -> "AhoCorasick":
        """BFS sobre goto para calcular los enlaces de fallo (estilo Aho-Corasick)."""
        queue: list[int] = []
        for child in self.goto[0].values():          # profundidad 1 falla a la raíz
            self.fail[child] = 0
            queue.append(child)
        head = 0
        while head < len(queue):
            state = queue[head]
            head += 1
            for symbol, child in self.goto[state].items():
                queue.append(child)
                # fallo del hijo = seguir fallos del padre hasta poder avanzar
                f = self.fail[state]
                while f and symbol not in self.goto[f]:
                    f = self.fail[f]
                self.fail[child] = self.goto[f].get(symbol, 0)
                if self.fail[child] == child:
                    self.fail[child] = 0
                # herencia de salidas: un patrón que termina en el fallo también
                # termina aquí (p. ej. "a" dentro de "vamos a")
                self.out[child] = self.out[child] + self.out[self.fail[child]]
        return self

    # -- búsqueda -------------------------------------------------------------
    def scan(self, tokens) -> list[Match]:
        """Recorre la secuencia de símbolos y devuelve todas las coincidencias."""
        goto, fail, out = self.goto, self.fail, self.out
        state = 0
        matches: list[Match] = []
        for i, symbol in enumerate(tokens):
            while state and symbol not in goto[state]:
                state = fail[state]
            state = goto[state].get(symbol, 0)
            if state:
                for pid, length in out[state]:
                    matches.append(Match(pid, i, i - length + 1, length))
        return matches

    # -- utilidades de reporte ------------------------------------------------
    def stats(self) -> dict:
        return {
            "estados": len(self.goto),
            "patrones": len(self.patterns),
            "aristas": sum(len(g) for g in self.goto),
            "alfabeto": len(self._alpha),
            "profundidad_max": max((len(p) for p in self.patterns), default=0),
        }


# -----------------------------------------------------------------------------
# 2. Autómata de negación
# -----------------------------------------------------------------------------
# Estados: 0 = limpio · 1 = negado · 2 = doble negación (equivale a limpio) ·
#          3 = bloqueado (apareció puntuación fuerte: la ventana se cierra)
NEG_CLEAN, NEG_NEGATED, NEG_DOUBLE, NEG_BLOCKED = 0, 1, 2, 3

NEG_TRANSITIONS = {
    #             negador      puntuación   otro
    NEG_CLEAN:   (NEG_NEGATED, NEG_BLOCKED, NEG_CLEAN),
    NEG_NEGATED: (NEG_DOUBLE,  NEG_BLOCKED, NEG_NEGATED),
    NEG_DOUBLE:  (NEG_NEGATED, NEG_BLOCKED, NEG_DOUBLE),
    NEG_BLOCKED: (NEG_BLOCKED, NEG_BLOCKED, NEG_BLOCKED),
}


class NegationAutomaton:
    """DFA que se lee hacia atrás desde el disparador y decide el alcance.

    Trabaja en el espacio de **palabras** (índice `pos`), pero mira la puntuación
    del texto crudo entre palabra y palabra: una coma no cierra la ventana de
    negación, un punto sí. Devuelve "afirmado", "negado" o "atenuado" (hedges).
    """

    STRONG_PUNCT = {".", ";", ":", "!", "?", "¿", "¡"}

    def __init__(self, window: int | None = None):
        self.window = config.NEGATION_WINDOW if window is None else window

    def _punct_between(self, words, i: int, text: str | None) -> str:
        """Puntuación fuerte que separa words[i-1] de words[i] (si hay texto)."""
        if text is None or i == 0:
            return ""
        gap = text[words[i - 1].end - words[0].start:words[i].start - words[0].start]
        return "".join(ch for ch in gap if ch in self.STRONG_PUNCT)

    def state_before(self, words, pos: int, text: str | None = None) -> int:
        """Corre el DFA sobre los `window` tokens previos a `pos`."""
        state = NEG_CLEAN
        start = max(0, pos - self.window)
        for i in range(start, pos):
            if self._punct_between(words, i, text):
                state = NEG_TRANSITIONS[state][1]
                continue
            if words[i].folded in lexicon.NEGATORS:
                state = NEG_TRANSITIONS[state][0]
            else:
                state = NEG_TRANSITIONS[state][2]
        return state

    def polarity(self, words, pos: int, text: str | None = None) -> str:
        state = self.state_before(words, pos, text)
        window = words[max(0, pos - self.window):pos]
        if state == NEG_NEGATED:
            return "negado"
        if any(w.folded in lexicon.HEDGES for w in window):
            return "atenuado"
        return "afirmado"

    def describe(self) -> dict:
        """Tabla de transición en forma legible (para el reporte)."""
        names = {0: "limpio", 1: "negado", 2: "doble", 3: "bloqueado"}
        return {
            names[s]: {"negador": names[t[0]], "puntuacion": names[t[1]], "otro": names[t[2]]}
            for s, t in NEG_TRANSITIONS.items()
        }


# -----------------------------------------------------------------------------
# 3. Construcción de los autómatas del proyecto
# -----------------------------------------------------------------------------


def build_trigger_automaton() -> AhoCorasick:
    """Disparadores + formas de futuro sintético + gazetteers de lugares/organizaciones.

    Las formas de futuro ("construiré") son patrones de un solo token, así que el
    mismo autómata resuelve los tres léxicos en una sola pasada.
    """
    ac = AhoCorasick()
    for seq, (cls, canon) in lexicon.TRIGGERS.items():
        ac.add_pattern(seq, tag=("trigger", cls, canon))

    for verb in lexicon.ACTION_VERBS:
        for form in lexicon.future_forms_1st(verb):
            # "construiré / entregaremos": el verbo es el disparador.
            ac.add_pattern((form,), tag=("trigger", "future_1p_syn", "futuro"))
        for form in lexicon.future_forms_3rd(verb):
            # "el gobierno entregará": sólo con sujeto atribuible (ver resolve_triggers)
            ac.add_pattern((form,), tag=("trigger", "future_3p_syn", "futuro"))

    for seq in lexicon.PLACE_SEQS:
        ac.add_pattern(seq, tag=("place", None, " ".join(seq)))
    for seq in lexicon.ORG_SEQS:
        ac.add_pattern(seq, tag=("org", None, " ".join(seq)))
    return ac.build()


@dataclass
class TriggerHit:
    """Disparador ya resuelto sobre una oración (sin solapes)."""

    start: int
    end: int
    seq: tuple[str, ...]
    cls: str
    canon: str
    strength: float
    polarity: str = "afirmado"


# Marcadores de atribución: el anuncio tiene que ser del hablante o de su gobierno.
ATTRIBUTION_MARKERS = {
    "vamos", "nosotros", "nos", "nuestro", "nuestra", "nuestros", "nuestras",
    "gobierno", "mi", "voy", "comprometo", "comprometemos", "prometo",
}

# Personas que "van a" hacer algo: si el sujeto es el gobierno, es anuncio.
THIRD_PERSON_TRIGGERS = {("va", "a"), ("van", "a")}
THIRD_PERSON_CLASSES = {"future_3p_syn"}


def resolve_triggers(matches: list[Match], ac: AhoCorasick, words, text: str | None = None,
                     check_attribution: bool = True) -> list[TriggerHit]:
    """Filtra coincidencias de tipo trigger y elimina solapes.

    `words` es la lista de tokens de palabra (sin puntuación): los índices de
    `Match` viven en ese espacio.

    Reglas aplicadas, en orden:
      1. el patrón más largo gana (\"me comprometo a\" > \"me comprometo\");
      2. \"vamos a ver / vamos a platicar\" no son promesas (verbo de relleno);
      3. \"el anuncio de\" no es performativo (determinante antes): se descarta;
      4. \"va a / van a\" exige atribución al hablante/gobierno en la oración.
    """
    neg = NegationAutomaton()
    hits: list[TriggerHit] = []
    trig = [m for m in matches if ac.tags[m.pid][0] == "trigger"]
    trig.sort(key=lambda m: (-m.length, m.start))
    occupied: set[int] = set()

    attribution = None
    if check_attribution:
        attribution = any(w.folded in ATTRIBUTION_MARKERS for w in words)

    for m in trig:
        span = set(range(m.start, m.end + 1))
        if span & occupied:
            continue
        pattern = ac.patterns[m.pid]
        _, cls, canon = ac.tags[m.pid]

        # 2. perífrasis + verbo de relleno
        if pattern in {("voy", "a"), ("vamos", "a"), ("va", "a"), ("van", "a")}:
            nxt = m.end + 1
            if nxt < len(words) and words[nxt].folded in lexicon.STOP_VERBS:
                continue
        # 3. \"el anuncio\": sustantivo, no acto de habla
        if pattern == ("anuncio",) and m.start > 0:
            prev = words[m.start - 1].folded
            if prev in lexicon.DETS or prev in {"dicho", "gran", "primer", "nuevo", "ultimo"}:
                continue
        # 4. tercera persona sin sujeto atribuible
        if check_attribution and not attribution:
            if pattern in THIRD_PERSON_TRIGGERS or cls in THIRD_PERSON_CLASSES:
                continue

        occupied |= span
        hits.append(
            TriggerHit(
                start=m.start,
                end=m.end,
                seq=pattern,
                cls=cls,
                canon=canon,
                strength=config.STRENGTH[cls],
                polarity=neg.polarity(words, m.start, text),
            )
        )
    hits.sort(key=lambda h: h.start)
    return hits
