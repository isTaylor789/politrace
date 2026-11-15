"""Gramática independiente del contexto y parser CYK (EI 3.2.3).

Idea central: no se parsea español completo (inviable con una CFG escrita a
mano), sino el **molde de la promesa** sobre una secuencia de chunks. El chunker
colapsa el léxico en unidades con papel semántico (disparador, acción, sintagma
nominal, plazo, monto, lugar, beneficiario) y la CFG decide si esas unidades
forman o no un marco de evento válido.

Gramática (terminales = etiquetas de chunk):

    S      -> TRIG REST                    (la promesa empieza por el disparador)
    S      -> PRE TRIG REST                (contexto previo: lugar, sintagma, ...)
    REST   -> ACC                          (sólo la acción)
    REST   -> ACC MODS
    REST   -> MODS ACC
    REST   -> MODS ACC MODS
    MODS   -> MOD
    MODS   -> MOD MODS
    MOD    -> SN | LOC | ORG | BENEF | PLAZO | MONTO | MISC
    PRE    -> MOD

`ACC` es obligatorio: sin un verbo de acción prometible no hay promesa. Eso es lo
que descarta "vamos a ver", "vamos a platicar", "hay que ver".

El parser es el CYK clásico sobre la secuencia de chunks. Con n chunks (n ≤ 40 en
la práctica) el coste es O(n^3 · |P|), despreciable: milisegundos por oración.
"""

from dataclasses import dataclass, field

from ei_scripts import lexicon

# -----------------------------------------------------------------------------
# Reglas
# -----------------------------------------------------------------------------
TERMINALS = {"TRIG", "VTRIG", "ACC", "SN", "LOC", "ORG", "BENEF", "PLAZO", "MONTO", "MISC"}

# La gramática se escribe en forma binaria (CNF) para que el CYK la use tal cual.
RULES: list[tuple[str, tuple[str, ...]]] = [
    # -- oración ------------------------------------------------------------
    ("S", ("PRE", "S2")),
    ("S", ("S2",)),
    ("S2", ("TRIG", "REST")),
    ("S2", ("VTRIG", "MODS")),
    # -- cláusula: la acción es obligatoria ---------------------------------
    ("REST", ("ACC",)),
    ("REST", ("ACC", "MODS")),
    ("REST", ("MODS", "ACC")),
    ("REST", ("MODS", "TAIL")),
    ("TAIL", ("ACC", "MODS")),
    # -- modificadores (cierre recursivo por la derecha) --------------------
    ("MODS", ("MOD",)),
    ("MODS", ("MOD", "MODS")),
    ("MOD", ("SN",)),
    ("MOD", ("LOC",)),
    ("MOD", ("ORG",)),
    ("MOD", ("BENEF",)),
    ("MOD", ("PLAZO",)),
    ("MOD", ("MONTO",)),
    ("MOD", ("MISC",)),
    # -- contexto previo al disparador -------------------------------------
    ("PRE", ("SN",)),
    ("PRE", ("LOC",)),
    ("PRE", ("ORG",)),
    ("PRE", ("PLAZO",)),
    ("PRE", ("MONTO",)),
    ("PRE", ("MISC",)),
]

BINARY = [(lhs, rhs) for lhs, rhs in RULES if len(rhs) == 2]
UNARY = [(lhs, rhs[0]) for lhs, rhs in RULES if len(rhs) == 1]

# Etiqueta de slot -> etiqueta de chunk
SLOT_TO_CHUNK = {
    "DATE": "PLAZO",
    "MONEY": "MONTO",
    "QTY": "MONTO",
    "PCT": "MONTO",
    "PLACE": "LOC",
    "ORG": "ORG",
    "BENEF": "BENEF",
}

# Prioridad al asignar un chunk a una palabra (mayor gana)
CHUNK_PRIORITY = {
    "TRIG": 100, "VTRIG": 100, "PLAZO": 60, "MONTO": 55, "LOC": 50,
    "ORG": 45, "BENEF": 40, "ACC": 30, "SN": 20, "MISC": 0,
}

# Clases de disparador que NO son disparador de chunk:
#  * goal  -> es un marco temporal, se ancla como PLAZO;
#  * deontic -> sólo cuenta si abre la oración ("Hay que ..."); si no, es ruido.
GOAL_CLASSES = {"goal"}
DEONTIC_CLASS = "deontic"


@dataclass(slots=True)
class Chunk:
    label: str            # TRIG | ACC | SN | LOC | ORG | BENEF | PLAZO | MONTO | MISC
    w_start: int          # índice de palabra (0-based dentro de la oración)
    w_end: int            # inclusivo
    text: str = ""
    payload: object = None  # TriggerHit o Slot que originó el ancla
    strength: float = 0.0


def is_action_token(token) -> bool:
    """¿La palabra es una forma de verbo de acción prometible?"""
    folded = token.folded
    if folded in lexicon.ACTION_FORMS:
        return True
    if len(folded) < 4:
        return False
    return token.stem in lexicon.ACTION_STEMS


def is_content_action(token) -> bool:
    """¿La palabra es una acción de contenido (entrega verificable)?"""
    if token.folded in lexicon.CONTENT_FORMS:
        return True
    return len(token.folded) >= 4 and token.stem in lexicon.CONTENT_STEMS


def _label_for_gap(words: list, lo: int, hi: int) -> tuple[str, float]:
    """Etiqueta de un tramo de palabras sin ancla."""
    if lo > hi:
        return "MISC", 0.0
    if any(is_action_token(words[i]) for i in range(lo, hi + 1)):
        return "ACC", 1.0
    stems = [words[i].stem for i in range(lo, hi + 1)]
    if any(s in lexicon.NOUN_STEMS for s in stems):
        return "SN", 0.6
    if len(stems) >= 2 and any(
        tuple(stems[k:k + len(p)]) == p
        for p in lexicon.NOUN_PHRASES
        for k in range(len(stems) - len(p) + 1)
    ):
        return "SN", 0.6
    return "MISC", 0.0


def build_chunks(words: list, hits: list, slots: list, gaz: list) -> list[Chunk]:
    """Convierte palabras + anclas (disparadores, slots, gazetteer) en chunks.

    Se recorre la oración palabra a palabra; cada palabra puede estar cubierta por
    un ancla (disparador, slot o entrada de gazetteer) o caer en un "hueco" que se
    etiqueta después como ACC/SN/MISC.
    """
    n = len(words)
    if n == 0:
        return []

    anchor = [None] * n            # (etiqueta, prioridad, payload)
    def mark(lo, hi, label, payload, priority=None):
        prio = CHUNK_PRIORITY[label] if priority is None else priority
        for i in range(max(0, lo), min(n, hi + 1)):
            current = anchor[i]
            if current is None or prio > current[1]:
                anchor[i] = (label, prio, payload)

    def word_span(start_char, end_char):
        lo, hi = None, None
        for i, w in enumerate(words):
            if w.start >= start_char and w.end <= end_char:
                lo = i if lo is None else lo
                hi = i
        return lo, hi

    for hit in hits:
        if hit.cls in GOAL_CLASSES:
            mark(hit.start, hit.end, "PLAZO", hit)
            continue
        if hit.cls == DEONTIC_CLASS and hit.start > 1:
            continue
        label = "VTRIG" if hit.cls in {"future_1p_syn", "future_3p_syn"} else "TRIG"
        mark(hit.start, hit.end, label, hit)
    for slot in slots:
        lo, hi = word_span(slot.start, slot.end)
        if lo is None:
            continue
        label = SLOT_TO_CHUNK[slot.kind]
        # un slot explícito ("antes de diciembre") gana al disparador temporal
        mark(lo, hi, label, slot, priority=CHUNK_PRIORITY[label] + 5)
    for kind, lo, hi in gaz:      # (etiqueta, w_start, w_end) de gazetteers
        mark(lo, hi, kind, None)

    chunks: list[Chunk] = []
    i = 0
    while i < n:
        if anchor[i] is not None:
            label, _, payload = anchor[i]
            j = i
            while (j + 1 < n and anchor[j + 1] is not None
                   and anchor[j + 1][0] == label and anchor[j + 1][2] is payload):
                j += 1
            chunks.append(
                Chunk(label, i, j, " ".join(w.surface for w in words[i:j + 1]),
                      payload, getattr(payload, "strength", 0.0) or _chunk_strength(label))
            )
            i = j + 1
            continue
        j = i
        while j + 1 < n and anchor[j + 1] is None:
            j += 1
        label, strength = _label_for_gap(words, i, j)
        chunks.append(
            Chunk(label, i, j, " ".join(w.surface for w in words[i:j + 1]), None, strength)
        )
        i = j + 1

    return _merge(chunks)


def _chunk_strength(label: str) -> float:
    return 0.5 if label in {"PLAZO", "MONTO", "LOC", "ORG", "BENEF"} else 0.0


def _merge(chunks: list[Chunk]) -> list[Chunk]:
    """Fusiona chunks contiguos del mismo tipo (dos disparadores, dos fechas...)."""
    out: list[Chunk] = []
    for ch in chunks:
        if out and out[-1].label == ch.label and out[-1].w_end + 1 == ch.w_start:
            prev = out[-1]
            prev.w_end = ch.w_end
            prev.text = prev.text + " " + ch.text
            prev.strength = max(prev.strength, ch.strength)
            continue
        out.append(Chunk(ch.label, ch.w_start, ch.w_end, ch.text, ch.payload, ch.strength))
    return out


# -----------------------------------------------------------------------------
# CYK
# -----------------------------------------------------------------------------


def cyk(labels: list[str]):
    """CYK sobre la secuencia de etiquetas de chunk.

    Devuelve `(aceptada, árbol)`. Se guarda un backpointer por (símbolo, i, j), no
    por cada parse: sólo interesa si `S` cubre la oración completa y un árbol de
    ejemplo para el reporte.

    Nodos del árbol:
        ("T", etiqueta, i, j)              terminal (chunk)
        (A, hijo, i, j)                    regla unaria A -> hijo
        (A, hijo_izq, hijo_der, i, j)      regla binaria A -> B C
    """
    n = len(labels)
    if n == 0:
        return False, None

    table: list[list[set[str]]] = [[set() for _ in range(n)] for _ in range(n)]
    back: dict[tuple[str, int, int], tuple] = {}

    for i, label in enumerate(labels):
        if label in TERMINALS:
            table[i][i].add(label)

    def unary_closure(i: int, j: int) -> None:
        changed = True
        while changed:
            changed = False
            for lhs, rhs in UNARY:
                if rhs in table[i][j] and lhs not in table[i][j]:
                    table[i][j].add(lhs)
                    back[(lhs, i, j)] = ("unary", rhs)
                    changed = True

    for i in range(n):
        unary_closure(i, i)

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                left, right = table[i][k], table[k + 1][j]
                if not left or not right:
                    continue
                for lhs, (b, c) in BINARY:
                    if b in left and c in right and lhs not in table[i][j]:
                        table[i][j].add(lhs)
                        back[(lhs, i, j)] = ("binary", b, k, c)
            unary_closure(i, j)

    accepted = "S" in table[0][n - 1]
    tree = _build_tree(back, labels, "S", 0, n - 1) if accepted else None
    return accepted, tree


def _build_tree(back, labels, sym, i, j):
    if i == j and (sym, i, j) not in back:
        return ("T", labels[i], i, j)
    entry = back.get((sym, i, j))
    if entry is None:
        return ("T", labels[i], i, j)
    if entry[0] == "unary":
        return (sym, _build_tree(back, labels, entry[1], i, j), i, j)
    _, b, k, c = entry
    return (sym, _build_tree(back, labels, b, i, k),
            _build_tree(back, labels, c, k + 1, j), i, j)


def render_tree(node) -> str:
    """Árbol en texto indentado (para la libreta y el reporte)."""
    lines: list[str] = []

    def walk(nd, depth: int):
        pad = "  " * depth
        if nd[0] == "T":
            lines.append(f"{pad}{nd[1]}")
            return
        if len(nd) == 4:                       # unaria
            lines.append(f"{pad}{nd[0]}")
            walk(nd[1], depth + 1)
            return
        lines.append(f"{pad}{nd[0]}")
        walk(nd[1], depth + 1)
        walk(nd[2], depth + 1)

    walk(node, 0)
    return "\n".join(lines)


def covered_labels(tree) -> dict[str, list[tuple[int, int]]]:
    """Chunks (terminales) que el árbol usa, por etiqueta."""
    out: dict[str, list[tuple[int, int]]] = {}

    def walk(nd):
        if nd is None:
            return
        if nd[0] == "T":
            out.setdefault(nd[1], []).append((nd[2], nd[3]))
            return
        for child in nd[1:-2]:
            walk(child)

    walk(tree)
    return out


# -----------------------------------------------------------------------------
# Tipo de evento a partir de los sustantivos del marco
# -----------------------------------------------------------------------------


def event_type(chunks: list[Chunk], words: list) -> tuple[str, str]:
    """Tipo de evento dominante y la palabra que lo justifica.

    Vota cada sustantivo del lexicón que aparezca en los chunks de contenido
    (ACC/SN/MONTO/PLAZO/BENEF). Empate -> el tipo que aparece primero en el texto;
    sin votos -> "otro".
    """
    votes: dict[str, list] = {}
    order = 0
    for ch in chunks:
        if ch.label not in {"ACC", "SN", "MONTO", "PLAZO", "BENEF"}:
            continue
        for i in range(ch.w_start, ch.w_end + 1):
            etype = lexicon.NOUN_STEMS.get(words[i].stem)
            if etype is None:
                continue
            if etype not in votes:
                votes[etype] = [0, words[i].surface, order]
                order += 1
            votes[etype][0] += 1
    if not votes:
        return "otro", ""
    best = max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][2]))
    return best[0], best[1][1]
