"""Detección de menciones de entidad (EI 3.3: reconocimiento e identificación).

Tres capas, en este orden:

1. **Gazetteer + autómata** (Aho-Corasick reutilizado de `ei_scripts.automaton`):
   personas, organizaciones, lugares y cargos multi-palabra. Es la capa de alta
   precisión: lo que está en la lista se detecta con su forma canónica.
2. **Patrones sobre mayúsculas** (regex/token runs): secuencias de palabras
   capitalizadas que no están en el gazetteer — sirven para nombres de terceros
   países, ministros nuevos, empresas, etc.
3. **Siglas**: token todo en mayúsculas de 2 a 6 letras; se enlaza con su
   expansión cuando aparece en el mismo documento.

Cada mención guarda su **span de caracteres absoluto** en el discurso, su tipo
(`PER`, `ORG`, `LOC`, `CARGO`) y su identificador canónico de nodo
(`PER:Nombre Canónico`). La resolución de pronombres y de cargos a su ocupante se
hace después, en `coref.py`, para que la detección quede auditable por separado.
"""

import re
from dataclasses import dataclass

from ei_scripts.automaton import AhoCorasick
from ri_scripts.preprocess import fold

from er_scripts import config, lexicon

# -----------------------------------------------------------------------------
# Tipos
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class Mention:
    doc_id: int | None
    sent_idx: int
    start: int          # offset de carácter absoluto en el discurso
    end: int
    surface: str
    type: str           # PER | ORG | LOC | CARGO
    canonical: str      # identificador de nodo: "PER:...", "ORG:...", "LOC:...", "CARGO:..."
    source: str         # gazetteer | nombre | sigla | cargo_de | coref
    resolved: str | None = None   # nodo final tras coreferencia (por defecto = canonical)

    @property
    def key(self) -> tuple[int, int]:
        return (self.start, self.end)

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "sent_idx": self.sent_idx,
            "start": self.start, "end": self.end, "surface": self.surface,
            "type": self.type, "canonical": self.canonical, "source": self.source,
            "resolved": self.resolved or self.canonical,
        }


# -----------------------------------------------------------------------------
# Autómata de gazetteers
# -----------------------------------------------------------------------------
_AC: AhoCorasick | None = None

TYPE_PRIORITY = {"PER": 4, "ORG": 3, "LOC": 2, "CARGO": 1}


def automaton() -> AhoCorasick:
    """Autómata único con personas + organizaciones + lugares (los cargos van por patrón)."""
    global _AC
    if _AC is not None:
        return _AC
    ac = AhoCorasick()
    for seq, canon in lexicon.PERSONS.items():
        ac.add_pattern(seq, tag=("PER", canon))
    for seq, canon in lexicon.ORGS.items():
        ac.add_pattern(seq, tag=("ORG", canon))
    for seq, canon in lexicon.PLACES.items():
        ac.add_pattern(seq, tag=("LOC", canon))
    _AC = ac.build()
    return _AC


# -----------------------------------------------------------------------------
# Patrones
# -----------------------------------------------------------------------------
_HONORIFICS = {
    "presidente", "presidenta", "vicepresidente", "vicepresidenta", "doctor",
    "doctora", "licenciado", "licenciada", "ingeniero", "ingeniera", "general",
    "almirante", "secretario", "secretaria", "ministro", "ministra", "senador",
    "senadora", "diputado", "diputada", "gobernador", "gobernadora", "canciller",
    "fiscal", "embajador", "embajadora", "director", "directora", "rector",
    "rectora", "juez", "jueza", "magistrado", "comandante", "delegado",
    "subsecretario", "subsecretaria", "coordinador", "coordinadora", "papa",
}

_ACRONYM_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9\-]{1,11}$")

# "Secretaría de la Defensa", "Instituto Nacional de Migración": la cabecera en
# mayúscula sostiene la lectura de organización aunque dentro haya minúsculas.
_CAP_W = r"[A-ZÁÉÍÓÚÑ][\wáéíóúüñÁÉÍÓÚÜÑ]*"
_ORG_HEAD_PATTERN = (
    r"(?P<org>\b(?:"
    + "|".join(sorted(config.ORG_HEADS_DISPLAY, key=len, reverse=True))
    # el complemento es obligatorio: "Secretaría de la Defensa", no "Secretaría de"
    + r")(?:\s+(?:de\s+la|de\s+las|de\s+los|del|de|para\s+el|para\s+la|nacional|"
    r"general|federal|mexicano|mexicana))?\s+"
    + _CAP_W
    + r"(?:\s+(?:de\s+los|de\s+las|de\s+la|del|de|la|las|los|y|e)?\s*" + _CAP_W + r"){0,3})"
)
_ORG_RE = re.compile(_ORG_HEAD_PATTERN)


def _is_cap(token) -> bool:
    return bool(token.surface) and token.surface[0].isupper()


def _name_runs(words) -> list[list[int]]:
    """Rachas de palabras capitalizadas, admitiendo conectores ('de', 'del', 'la')."""
    runs: list[list[int]] = []
    i = 0
    n = len(words)
    while i < n:
        if not _is_cap(words[i]):
            i += 1
            continue
        run = [i]
        j = i
        while j + 1 < n:
            nxt = words[j + 1]
            if _is_cap(nxt):
                j += 1
                run.append(j)
            elif (nxt.folded in config.NAME_CONNECTORS and j + 2 < n
                  and _is_cap(words[j + 2])):
                run.extend([j + 1, j + 2])
                j += 2
            else:
                break
        runs.append(run)
        i = j + 1
    return runs


def _classify_run(words, run) -> tuple[str, str] | None:
    """Tipo y canónico de una racha de palabras capitalizadas (o None si no es nombre)."""
    surfaces = [words[i].surface for i in run]
    folded = [words[i].folded for i in run]
    if any(f in config.NOT_A_NAME_START for f in folded):
        return None
    if len(run) > config.NAME_MAX_TOKENS:
        return None

    # Sigla / acrónimo (salvo que el token sea un topónimo escrito en mayúsculas)
    if len(run) == 1 and _ACRONYM_RE.match(surfaces[0]):
        if tuple(folded) in lexicon.PLACES:
            return None
        sigla = surfaces[0].upper()
        if sigla in lexicon.SIGLAS or (len(sigla) >= 2 and not sigla.islower()):
            return "ORG", sigla           # sigla conocida o acrónimo nuevo

    # Organización por palabra cabecera ("Instituto…", "Secretaría…")
    stems = {words[i].stem for i in run}
    if len(run) >= 2 and (stems & config.ORG_SUFFIX_HINTS or set(folded) & config.ORG_SUFFIX_HINTS):
        return "ORG", " ".join(surfaces)

    # Nombre de persona: al menos dos tokens no conectores
    content = [f for f in folded if f not in config.NAME_CONNECTORS]
    if len(content) == 1 and content[0] in config.GENERIC_ADJECTIVES:
        return None
    # "Secretario General", "Fiscal Nacional": cargos, no personas con nombre
    if any(f in _HONORIFICS for f in content):
        return None
    if " ".join(folded) in config.PERSON_BLOCKLIST:
        return None
    if folded and folded[0] in config.NOT_A_PERSON_HEADS:
        return None
    if len(content) >= config.NAME_MIN_TOKENS:
        return "PER", _clean_name(" ".join(surfaces))

    return None


def _is_marker(surface: str) -> bool:
    """Cabeceras de transcripción ('~~~PREGUNTA', 'INTERVENCIÓN') no son entidades."""
    if "~" in surface:
        return True
    first = fold(surface.split()[0]) if surface.split() else ""
    return first in config.TRANSCRIPT_MARKERS and len(surface.split()) == 1


def _overlaps_any(start: int, end: int, mentions: list[Mention]) -> bool:
    return any(not (end <= m.start or start >= m.end) for m in mentions)


def _clean_name(surface: str) -> str:
    """Quita honoríficos y cargos pegados al inicio del nombre."""
    parts = surface.split()
    while parts and fold(parts[0]) in _HONORIFICS:
        parts.pop(0)
    return " ".join(parts) or surface


# -----------------------------------------------------------------------------
# Detección
# -----------------------------------------------------------------------------


class EntityDetector:
    """Detector de menciones de una oración."""

    def __init__(self, use_patterns: bool = True, use_acronyms: bool = True):
        self.ac = automaton()
        self.use_patterns = use_patterns
        self.use_acronyms = use_acronyms

    # -- capa 1b: cabecera de organización ------------------------------------
    def _org_phrases(self, sentence, taken: list[Mention]) -> list[Mention]:
        """Organizaciones por cabecera ('Secretaría de la Defensa')."""
        out: list[Mention] = []
        for m in _ORG_RE.finditer(sentence.text):
            surface = m.group("org").strip().rstrip(",.;:")
            if len(surface) < 8 or "~" in surface:
                continue
            start = sentence.start + m.start("org")
            end = start + len(surface)
            if _overlaps_any(start, end, taken) or _overlaps_any(start, end, out):
                continue
            out.append(Mention(None, -1, start, end, surface, "ORG", "ORG:" + surface,
                               "nombre"))
        return out

    # -- capa 1: gazetteer ----------------------------------------------------
    def _gazetteer(self, words) -> list[Mention]:
        folded = [w.folded for w in words]
        matches = self.ac.scan(folded)
        out: list[Mention] = []
        occupied: set[int] = set()
        matches.sort(key=lambda m: (-m.length, -TYPE_PRIORITY.get(self.ac.tags[m.pid][0], 0), m.start))
        for m in matches:
            span = set(range(m.start, m.end + 1))
            if span & occupied:
                continue
            etype, canon = self.ac.tags[m.pid]
            surface = " ".join(w.surface for w in words[m.start:m.end + 1])
            if _is_marker(surface):
                continue
            # "…Rodríguez Vallejo": un apellido suelto no es una mención propia si
            # está pegado a otra palabra capitalizada (sería parte de un nombre).
            if etype == "PER" and m.length == 1:
                prev_cap = m.start > 0 and _is_cap(words[m.start - 1])
                next_cap = m.end + 1 < len(words) and _is_cap(words[m.end + 1])
                if prev_cap or next_cap:
                    continue
            if etype == "PER" and config.REQUIRE_CAPITALIZED_PERSON:
                if not lexicon.is_person_surface_capitalized(words[m.start].surface):
                    continue
            if etype == "ORG" and config.REQUIRE_CAPITALIZED_ORG:
                if not lexicon.is_person_surface_capitalized(surface):
                    continue
            occupied |= span
            if etype == "PER":
                node = lexicon.person_id(canon)
            elif etype == "LOC":
                seq = tuple(w.folded for w in words[m.start:m.end + 1])
                node = "LOC:" + (lexicon.PLACE_DISPLAY.get(seq)
                                 or lexicon.PLACES.get(seq) or surface)
            else:
                node = "ORG:" + canon
            out.append(Mention(None, -1, words[m.start].start, words[m.end].end,
                               surface, etype, node, "gazetteer"))
        return out

    # -- capa 2: patrones -----------------------------------------------------
    def _patterns(self, words, taken: list[Mention]) -> list[Mention]:
        if not self.use_patterns:
            return []
        covered = set()
        for m in taken:
            covered.update(range(m.start, m.end + 1))
        out: list[Mention] = []

        for run in _name_runs(words):
            while run and words[run[0]].folded in config.NAME_CONNECTORS:
                run = run[1:]
            if len(run) < 1:
                continue
            first, last = run[0], run[-1]
            if words[first].start in covered or words[last].end in covered:
                continue
            guess = _classify_run(words, run)
            if guess is None:
                continue
            etype, canon = guess
            surface = " ".join(w.surface for w in words[first:last + 1])
            if _is_marker(surface):
                continue
            source = "sigla" if (etype == "ORG" and len(run) == 1 and _ACRONYM_RE.match(words[first].surface)) else "nombre"
            if etype == "PER":
                canon = lexicon.person_id(canon)
            else:
                canon = "ORG:" + canon

            # Un cargo seguido de nombre propio refuerza la lectura de persona
            if first > 0 and words[first - 1].folded in _HONORIFICS and etype != "PER":
                etype, canon, source = "PER", lexicon.person_id(surface), "nombre"
            out.append(Mention(None, -1, words[first].start, words[last].end,
                               surface, etype, canon, source))
        return out

    # -- capa 3: cargo + de + entidad ----------------------------------------
    def _cargo_patterns(self, words, taken: list[Mention]) -> list[Mention]:
        """'el secretario de Hacienda' -> mención CARGO con su span completo.

        Sólo se emite cuando el cargo va seguido de la entidad sobre la que manda
        ('de Hacienda', 'del IMSS'). El span se corta antes de cualquier nombre
        propio ya detectado, para no tragarse a la persona ("el director del IMSS,
        Zoé Robledo" produce el cargo 'director del IMSS' y la PER 'Zoé Robledo').
        """
        out: list[Mention] = []
        i = 0
        while i < len(words):
            f = words[i].folded
            if (f,) not in lexicon.CARGOS:
                i += 1
                continue
            canon, _ = lexicon.CARGOS[(f,)]
            end = i
            if i + 1 < len(words) and (f, words[i + 1].folded) in lexicon.CARGOS:
                canon, _ = lexicon.CARGOS[(f, words[i + 1].folded)]
                end = i + 1
            prep = end + 1
            if prep >= len(words) or words[prep].folded not in {"de", "del", "en"}:
                i = end + 1
                continue

            j = prep + 1
            if j < len(words) and words[j].folded in {"la", "las", "los", "el"}:
                j += 1
            last = prep
            while j < len(words) and len(range(prep, j + 1)) <= 4:
                tok = words[j]
                if tok.surface[:1].isupper():
                    # una persona ya detectada no se traga dentro del cargo
                    if any(not (tok.end <= m.start or words[i].start >= m.end)
                           for m in taken if m.type == "PER"):
                        break
                    last = j
                    j += 1
                elif tok.folded in config.NAME_CONNECTORS and j + 1 < len(words) \
                        and words[j + 1].surface[:1].isupper():
                    j += 1
                else:
                    break
            if last == prep:
                i = end + 1
                continue
            out.append(Mention(None, -1, words[i].start, words[last].end,
                               " ".join(w.surface for w in words[i:last + 1]),
                               "CARGO", "CARGO:" + canon, "cargo_de"))
            i = last + 1
        return out

    # -- API ------------------------------------------------------------------
    def detect(self, sentence, doc_id: int | None = None) -> list[Mention]:
        """Todas las menciones no solapadas de una oración."""
        words = [t for t in sentence.tokens if t.is_word]
        if not words:
            return []
        mentions = self._gazetteer(words)
        mentions += self._org_phrases(sentence, mentions)
        mentions += self._patterns(words, mentions)
        mentions += self._cargo_patterns(words, mentions)

        # resolver solapes: gana el span más largo, luego el tipo de mayor prioridad.
        # Excepción: un CARGO puede contener una ORG/LOC ("director del IMSS"), porque
        # el organigrama es justamente la relación que interesa extraer.
        mentions.sort(key=lambda m: (-(m.end - m.start), -TYPE_PRIORITY[m.type], m.start))
        kept: list[Mention] = []
        for m in mentions:
            conflicts = [k for k in kept if not (m.end <= k.start or m.start >= k.end)]
            if conflicts:
                nested = (
                    (m.type == "CARGO" and all(k.type in {"ORG", "LOC"} for k in conflicts))
                    or (m.type in {"ORG", "LOC"} and all(k.type == "CARGO" for k in conflicts))
                )
                if not nested:
                    continue
            kept.append(m)
        for m in kept:
            m.doc_id = doc_id
            m.sent_idx = sentence.idx
            m.resolved = m.canonical
        kept.sort(key=lambda m: m.start)
        return kept


# -----------------------------------------------------------------------------
# Utilidades para el módulo de relaciones
# -----------------------------------------------------------------------------
def words_of(sentence) -> list:
    return [t for t in sentence.tokens if t.is_word]


def surface_of(sentence, mention: Mention) -> str:
    return sentence.text[mention.start - sentence.start:mention.end - sentence.start]


def canonical_label(node_id: str) -> str:
    """'PER:Javier Milei' -> 'Javier Milei'."""
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


def node_type(node_id: str) -> str:
    return node_id.split(":", 1)[0] if ":" in node_id else "OTRO"
