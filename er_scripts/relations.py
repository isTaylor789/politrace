"""Extracción de relaciones (EI 3.5).

Dos estrategias complementarias, ambas simbólicas:

1. **Coocurrencia en ventana** (`config.COOC_WINDOW` tokens): toda pareja de
   menciones de entidades distintas que aparecen cerca genera una relación
   `coocurre`. Es la estrategia de *recall*: barata, sin sesgo de vocabulario,
   pero no dice **qué** relación hay.
2. **Patrones verbales**: se busca un verbo relacional del léxico
   (`inaugurar`, `entregar`, `visitar`, `acusar`, `nombrar`, …) y se toma la
   entidad más cercana a la izquierda (sujeto) y a la derecha (objeto) dentro de la
   misma oración. Da el **tipo** de relación y es la parte de precisión.

Además, la relación `ocupa_cargo` sale de una mención de cargo resuelta a su
ocupante junto a una organización o lugar.

Las ocurrencias no se guardan una por una: se agregan por arista
`(a, b, tipo)` con peso = número de ocurrencias y hasta `MAX_EVIDENCE` oraciones de
evidencia. Eso mantiene `relations.jsonl` en tamaño manejable.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

from ri_scripts.preprocess import fold

from er_scripts import config, lexicon
from er_scripts.mentions import Mention, node_type, words_of


def _verb_forms() -> dict[str, str]:
    """Formas verbales de 3ª persona y 1ª plural -> tipo de relación.

    Se construyen por regla morfológica en vez de usar el stem, porque el stem
    confunde el sustantivo con el verbo: "el apoyo" y "apoyó" comparten stem.
    """
    forms: dict[str, str] = {}
    for lemma, rtype in config.RELATIONAL_LEMMAS.items():
        base = fold(lemma)
        if base == "reunirse":
            for f in ("reune", "reunen", "reunio", "reunieron", "reunirse"):
                forms[f] = rtype
            continue
        if base.endswith("ar"):
            stem_ = base[:-2]
            suffixes = ("a", "an", "o_", "aron", "aba", "aban", "ando", "amos")
        elif base.endswith(("er", "ir")):
            stem_ = base[:-2]
            suffixes = ("e", "en", "io", "ieron", "ia", "ian", "iendo", "emos", "imos")
        else:
            continue
        for suf in suffixes:
            if suf == "o_":          # 1ª persona del singular: se descarta (ruido)
                continue
            forms[stem_ + suf] = rtype
        # irregulares frecuentes del pretérito
        if base == "ir":
            continue
    # formas irregulares que no siguen la regla
    forms.update({
        "firmo": "firma_con", "firmaron": "firma_con",
        "reunio": "se_reune_con", "reunieron": "se_reune_con",
        "pidio": "pide_a", "pidieron": "pide_a",
        "exigio": "exige_a", "exigieron": "exige_a",
        "goberno": "gobierna", "gobernaron": "gobierna",
        "recibio": "recibe", "recibieron": "recibe",
        "entregaron": "entrega", "inauguraron": "inaugura",
    })
    return forms


VERB_FORMS = _verb_forms()


@dataclass(slots=True)
class Occurrence:
    """Una relación observada en una oración concreta."""

    a: str
    b: str
    type: str
    sentence: str
    doc_id: int | None
    sent_idx: int
    subject_type: str = ""


def _word_index(mentions: list[Mention], words) -> list[tuple[int, int, Mention]]:
    """Traduce spans de carácter a índices de palabra: (ini, fin, mención)."""
    out = []
    for m in mentions:
        lo = next((i for i, w in enumerate(words) if w.start >= m.start), None)
        hi = next((i for i in range(len(words) - 1, -1, -1) if words[i].end <= m.end), None)
        if lo is None or hi is None or hi < lo:
            continue
        out.append((lo, hi, m))
    out.sort(key=lambda t: t[0])
    return out


def _node(mention: Mention) -> str:
    return mention.resolved or mention.canonical


def _unique_mentions(mentions: list[Mention]) -> list[Mention]:
    """Una mención por nodo y oración (el mismo nodo puede llegar por varias vías)."""
    seen: dict[str, Mention] = {}
    for m in mentions:
        node = _node(m)
        best = seen.get(node)
        if best is None or (m.end - m.start) > (best.end - best.start):
            seen[node] = m
    return sorted(seen.values(), key=lambda m: m.start)


def cooccurrences(sentence, mentions: list[Mention]) -> list[Occurrence]:
    """Parejas de entidades distintas dentro de la ventana de tokens."""
    words = words_of(sentence)
    mentions = _unique_mentions(mentions)
    if len(mentions) < 2:
        return []
    spans = _word_index(mentions, words)[: config.COOC_MAX_ENTITIES]
    out: list[Occurrence] = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            lo_i, hi_i, mi = spans[i]
            lo_j, hi_j, mj = spans[j]
            gap = lo_j - hi_i
            if gap > config.COOC_WINDOW:
                continue
            a, b = _node(mi), _node(mj)
            if a == b:
                continue
            out.append(Occurrence(a, b, "coocurre", sentence.text, mi.doc_id,
                                  sentence.idx, f"{node_type(a)}-{node_type(b)}"))
    return out


def verbal_relations(sentence, mentions: list[Mention]) -> list[Occurrence]:
    """Patrón sujeto–verbo–objeto con verbos relacionales del léxico."""
    words = words_of(sentence)
    mentions = _unique_mentions(mentions)
    if len(mentions) < 2:
        return []
    spans = _word_index(mentions, words)
    if not spans:
        return []
    out: list[Occurrence] = []
    seen: set[tuple[str, str, str]] = set()

    for idx, tok in enumerate(words):
        rel_type = VERB_FORMS.get(tok.folded)
        if rel_type is None:
            continue
        left = None
        for lo, hi, m in spans:
            if hi < idx:
                left = m           # el más cercano a la izquierda gana (spans ordenados)
            elif lo > idx:
                break
        if left is None:
            continue
        right = next((m for lo, hi, m in spans if lo > idx), None)
        if right is None:
            continue
        a, b = _node(left), _node(right)
        if a == b or (a, b, rel_type) in seen:
            continue
        seen.add((a, b, rel_type))
        out.append(Occurrence(a, b, rel_type, sentence.text, left.doc_id,
                              sentence.idx, f"{node_type(a)}-{node_type(b)}"))
    return out


def cargo_relations(sentence, mentions: list[Mention], country: str) -> list[Occurrence]:
    """'secretario de Hacienda' resuelto a persona -> relación con la organización.

    La organización tiene que estar **dentro** del span del cargo ("director del
    IMSS"): así la relación es "X ocupa_cargo IMSS" y no "X cerca de cualquier
    lugar mencionado en la oración".
    """
    out: list[Occurrence] = []
    for m in mentions:
        if m.type != "CARGO" or not m.resolved or not m.resolved.startswith("PER:"):
            continue
        inside = [o for o in mentions
                  if o is not m and o.type == "ORG" and o.start >= m.start and o.end <= m.end]
        rtype = "ocupa_cargo" if m.source == "cargo_explicito" else "cargo_inferido"
        for near in inside:
            out.append(Occurrence(m.resolved, _node(near), rtype, sentence.text,
                                  m.doc_id, sentence.idx, "PER-ORG"))
    return out


def relations_of(sentence, mentions: list[Mention], meta: dict) -> list[Occurrence]:
    """Todas las relaciones de una oración (coocurrencia + verbales + cargo)."""
    if len(mentions) < 2:
        return []
    out = cooccurrences(sentence, mentions)
    out += verbal_relations(sentence, mentions)
    out += cargo_relations(sentence, mentions, meta.get("country") or "")
    return out


# -----------------------------------------------------------------------------
# Agregación
# -----------------------------------------------------------------------------
class RelationAggregator:
    """Acumula ocurrencias en aristas ponderadas con evidencia."""

    def __init__(self, max_evidence: int = config.MAX_EVIDENCE):
        self.weights: Counter = Counter()          # (a, b, tipo) -> n
        self.evidence: dict[tuple, list] = defaultdict(list)
        self.docs: dict[tuple, set] = defaultdict(set)
        self.max_evidence = max_evidence
        self.total = 0

    def add(self, occ: Occurrence) -> None:
        key = (occ.a, occ.b, occ.type)
        self.weights[key] += 1
        self.total += 1
        if occ.doc_id is not None:
            self.docs[key].add(occ.doc_id)
        if len(self.evidence[key]) < self.max_evidence:
            self.evidence[key].append({
                "doc_id": occ.doc_id, "sent_idx": occ.sent_idx,
                "sentence": occ.sentence.strip()[:300],
            })

    def add_all(self, occurrences) -> None:
        for occ in occurrences:
            self.add(occ)

    def edges(self, min_weight: int = config.MIN_RELATION_WEIGHT) -> list[dict]:
        out = []
        for (a, b, rtype), w in self.weights.items():
            if w < min_weight:
                continue
            out.append({
                "source": a, "target": b, "type": rtype, "weight": w,
                "docs": len(self.docs[(a, b, rtype)]),
                "evidence": self.evidence[(a, b, rtype)],
            })
        out.sort(key=lambda e: -e["weight"])
        return out

    def type_counts(self) -> dict:
        return dict(Counter(t for (_, _, t) in self.weights).most_common())
