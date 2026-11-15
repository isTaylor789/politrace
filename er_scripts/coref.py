"""Coreferencia por reglas (EI 3.3.2: identificación y clasificación de nombres).

No hay modelo: son cuatro reglas explícitas, cada una con su justificación
lingüística y su límite.

1. **Hablante** — en un discurso, "el presidente", "el mandatario", "el Ejecutivo",
   "mi gobierno", "este gobierno" remiten al hablante o al gobierno de su país.
   Es la regla de mayor rendimiento: en un corpus de discursos presidenciales, la
   mayor parte de las menciones elípticas son autorreferencias.
2. **Cargo → ocupante** — "el secretario de Hacienda" se resuelve a la persona que
   ocupa ese cargo según `lexicon.CARGO_HOLDERS` (tabla estática por país).
3. **Sigla ↔ expansión** — "Instituto Mexicano del Seguro Social" e "IMSS" se
   fusionan en el mismo nodo usando `lexicon.EXPANSION_TO_SIGLA`.
4. **Elipsis con cargo + nombre** — "el presidente Boric": el detector ya toma
   "Boric" como PER; la regla 1 no debe pisarlo (se comprueba solapamiento).

Lo que **no** se hace (y se reporta como limitación): resolución de pronombres
personales ("él", "ella") a antecedentes lejanos, y desambiguación de homónimos.
"""

import re

from er_scripts import lexicon
from er_scripts.mentions import Mention, canonical_label, node_type

# Sólo con artículo/posesivo de 1ª persona: "su presidente" habla de otro país y
# "la presidenta" no puede ser el hablante (los cinco mandatarios del corpus son
# hombres; en México "la presidenta municipal" es otra persona).
_SPEAKER_RE = re.compile(
    r"\b(?:el|este|mi|nuestro)\s+"
    r"(?:presidente|mandatario|ejecutivo|jefe del ejecutivo)\b",
    re.IGNORECASE,
)
_GOV_RE = re.compile(
    r"\b(?:el|este|mi|nuestro|nuestra|del|al)\s+gobierno"
    r"(?:\s+(?:federal|nacional|de la rep[uú]blica|espa[ñn]ol|mexicano|argentino|"
    r"chileno|venezolano))?\b",
    re.IGNORECASE,
)
_GOV_OF_RE = re.compile(r"\bgobierno\s+(?:de|del)\s+([A-ZÁÉÍÓÚÑ][\wáéíóúüñ]+)", re.IGNORECASE)
_ADJACENT_NAME = re.compile(r"^\s*(?:de\s+|del\s+)?[A-ZÁÉÍÓÚÑ]")


def _overlaps(start: int, end: int, mentions: list[Mention]) -> bool:
    return any(not (end <= m.start or start >= m.end) for m in mentions)


def resolve(sentence, mentions: list[Mention], meta: dict) -> list[Mention]:
    """Asigna `resolved` a cada mención y agrega las menciones elípticas."""
    country = meta.get("country") or ""
    speaker = meta.get("speaker") or ""
    gov_node = "ORG:" + lexicon.COUNTRY_GOVERNMENT.get(country, "Gobierno")
    speaker_node = lexicon.person_id(speaker) if speaker else None
    text = sentence.text
    base = sentence.start

    # --- regla 1: hablante ---------------------------------------------------
    if speaker_node:
        for m in _SPEAKER_RE.finditer(text):
            start, end = base + m.start(), base + m.end()
            if _overlaps(start, end, mentions):
                continue        # "el presidente Boric": manda el nombre propio
            if _ADJACENT_NAME.match(text[m.end():m.end() + 40]):
                continue
            # "el presidente municipal" / "el presidente del Senado" no son el hablante
            if re.match(r"\s+(?:municipal|del\s+senado|de\s+la\s+c[aá]mara|de\s+la\s+rep[uú]blica\s+de)",
                        text[m.end():m.end() + 40], re.IGNORECASE):
                continue
            mentions.append(Mention(
                meta.get("doc_id"), sentence.idx, start, end, m.group(0),
                "PER", speaker_node, "coref", speaker_node,
            ))

    # --- regla 1b: gobierno (con país explícito primero) ---------------------
    for m in _GOV_OF_RE.finditer(text):
        start, end = base + m.start(), base + m.end()
        if _overlaps(start, end, mentions):
            continue
        place = canonical_label(m.group(1))
        mentions.append(Mention(
            meta.get("doc_id"), sentence.idx, start, end, m.group(0),
            "ORG", "ORG:Gobierno de " + place, "coref", "ORG:Gobierno de " + place,
        ))
    for m in _GOV_RE.finditer(text):
        start, end = base + m.start(), base + m.end()
        if _overlaps(start, end, mentions):
            continue
        # "el gobierno francés" no es el gobierno del hablante
        if re.match(r"\s+(?:franc[eé]s|alem[aá]n|italiano|brit[aá]nico|estadounidense|"
                    r"argentino|chileno|venezolano|mexicano|ruso|ucraniano|chino|"
                    r"brasile[ñn]o|colombiano|espa[ñn]ol|de\s+otro)",
                    text[m.end():m.end() + 30], re.IGNORECASE):
            continue
        mentions.append(Mention(
            meta.get("doc_id"), sentence.idx, start, end, m.group(0),
            "ORG", gov_node, "coref", gov_node,
        ))

    # --- regla 3: siglas <-> expansión --------------------------------------
    for m in mentions:
        if m.type != "ORG":
            continue
        label = canonical_label(m.resolved or m.canonical)
        sigla = lexicon.EXPANSION_TO_SIGLA.get(tuple(label.lower().split()))
        if sigla:
            m.resolved = "ORG:" + sigla        # forma larga -> sigla
        elif label.upper() in lexicon.SIGLAS:
            m.resolved = "ORG:" + label.upper()  # sigla ya normalizada

    # --- regla 2: cargo -> ocupante -----------------------------------------
    # Se distingue el caso explícito ("el secretario de Hacienda, Rogelio Ramírez")
    # del inferido (sólo aparece el cargo y el ocupante sale de la tabla estática).
    for m in mentions:
        if m.type != "CARGO":
            continue
        holder = _holder_for(sentence, m, country, mentions)
        if not holder:
            continue
        m.resolved = lexicon.person_id(holder)
        m.source = "cargo_explicito" if _holder_named(holder, m, mentions) else "cargo_inferido"

    mentions.sort(key=lambda m: m.start)
    return mentions


def _holder_named(holder: str, cargo: Mention, mentions: list[Mention]) -> bool:
    """¿El nombre del ocupante aparece junto al cargo en esta oración?"""
    node = lexicon.person_id(holder)
    return any(m.type == "PER" and (m.resolved or m.canonical) == node
               and abs(m.start - cargo.end) < 80 for m in mentions)


def _holder_for(sentence, mention: Mention, country: str, mentions: list[Mention]) -> str | None:
    """Busca el ocupante de un cargo usando el texto que sigue ('de Hacienda')."""
    if not country:
        return None
    tail_start = mention.end - sentence.start
    tail = sentence.text[tail_start:tail_start + 60]
    candidates = [mention.surface]
    # "secretario de Hacienda" / "secretario de la Defensa Nacional"
    m = re.match(r"\s+(?:de|del|de la)\s+([\wáéíóúüñÁÉÍÓÚÜÑ\s]{3,40})", tail)
    if m:
        following = m.group(1).strip()
        # recortar en la primera preposición o verbo probable
        following = re.split(r"\s+(?:y|para|con|en|que|de|del)\s+", following)[0]
        candidates.insert(0, f"{mention.surface} de {following}")
    for cand in candidates:
        holder = lexicon.cargo_holder(cand, country)
        if holder:
            return holder
    return None


def coreferent_nodes(mention: Mention) -> set[str]:
    """Nodos que representa una mención (canónico y resuelto) para estadísticas."""
    out = {mention.canonical}
    if mention.resolved:
        out.add(mention.resolved)
    return out


def is_elliptical(mention: Mention) -> bool:
    return mention.source == "coref" and node_type(mention.canonical) in {"PER", "ORG"}
