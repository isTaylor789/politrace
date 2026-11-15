"""Extracción de slots por expresiones regulares (EI 3.2.1).

Los slots son la parte "cerrada" del marco de evento: fecha, monto, cantidad,
porcentaje, lugar y beneficiario. Se buscan sobre el **texto crudo** de la
oración (no plegado) por dos razones:

* los lugares y beneficiarios dependen de la mayúscula;
* los patrones de fecha usan nombres de meses con acento.

Cada slot guarda su span de caracteres, que el extractor traduce a índices de
token para alimentar el chunker de la CFG.

Convención de valores normalizados (`value`):

* DATE  -> "2024", "2024-12", "rel:100dias", "anio_del_discurso"
* MONEY -> importe en la moneda leída: "3,000 millones de pesos" -> 3000000000
* QTY   -> el número sin separadores: "15000"
* PCT   -> "35%"
"""

import re
from dataclasses import dataclass

from ei_scripts import lexicon

MONTHS_ALT = "|".join(lexicon.MONTHS)
MONTHS_RE = re.compile(
    rf"\b(\d{{1,2}})\s+de\s+({MONTHS_ALT})(?:\s+(?:de|del)\s+(\d{{4}}))?",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

DATE_PATTERNS = [
    ("abs", re.compile(rf"\b\d{{1,2}}\s+de\s+(?:{MONTHS_ALT})(?:\s+(?:de|del)\s+\d{{4}})?", re.I)),
    ("rel", re.compile(
        r"\b(?:en|durante|para|antes\s+de)\s+(?:el\s+|los\s+)?(?:pr[oó]xim[oa]s?|siguientes?)"
        r"\s+(\d{1,3})\s+(d[ií]as|meses|a[ñn]os|semanas)\b", re.I)),
    ("rel", re.compile(r"\b(?:primeros|cien|100)\s+d[ií]as\b", re.I)),
    ("rel", re.compile(r"\ba\s+m[aá]s\s+tardar\b", re.I)),
    ("rel", re.compile(
        r"\bantes\s+de\s+que\s+termine\s+(?:el|este)\s+(?:a[ñn]o|sexenio|periodo|gobierno)\b", re.I)),
    ("rel", re.compile(
        r"\b(?:en|durante)\s+(?:este|el\s+presente)\s+(?:a[ñn]o|sexenio|periodo|mes|gobierno)\b", re.I)),
    ("rel", re.compile(r"\b(?:el\s+)?pr[oó]ximo\s+(?:a[ñn]o|sexenio|mes)\b", re.I)),
    ("rel", re.compile(r"\bantes\s+de\s+(?:diciembre|que\s+concluya|finalizar)\b", re.I)),
    ("abs", re.compile(r"\b(?:19|20)\d{2}\b")),
]

MONEY_RE = re.compile(
    r"\b(\d[\d.,]*)\s*(mil|mill[oó]n|millones|bill[oó]n|billones)?\s*(?:de\s+)?"
    r"(pesos|d[oó]lares|euros|centavos)\b",
    re.I,
)
MONEY_ABBR_RE = re.compile(r"\b(\d[\d.,]*)\s*(mdp|mmdp|mdd|mde)\b", re.I)

PCT_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:por\s+ciento|%)\b", re.I)

QTY_NOUNS = (
    "becas|viviendas|casas|empleos|trabajos|escuelas|hospitales|cl[ií]nicas|"
    "kil[oó]metros|kms|cr[eé]ditos|pensiones|apoyos|unidades|familias|personas|"
    "ni[ñn]os|j[oó]venes|mujeres|docentes|maestros|m[eé]dicos|dosis|vacunas|"
    "[aá]rboles|hect[aá]reas|litros|barriles|mesas|acciones|c[aá]maras|"
    "viviendas|obras|puentes|carreteras|centros|polic[ií]as|elementos|plazas|"
    "toneladas|pesos|d[oó]lares|millones"
)
QTY_RE = re.compile(
    rf"\b(\d[\d.,]*)\s+(mil\s+|mill[oó]n(?:es)?\s+de\s+)?({QTY_NOUNS})\b", re.I
)

BENEF_RE = re.compile(
    r"\b(?:para|a\s+favor\s+de|en\s+beneficio\s+de)\s+"
    r"(?:todos\s+los|todas\s+las|los|las|el|la)\s+"
    r"([a-záéíóúüñ]+(?:\s+[a-záéíóúüñ]+){0,3})",
    re.I,
)

# Lugar por mayúscula tras preposición: "en Tijuana", "en el Estado de México".
# El gazetteer (vía Aho-Corasick) cubre los nombres conocidos; esto cubre el resto.
_CAP = r"[A-ZÁÉÍÓÚÑ][\wáéíóúüñÁÉÍÓÚÜÑ]+"
PLACE_CAP_RE = re.compile(
    rf"\b[Ee]n\s+((?:[Ee]l|[Ll]a|[Ll]os|[Ll]as)\s+)?"
    rf"({_CAP}(?:\s+(?:de|del|la|las|los)\s+{_CAP}|\s+{_CAP})?)"
)

@dataclass(slots=True)
class Slot:
    kind: str      # DATE | MONEY | QTY | PCT | PLACE | BENEF
    start: int     # offset de carácter absoluto dentro del documento
    end: int
    surface: str   # texto tal cual
    value: str     # valor normalizado
    evidence: str = ""   # nombre del patrón que lo encontró (para auditoría)


def _abs_month(text: str, day: str, month: str, year: str | None) -> str:
    from ei_scripts import lexicon as lx

    m = lx.MONTHS.index(month.lower()) + 1 if month.lower() in lx.MONTHS else None
    if m is None:
        return ""
    return f"{year}-{m:02d}" if year else f"??-{m:02d}"


def _parse_number(number: str) -> float | None:
    """Número con separadores de miles (\"1.290.000\", \"3,000\") o decimal (\"3.5\")."""
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", number):
        return float(re.sub(r"[.,]", "", number))
    try:
        return float(number.replace(",", "."))
    except ValueError:
        return None


def _money_value(number: str, mult: str | None) -> str:
    val = _parse_number(number)
    if val is None:
        return ""
    mult = (mult or "").lower()
    if mult in {"mil"}:
        val *= 1e3
    elif mult in {"millon", "millón", "millones"}:
        val *= 1e6
    elif mult in {"billon", "billón", "billones"}:
        val *= 1e12
    # el valor se publica en la unidad monetaria tal como se leyó (pesos/dólares)
    return f"{val:.0f}" if abs(val) >= 1 else f"{val:g}"


def _qty_value(number: str, mult: str | None) -> str:
    """Cantidad como entero ("15 mil becas" -> 15000)."""
    val = _parse_number(number)
    if val is None:
        return ""
    num = int(val)
    mult = (mult or "").lower()
    if mult.startswith("mil"):
        num *= 1000
    elif mult.startswith(("millon", "millón")):
        num *= 1_000_000
    return str(num)


def extract_slots(sentence) -> list[Slot]:
    """Todos los slots de una oración (`ei_scripts.segment.Sentence`)."""
    text = sentence.text
    base = sentence.start
    found: list[Slot] = []

    def add(kind, m, value, group=0, evidence=""):
        start, end = m.start(group), m.end(group)
        found.append(
            Slot(kind, base + start, base + end, text[start:end], value, evidence)
        )

    for m in MONTHS_RE.finditer(text):
        add("DATE", m, _abs_month(text, m.group(1), m.group(2), m.group(3)), 0, "mes_dia")
    for name, pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            if name == "rel":
                value = "rel:" + m.group(0)
            else:
                value = m.group(0)
            add("DATE", m, str(value), 0, name)
    for m in MONEY_RE.finditer(text):
        add("MONEY", m, _money_value(m.group(1), m.group(2)), 0, "moneda")
    for m in MONEY_ABBR_RE.finditer(text):
        add("MONEY", m, m.group(1) + " " + m.group(2).upper(), 0, "moneda_abbr")
    for m in PCT_RE.finditer(text):
        add("PCT", m, m.group(1) + "%", 0, "porcentaje")
    for m in QTY_RE.finditer(text):
        add("QTY", m, _qty_value(m.group(1), m.group(2)), 0, "cantidad")
    for m in BENEF_RE.finditer(text):
        add("BENEF", m, m.group(1).strip(), 0, "beneficiario")
    for m in PLACE_CAP_RE.finditer(text):
        surface = m.group(0)
        if len(surface) < 7:            # "en El" no es un lugar
            continue
        add("PLACE", m, m.group(2).strip(), 0, "lugar_mayuscula")

    return _resolve_overlaps(found)


def _resolve_overlaps(slots: list[Slot]) -> list[Slot]:
    """Un span sólo puede tener un slot; gana el de mayor prioridad.

    Prioridad: DATE > MONEY > PCT > QTY > PLACE > BENEF (de más específico a más
    laxo, para que "mil millones de pesos" no quede como dos slots).
    """
    priority = {"DATE": 0, "MONEY": 1, "PCT": 2, "QTY": 3, "PLACE": 4, "BENEF": 5}
    slots = sorted(slots, key=lambda s: (priority[s.kind], -(s.end - s.start)))
    kept: list[Slot] = []
    for slot in slots:
        if any(not (slot.end <= k.start or slot.start >= k.end) for k in kept):
            continue
        kept.append(slot)
    kept.sort(key=lambda s: s.start)
    return kept


def slots_by_kind(slots: list[Slot], kind: str) -> list[Slot]:
    return [s for s in slots if s.kind == kind]
