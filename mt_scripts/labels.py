"""Etiquetado legible de tópicos: stems -> formas reales + reglas manuales.

Dos capas, y conviene no confundirlas:

1. **Presentación**: los modelos devuelven stems ("inflacion", "segur"). El
   artefacto `stem_surface.json` guarda la forma superficial más frecuente de
   cada stem en el corpus, así que las tablas se leen ("inflación", "seguridad").
   Esta capa no interviene en ningún cálculo.
2. **Interpretación**: un gazetteer de ~12 áreas temáticas con anclas (stems) y
   reglas de decisión. Si un tópico contiene al menos 2 anclas de un área, se
   etiqueta con ese nombre; si no, se queda con la etiqueta automática (sus tres
   términos de mayor peso). Es la parte simbólica del módulo 6: no hay ningún
   modelo entrenado decidiendo el nombre.

La etiqueta manual es una **hipótesis auditable**: en `tables.md` se muestra
junto al nº de anclas que la dispararon, y la evaluación de pureza (§8 del
documento) mide si coincide con el gold manual de 25 discursos.
"""

import json
from pathlib import Path

import mt_scripts.config as config

# --- Gazetteer de áreas temáticas -------------------------------------------
# Las anclas son stems (salida de `ri_scripts.preprocess.analyze`). Se admiten
# varias formas cuando el stemmer parte distinto ("gaz"/"gaza", "israel"/"israeli").
MANUAL_RULES = [
    ("economía, inflación y fiscal", (
        "inflacion", "preci", "fiscal", "deficit", "superavit", "monetari",
        "dolar", "tas", "deud", "ajust", "presupuest", "econom", "economic",
        "crecimient", "pbi", "product",
    )),
    ("empleo y trabajo", (
        "emple", "trabaj", "salari", "desemple", "paro", "laboral", "sindicat",
        "informal", "capacit",
    )),
    ("salud, educación y protección social", (
        "salud", "medic", "vacun", "pension", "program", "social", "hospital",
        "educ", "escuel", "univers", "bienest", "imss", "obra",
    )),
    ("seguridad y justicia", (
        "segur", "polic", "polici", "crimen", "delit", "homicid", "justici",
        "juez", "carcel", "narcotraf", "narcotrafic", "detencion", "violenc",
        "fiscalia", "guardia", "carabiner", "delincuent", "marina", "militar",
    )),
    ("política exterior", (
        "ucrani", "europe", "union", "rusia", "israel", "gaz", "gaza",
        "palestin", "chin", "marruec", "mercosur", "internacional", "exterior",
        "zelenski", "otan", "diplomat", "guerra", "putin", "multilateral",
        "europ", "cumbre", "consejo",
    )),
    ("infraestructura y obras", (
        "obra", "tren", "maya", "carreter", "aeropuert", "construcc",
        "infraestructur", "corredor", "istmo", "puent", "metr", "tramo",
        "sureste", "cancun", "arqueologic",
    )),
    ("energía, clima y medio ambiente", (
        "energet", "electric", "petrole", "gas", "liti", "clim", "ambient",
        "dana", "agua", "renov", "hidrocarbur", "combust", "sequia",
    )),
    ("vivienda y territorio", (
        "viviend", "alquil", "hipotec", "urban", "territori", "municip",
        "comunidad", "barri", "suelo", "terreno", "habitacional",
    )),
    ("democracia, instituciones y electoral", (
        "democraci", "electoral", "eleccion", "vot", "partid", "congres",
        "judicial", "constitucion", "reform", "dnu", "decret", "amnisti",
        "investidur", "oposicion", "gobiern", "cataluñ", "autonom", "moncloa",
        "parlamentar", "independencia", "popular",
    )),
    ("pandemia y crisis sanitaria", (
        "pandemi", "covid", "coronavirus", "confinamient", "contagi",
        "cuarenten", "virus",
    )),
    ("migración y frontera", (
        "migrac", "migrant", "fronter", "inmigrac", "refugi", "extranjer",
    )),
    ("igualdad, género y derechos", (
        "iguald", "mujer", "mujeres", "gener", "violenc", "infanci", "niñ",
        "nino", "ninos", "diversidad", "lgbt", "feminicidi", "machism",
        "discrimin", "derech",
    )),
    ("deporte y cultura", (
        "deport", "medall", "olimp", "futbol", "beisbol", "cultur", "museo",
        "libro", "cine", "festival", "premi", "art"
    )),
    ("ideología, libertad y valores", (
        "libert", "liberal", "libertari", "ideolog", "socialism", "comunism",
        "meritocraci", "estatism", "colectivism", "caste", "progresism",
    )),
    ("discurso y comunicación política", (
        "interlocutor", "puebl", "nacion", "pais", "ciudadan", "president",
        "gobiern", "verdad", "mentir", "palabr",
    )),
]

# Anclas que aparecen en el gold manual (ver `gold/topics_gold.json`): sirven
# para comprobar que el gazetteer cubre el vocabulario realmente usado.
MIN_ANCHOR_HITS = 2      # anclas necesarias para aceptar el nombre manual
LABEL_WINDOW = 20        # se buscan anclas en los top-20 términos
CORE_WINDOW = 10         # y al menos una debe estar en el top-10


def load_surfaces(path: Path = None) -> dict:
    path = Path(path or config.STEM_SURFACE_JSON)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def readable(term: str, surfaces: dict) -> str:
    """Forma superficial más frecuente del stem (o el stem si no está)."""
    return surfaces.get(term, term)


def auto_label(terms: list, surfaces: dict, n: int = 3) -> str:
    """Etiqueta automática: los n términos de mayor peso, legibles y unidos."""
    shown = [readable(t, surfaces) for t in terms[:n]]
    if not shown:
        return ""
    shown[0] = shown[0][:1].upper() + shown[0][1:]
    return " · ".join(shown)


def match_rule(terms: list, core: set = None) -> tuple | None:
    """Área temática del gazetteer que mejor cubre el tópico.

    Se exige ``MIN_ANCHOR_HITS`` anclas entre los top-``LABEL_WINDOW`` términos
    **y** al menos una en el top-``CORE_WINDOW``: así no se etiqueta un tópico
    como "economía" porque una palabra suelta de ese campo aparezca en la cola
    de su lista de términos.
    """
    terms_set = set(terms)
    core = core or set(terms[:CORE_WINDOW])
    best = None
    for name, anchors in MANUAL_RULES:
        hits = sorted(a for a in anchors if a in terms_set)
        in_core = any(h in core for h in hits)
        if len(hits) >= MIN_ANCHOR_HITS and in_core and (best is None or len(hits) > len(best[2])):
            best = (name, hits[0], hits)
    return best


def topic_label(model, topic: int, surfaces: dict, n_terms: int = config.TOP_TERMS_WINDOW) -> dict:
    """Etiqueta completa de un tópico: automática, manual y evidencia de la regla.

    Devuelve ``{"auto", "manual", "name", "anchors"}``. ``name`` es la manual si
    existe y la automática en caso contrario: nunca se inventa un nombre sin
    anclas que lo respalden.
    """
    terms = model.top_terms(topic, n_terms)
    auto = auto_label(terms, surfaces)
    core = set(model.top_terms(topic, CORE_WINDOW))
    rule = match_rule(model.top_terms(topic, LABEL_WINDOW), core)
    if rule is None:
        return {"auto": auto, "manual": None, "name": auto, "anchors": []}
    name, _, hits = rule
    return {"auto": auto, "manual": name, "name": name, "anchors": hits}


def label_table(model, surfaces: dict, n_terms: int = config.TOP_TERMS) -> list:
    """Filas ``{topic, name, auto, anchors, top_terms}`` para las tablas."""
    rows = []
    for t in range(model.k):
        info = topic_label(model, t, surfaces)
        rows.append({
            "topic": t,
            "name": info["name"],
            "auto": info["auto"],
            "manual": info["manual"] or "",
            "anchors": len(info["anchors"]),
            "anchor_terms": ", ".join(info["anchors"][:6]),
            "top_terms": ", ".join(readable(x, surfaces) for x in model.top_terms(t, n_terms)),
        })
    return rows
