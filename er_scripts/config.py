"""Configuración del módulo de Entidades y Relaciones (ER).

Módulo 3 del plan: grafo de entidades y relaciones sobre el corpus de discursos.
Reutiliza el segmentador y el autómata del módulo de EI, y el corpus/`doc_id` del
módulo de RI (mismo espacio de identificadores en todo el proyecto).
"""

from pathlib import Path

from ri_scripts import config as ri_config

# --- Rutas -------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
ARTIFACTS = PKG_DIR / "artifacts"

CORPUS_JSONL = ri_config.CORPUS_JSONL
CORPUS_OFFSETS = ri_config.CORPUS_OFFSETS

MENTIONS_JSONL = ARTIFACTS / "mentions.jsonl"
RELATIONS_JSONL = ARTIFACTS / "relations.jsonl"
GRAPH_JSON = ARTIFACTS / "graph.json"
GRAPH_GRAPHML = ARTIFACTS / "graph.graphml"
STATS_JSON = ARTIFACTS / "stats.json"
EVAL_JSON = ARTIFACTS / "eval.json"
TABLES_MD = ARTIFACTS / "tables.md"
GOLD_SENTENCES_CSV = PKG_DIR / "gold" / "gold_sentences.csv"

# --- Menciones ---------------------------------------------------------------
# Para el gazetteer de personas se exige que la superficie venga capitalizada
# (si no, "maduro" el adjetivo se confundiría con Maduro el presidente).
REQUIRE_CAPITALIZED_PERSON = True
# Igual para organizaciones: "ciudadanos", "podemos" o "firma" en minúscula son
# palabras comunes, no entidades (la capitalización es la única pista simbólica).
REQUIRE_CAPITALIZED_ORG = True

# Un nombre propio "de patrón" se acepta si tiene entre 2 y 4 tokens.
NAME_MIN_TOKENS = 2
NAME_MAX_TOKENS = 4

# Tokens que nunca inician un nombre propio de patrón (arranques de oración,
# conectores, meses, días...). Se comparan plegados.
NOT_A_NAME_START = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "este", "esta", "estos",
    "estas", "ese", "esa", "eso", "aquel", "aquella", "mi", "mis", "su", "sus",
    "nuestro", "nuestra", "todo", "toda", "todos", "todas", "otro", "otra",
    "pero", "sin", "con", "para", "por", "como", "cuando", "donde", "porque",
    "aunque", "mientras", "desde", "entre", "según", "segun", "sobre", "tras",
    "hoy", "ayer", "mañana", "manana", "ahora", "además", "ademas", "también",
    "tambien", "así", "asi", "entonces", "luego", "después", "despues", "primero",
    "segundo", "tercero", "finalmente", "incluso", "solo", "sólo", "muchos",
    "muchas", "pocos", "varios", "varias", "nada", "nadie", "nunca", "siempre",
    "sí", "si", "no", "ni", "ya", "muy", "más", "mas", "menos", "tan", "tanto",
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "setiembre", "octubre", "noviembre", "diciembre", "lunes",
    "martes", "miércoles", "miercoles", "jueves", "viernes", "sábado", "sabado",
    "domingo", "señor", "señora", "senor", "senora", "señorías", "senorias",
    "buenos", "buenas", "gracias", "hola", "discurso", "conferencia", "versión",
    "version", "presidente", "presidenta",
}

# Palabras que pueden aparecer dentro de un nombre ("de", "del", "la") pero que
# no aportan al nombre canónico.
NAME_CONNECTORS = {"de", "del", "la", "las", "los", "y", "e", "da", "do", "dos", "van", "von"}

# Un nombre de patrón debe contener al menos un token que no sea conector.
# Marcadores de transcripción que no son entidades (mañaneras de AMLO).
TRANSCRIPT_MARKERS = {
    "pregunta", "preguntas", "intervencion", "intervención", "inaudible",
    "aplausos", "aplauso", "presidente", "presidenta", "moderador", "moderadora",
    "version", "versión", "estenografica", "estenográfica", "traslado",
}

# Cabeceras de organización en su forma **de superficie** (con acentos y mayúscula
# inicial): sólo se aceptan capitalizadas, porque en minúscula son palabras comunes
# ("la secretaría", "el banco").
ORG_HEADS_DISPLAY = [
    "Secretaría", "Instituto", "Comisión", "Consejo", "Banco", "Fondo", "Ministerio",
    "Corte", "Partido", "Sindicato", "Universidad", "Organización", "Asociación",
    "Confederación", "Federación", "Cámara", "Centro", "Hospital", "Escuela",
    "Colegio", "Academia", "Fundación", "Empresa", "Grupo", "Consorcio", "Compañía",
    "Cooperativa", "Sociedad", "Unión", "Fiscalía", "Procuraduría", "Tribunal",
    "Asamblea", "Policía", "Guardia", "Ejército", "Armada", "Marina", "Fuerza",
    "Fuerzas", "Congreso", "Senado", "Parlamento", "Seguro", "Agencia", "Sistema",
    "Servicio", "Secretariado", "Generalitat", "Movimiento", "Frente", "Junta",
    "Coordinación", "Dirección", "Delegación", "Sindicatura", "Defensoría",
    "Contraloría", "Registro", "Instituto", "Hospital", "Centro", "Caja",
]

# Documentos, planes y procesos históricos: van en mayúscula pero no son personas.
NOT_A_PERSON_HEADS = {
    "acuerdo", "acuerdos", "declaracion", "declaración", "protocolo", "tratado",
    "revolucion", "revolución", "plan", "programa", "ley", "decreto", "informe",
    "convenio", "conferencia", "cumbre", "asamblea", "informe", "presupuesto",
    "constitución", "constitucion", "reforma", "iniciativa", "proyecto",
}

GENERIC_ADJECTIVES = {"nacional", "general", "federal", "internacional", "central",
                      "regional", "municipal", "estatal", "social", "publico", "público"}

# Secuencias capitalizadas que el patrón leería como persona y no lo son. Se
# obtuvieron revisando las etiquetas más frecuentes de la capa de patrones.
PERSON_BLOCKLIST = {
    "tren maya", "america latina", "latinoamerica", "distrito federal",
    "seguro social", "seguro popular", "covid", "covid 19", "covid-19",
    "estado de derecho", "sistema de salud", "sistema penal", "poder judicial",
    "estados unidos", "republica bolivariana", "nueva escuela mexicana",
    "aeropuerto internacional", "banco de mexico", "fondo monetario",
    "corte suprema", "consejo general", "casa blanca", "palacio nacional",
    "buenos dias", "buenas tardes", "buenas noches", "muchas gracias",
    "informe de gobierno", "estado de la republica", "union europea",
}

# Cabeceras plegadas (para clasificar rachas de mayúsculas).
ORG_HEADS = {w.lower() for w in """
    secretaria instituto comision consejo banco fondo ministerio corte partido
    sindicato universidad organizacion asociacion confederacion federacion camara
    centro hospital escuela colegio academia fundacion empresa grupo consorcio
    compania cooperativa sociedad union fiscalia procuraduria tribunal asamblea
    policia guardia ejercito armada marina
""".split()}

ORG_SUFFIX_HINTS = {
    "instituto", "secretaria", "secretaría", "comision", "comisión", "consejo",
    "banco", "fondo", "ministerio", "corte", "partido", "sindicato", "universidad",
    "organizacion", "organización", "asociacion", "asociación", "confederacion",
    "confederación", "federacion", "federación", "camara", "cámara", "centro",
    "hospital", "escuela", "colegio", "academia", "fundacion", "fundación",
    "empresa", "grupo", "consorcio", "compañia", "compañía", "petrolera",
    "cooperativa", "sociedad", "union", "unión", "senal", "señal", "comisionado",
    "comisionada", "fiscalia", "fiscalía", "procuraduria", "procuraduría",
}

# --- Coreferencia ------------------------------------------------------------
COREF_MAX_DISTANCE = 3       # oraciones hacia atrás para resolver pronombres
PRONOUNS = {"el", "ella", "ellos", "ellas", "este", "esta", "estos", "estas",
            "aquel", "aquella", "su", "sus"}
# "el presidente", "el mandatario"... se resuelven al hablante del discurso.
SPEAKER_ALIASES = {
    "presidente", "presidenta", "mandatario", "mandataria", "ejecutivo",
    "ejecutiva", "primer mandatario", "jefe del ejecutivo", "jefa del ejecutivo",
    "presidente de la republica", "presidente de la república",
}
# "el gobierno", "mi gobierno", "este gobierno" -> gobierno del país del hablante
GOVERNMENT_ALIASES = {
    "gobierno", "gobierno federal", "gobierno nacional", "mi gobierno",
    "este gobierno", "nuestro gobierno", "gobierno de la republica",
}

# --- Relaciones --------------------------------------------------------------
COOC_WINDOW = 12            # ventana de tokens para coocurrencia
COOC_MAX_ENTITIES = 10      # tope de entidades por oración (evita O(n²) explosivo)
MAX_EVIDENCE = 3            # documentos de evidencia por arista
MIN_RELATION_WEIGHT = 2     # aristas que se publican en el grafo

# Verbos relacionales: stem -> (tipo de relación, dirección)
RELATIONAL_LEMMAS = {
    "inaugurar": "inaugura",
    "entregar": "entrega",
    "visitar": "visita",
    "recibir": "recibe",
    "nombrar": "nombra",
    "designar": "nombra",
    "felicitar": "felicita",
    "agradecer": "agradece",
    "criticar": "critica",
    "acusar": "acusa",
    "denunciar": "denuncia",
    "apoyar": "apoya",
    "rechazar": "rechaza",
    "sancionar": "sanciona",
    "amenazar": "amenaza",
    "invitar": "invita",
    "reunirse": "se_reune_con",
    "firmar": "firma_con",
    "viajar": "viaja_a",
    "gobernar": "gobierna",
    "encabezar": "encabeza",
    "anunciar": "anuncia_a",
    "condenar": "condena",
    "exigir": "exige_a",
    "pedir": "pide_a",
    "decretar": "decreta",
    "promulgar": "promulga",
    "destituir": "destituye",
    "expulsar": "expulsa",
    "saludar": "saluda",
}

# --- Grafo -------------------------------------------------------------------
GRAPHML_ATTRS = ("type", "weight", "docs", "rel_types")
BETWEENNESS_K = 64          # aproximación por muestreo para betweenness

# --- Evaluación --------------------------------------------------------------
MENTION_MATCH_OVERLAP = 0.5  # solapamiento mínimo de span para contar acierto
