"""Léxicos del extractor de anuncios/promesas.

Todo es a mano y declarativo: aquí no hay nada aprendido. Hay seis bloques:

1. `TRIGGERS`   — disparadores de promesa/anuncio, multi-token, con clase y fuerza.
2. `ACTION_VERBS` — verbos de acción "prometible" (lo que puede ser objeto de un
   compromiso). Sirve para rechazar *"vamos a ver"*, *"vamos a platicar"*.
3. `NOUN_TYPES` — lexicón sustantivo -> tipo de evento (taxonomía cerrada).
4. `PLACES` / `ORGS` — gazetteers de lugares y organizaciones.
5. `MONTHS`, `DETS`, `PREPS`, ... — palabras cerradas para el chunker y la CFG.
6. `NEGATORS` — disparadores léxicos de negación para el autómata de negación.

Las claves se guardan **plegadas** (minúsculas, sin acentos) y, cuando aplica,
**con stemming Snowball**, reutilizando `ri_scripts.preprocess.stem` para que el
matching sea idéntico al del índice de RI.
"""

from ri_scripts.preprocess import fold, stem

# -----------------------------------------------------------------------------
# 1. Disparadores
# -----------------------------------------------------------------------------
# (secuencia plegada, clase, canónico). La clase fija la fuerza vía config.STRENGTH.
_RAW_TRIGGERS = [
    # --- compromiso explícito -------------------------------------------------
    (("me", "comprometo"), "commit", "compromiso"),
    (("me", "comprometo", "a"), "commit", "compromiso"),
    (("nos", "comprometemos"), "commit", "compromiso"),
    (("nos", "comprometemos", "a"), "commit", "compromiso"),
    (("mi", "compromiso"), "commit", "compromiso"),
    (("mi", "compromiso", "es"), "commit", "compromiso"),
    (("nuestro", "compromiso"), "commit", "compromiso"),
    (("asumo", "el", "compromiso"), "commit", "compromiso"),
    (("asumo", "la", "responsabilidad"), "commit", "compromiso"),
    (("tengo", "un", "compromiso"), "commit", "compromiso"),
    (("prometo"), "commit", "promesa"),
    (("prometemos"), "commit", "promesa"),
    (("les", "prometo"), "commit", "promesa"),
    (("te", "prometo"), "commit", "promesa"),
    (("doy", "mi", "palabra"), "commit", "promesa"),
    (("damos", "nuestra", "palabra"), "commit", "promesa"),
    (("cumplire"), "commit", "cumplimiento"),
    (("cumpliremos"), "commit", "cumplimiento"),
    (("vamos", "a", "cumplir"), "commit", "cumplimiento"),
    # --- futuro perifrástico --------------------------------------------------
    (("voy", "a"), "future_1p", "futuro"),
    (("vamos", "a"), "future_1p", "futuro"),
    (("van", "a"), "future_1p", "futuro"),          # 3a persona institucional
    (("va", "a"), "future_1p", "futuro"),
    (("vamos", "a", "seguir"), "future_1p", "continuidad"),
    # --- anuncio performativo -------------------------------------------------
    (("anuncio", "que"), "announce", "anuncio"),
    (("anuncio", "hoy"), "announce", "anuncio"),
    (("les", "anuncio"), "announce", "anuncio"),
    (("anunciamos"), "announce", "anuncio"),
    (("me", "permito", "anunciar"), "announce", "anuncio"),
    (("quiero", "anunciar"), "announce", "anuncio"),
    (("quiero", "informar"), "announce", "anuncio"),
    (("doy", "a", "conocer"), "announce", "anuncio"),
    (("damos", "a", "conocer"), "announce", "anuncio"),
    (("informo", "que"), "announce", "anuncio"),
    (("informo", "a"), "announce", "anuncio"),
    (("les", "informo"), "announce", "anuncio"),
    (("hago", "publico"), "announce", "anuncio"),
    # --- instrucción / acto administrativo -----------------------------------
    (("he", "girado", "instrucciones"), "instruct", "instruccion"),
    (("he", "dado", "instrucciones"), "instruct", "instruccion"),
    (("gire", "instrucciones"), "instruct", "instruccion"),
    (("he", "instruido"), "instruct", "instruccion"),
    (("he", "pedido"), "instruct", "instruccion"),
    (("he", "solicitado"), "instruct", "instruccion"),
    (("he", "ordenado"), "instruct", "instruccion"),
    (("he", "encargado"), "instruct", "instruccion"),
    # --- marco temporal / meta (no es promesa por sí solo) -------------------
    (("antes", "de", "que", "termine"), "goal", "plazo"),
    (("antes", "de", "terminar"), "goal", "plazo"),
    (("a", "mas", "tardar"), "goal", "plazo"),
    (("en", "los", "proximos"), "goal", "plazo"),
    (("en", "los", "siguientes"), "goal", "plazo"),
    (("este", "ano"), "goal", "plazo"),
    (("el", "proximo", "ano"), "goal", "plazo"),
    # --- deóntico (señal débil) ----------------------------------------------
    (("hay", "que"), "deontic", "deber"),
    (("tenemos", "que"), "deontic", "deber"),
    (("debemos"), "deontic", "deber"),
    # --- compromiso en forma nominal / plural --------------------------------
    (("tenemos", "un", "compromiso"), "commit", "compromiso"),
    (("tenemos", "compromiso"), "commit", "compromiso"),
    (("tenemos", "el", "compromiso"), "commit", "compromiso"),
    (("estamos", "comprometidos"), "commit", "compromiso"),
    (("estamos", "comprometido"), "commit", "compromiso"),
    (("estoy", "comprometido"), "commit", "compromiso"),
    (("esta", "comprometido"), "commit", "compromiso"),
    (("esta", "comprometida"), "commit", "compromiso"),
    (("el", "gobierno", "esta", "comprometido"), "commit", "compromiso"),
    # --- anuncio / propuesta en presente ------------------------------------
    (("anuncio",), "announce", "anuncio"),
    (("proponemos",), "announce", "propuesta"),
    (("propongo",), "announce", "propuesta"),
    (("planteamos",), "announce", "propuesta"),
]


def _triggers():
    table = {}
    for seq, cls, canon in _RAW_TRIGGERS:
        table[tuple(fold(t) for t in seq)] = (cls, canon)
    return table


TRIGGERS = _triggers()

# -----------------------------------------------------------------------------
# 2. Verbos de acción prometible
# -----------------------------------------------------------------------------
ACTION_VERBS = """
construir edificar levantar ampliar remodelar rehabilitar reparar restaurar
inaugurar entregar repartir distribuir becar pensionar subsidiar apoyar
aumentar incrementar subir reducir bajar eliminar quitar cancelar condonar
crear fundar abrir instalar equipar comprar adquirir contratar capacitar
invertir destinar asignar etiquetar presupuestar financiar pagar saldar
garantizar asegurar proteger cuidar atender beneficiar incorporar incluir
impulsar promover fomentar fortalecer mejorar modernizar transformar
digitalizar universalizar conectar cablear sembrar plantar reforestar
terminar concluir finalizar arrancar iniciar comenzar emprender ejecutar
poner llevar dar enviar presentar proponer anunciar firmar decretar promulgar
combatir perseguir sancionar castigar juzgar vigilar patrullar desplegar
regular normar supervisar auditar transparentar publicar informar
construiremos entregaremos
""".split()

# Formas de futuro sintético generadas por regla morfológica (1sg/1pl/3sg/3pl).
# Regla: infinitivo -ar -> -aré/-aremos/-ará/-arán; -er/-ir -> -eré/-eremos/-irá/-irán
def future_forms_1st(infinitive: str) -> list[str]:
    """Sólo 1a persona (singular y plural) del futuro sintético.

    La 3a persona de los verbos en -ar ("terminará" -> "terminara") es idéntica a
    la forma del imperfecto de subjuntivo, así que como disparador es ruido.
    """
    return [f for f in future_forms(infinitive) if f.endswith(("are", "aremos", "ere", "eremos"))]


def future_forms_3rd(infinitive: str) -> list[str]:
    """3a persona del futuro ("entregará", "entregarán").

    Se acepta sólo si la oración tiene sujeto atribuible al hablante/gobierno,
    porque en -ar es indistinguible del imperfecto de subjuntivo ("finalizara").
    """
    return [f for f in future_forms(infinitive) if f.endswith(("ara", "aran", "era", "eran"))]


def future_forms(infinitive: str) -> list[str]:
    """Genera las formas de futuro sintético de un infinitivo regular."""
    if infinitive.endswith("ar"):
        stem_ = infinitive[:-2]
        return [stem_ + "are", stem_ + "aremos", stem_ + "ara", stem_ + "aran"]
    if infinitive.endswith(("er", "ir")):
        stem_ = infinitive[:-2]
        return [stem_ + "ere", stem_ + "eremos", stem_ + "era", stem_ + "eran"]
    return []


def _action_forms():
    """Conjunto plegado de formas de acción (lema + futuro + participio/pasado 1pl)."""
    forms = set()
    for verb in ACTION_VERBS:
        v = fold(verb)
        forms.add(v)
        forms.update(future_forms(v))
        if v.endswith("ar"):
            forms.add(v[:-2] + "amos")
            forms.add(v[:-2] + "o")
        elif v.endswith(("er", "ir")):
            forms.add(v[:-2] + "imos")
            forms.add(v[:-2] + "io")
        forms.add(v + "se")
    return forms


# Subconjunto de acciones de *contenido* (entregan algo verificable). El resto de
# acciones son "de discurso" (informar, presentar, anunciar): son anuncios, pero
# no comprometen una entrega. Esto da granularidad al score de confianza.
CONTENT_VERBS = """
construir edificar levantar ampliar remodelar rehabilitar reparar restaurar
inaugurar entregar repartir distribuir becar pensionar subsidiar
aumentar incrementar subir reducir bajar eliminar condonar crear fundar
abrir instalar equipar comprar adquirir contratar capacitar invertir destinar
asignar pagar saldar garantizar asegurar proteger atender beneficiar
impulsar promover fomentar fortalecer mejorar modernizar transformar
digitalizar universalizar conectar cablear sembrar plantar reforestar
terminar concluir finalizar arrancar iniciar emprender ejecutar poner
llevar dar enviar firmar decretar promulgar combatir perseguir sancionar
castigar juzgar vigilar patrullar desplegar regular normar supervisar
auditar transparentar
""".split()

DISCOURSE_VERBS = """
informar presentar anunciar proponer proponer plantear convocar invitar
explicar comentar mencionar exponer decir hablar
""".split()

ACTION_FORMS = _action_forms()


def _forms_of(verbs) -> set[str]:
    forms = set()
    for verb in verbs:
        v = fold(verb)
        forms.add(v)
        forms.update(future_forms(v))
        if v.endswith("ar"):
            forms.add(v[:-2] + "amos")
            forms.add(v[:-2] + "o")
        elif v.endswith(("er", "ir")):
            forms.add(v[:-2] + "imos")
            forms.add(v[:-2] + "io")
    return forms


CONTENT_FORMS = _forms_of(CONTENT_VERBS)
CONTENT_STEMS = {stem(v) for v in CONTENT_FORMS}
ACTION_STEMS = {stem(v) for v in ACTION_FORMS}

# Verbos que aparecen pegados a "vamos a" sin ser promesa (muletillas de mitin).
STOP_VERBS = {"ver", "decir", "hablar", "platicar", "comentar", "preguntar",
              "escuchar", "seguir", "empezar", "comenzar", "terminar"}

# -----------------------------------------------------------------------------
# 3. Sustantivos -> tipo de evento
# -----------------------------------------------------------------------------
_NOUN_TYPES_RAW = {
    "infraestructura": """
        carretera carreteras autopista autopistas puente puentes tunel libramiento
        pavimentacion pavimento camino caminos brecha obra obras presa acueducto
        drenaje alcantarillado planta hidroelectrica refineria gasoducto oleoducto
        aeropuerto puerto muelle metro tren ferrocarril via vias carril carriles
        infraestructura construccion edificio unidad deportiva estadio parque
        mercado central de abasto luminaria alumbrado semaforo banqueta
    """,
    "salud": """
        hospital hospitales clinica clinicas centro de salud medicina medicamento
        medicamentos vacuna vacunas dosis camas medico medicos enfermera enfermeras
        imss issste insabi seguro popular atencion medica consultorio quirófano
        quirófanos mastografo tomografo ambulancia ambulancias hemodialisis
    """,
    "educacion": """
        escuela escuelas universidad universidades beca becas maestro maestros
        docente docentes alumno alumnos estudiante estudiantes libro libros
        biblioteca guarderia jardin de ninos preparatoria bachillerato
        educacion kinder secundaria tecnico normal rural aula aulas
    """,
    "social": """
        pension pensiones jubilacion apoyo apoyos programa programas subsidio
        transferencia vivienda viviendas casa casas agua potable luz electricidad
        tarifa tarifas salario minimo canasta basica despensa credito creditos
        guarderias estancia infantil adulto mayor discapacidad
    """,
    "economia_empleo": """
        empleo empleos trabajo trabajos salario salarios inversion inversiones
        empresa empresas economia inflacion impuesto impuestos deuda presupuesto
        petroleo exportacion turismo campo agricultura maiz frijol precio precios
        gasolina energia electrica industria comercio mercado salario minimo
        desarrollo economico derrama economica crecimiento
    """,
    "seguridad": """
        policia policias guardia seguridad crimen homicidio delito delitos
        carcel reclusorio penal militar ejercito marina armas justicia fiscalia
        juez jueces ministerio publico patrulla patrullas operativo extorsion
        secuestro narcotrafico violencia feminicidio
    """,
    "politica_institucional": """
        reforma constitucion ley leyes congreso diputados senadores eleccion
        elecciones democracia decreto instituto organismo corte suprema ine
        transparencia corrupcion auditoria consulta popular plebiscito
        federalismo municipio ayuntamiento gobernacion secretaria
    """,
}


# Palabras funcionales que no pueden ser "el sustantivo" de un tipo de evento.
_FUNCTION_WORDS = {
    "de", "del", "la", "el", "los", "las", "un", "una", "y", "o", "en", "a",
    "por", "para", "con", "sin", "al", "se", "su", "que", "mas", "muy",
}


def _noun_types():
    """Lexicón stem -> tipo de evento (el stem ya absorbe plurales)."""
    table = {}
    for etype, blob in _NOUN_TYPES_RAW.items():
        for phrase in blob.split("\n"):
            for word in phrase.split():
                if word in _FUNCTION_WORDS or len(word) < 3:
                    continue
                table.setdefault(stem(fold(word)), etype)
    # frases multi-palabra del léxico (centro de salud, adulto mayor, ...)
    for phrase, etype in {
        ("centro", "de", "salud"): "salud",
        ("adulto", "mayor"): "social",
        ("salario", "minimo"): "economia_empleo",
        ("desarrollo", "economico"): "economia_empleo",
        ("unidad", "deportiva"): "infraestructura",
        ("jardin", "de", "ninos"): "educacion",
        ("agua", "potable"): "social",
        ("canasta", "basica"): "social",
        ("ministerio", "publico"): "seguridad",
        ("corte", "suprema"): "politica_institucional",
    }.items():
        table[tuple(stem(fold(w)) for w in phrase)] = etype
    return table


NOUN_STEMS = {k: v for k, v in _noun_types().items() if isinstance(k, str)}
NOUN_PHRASES = {k: v for k, v in _noun_types().items() if isinstance(k, tuple)}

# -----------------------------------------------------------------------------
# 4. Gazetteers
# -----------------------------------------------------------------------------
# Topónimos de una palabra (se pliegan y se indexan como patrones de 1 token).
_PLACES_SINGLE = """
mexico cdmx jalisco monterrey guadalajara tijuana juarez queretaro puebla oaxaca
guerrero michoacan chiapas veracruz tabasco campeche yucatan sonora sinaloa
coahuila durango zacatecas aguascalientes hidalgo tlaxcala morelos colima
nayarit guanajuato tamaulipas chihuahua queretaro
argentina cordoba rosario mendoza tucuman salta misiones patagonia ushuaia
chile santiago valparaiso concepcion antofagasta arica iquique magallanes
atacama coquimbo maule biobio araucania temuco
españa madrid barcelona valencia sevilla zaragoza malaga murcia bilbao galicia
andalucia cataluna canarias baleares asturias cantabria aragon toledo granada
salamanca
venezuela caracas maracaibo barquisimeto maracay miranda zulia anzoategui
bolivar tachira
brasil colombia peru bolivia ecuador uruguay paraguay cuba haiti guatemala
honduras nicaragua panama ucrania rusia china europa africa asia
rengo navidad
""".split()

# Topónimos de varias palabras (patrones multi-token).
_PLACES_MULTI = [
    ("ciudad", "de", "mexico"), ("estado", "de", "mexico"), ("nuevo", "leon"),
    ("baja", "california"), ("quintana", "roo"), ("san", "luis", "potosi"),
    ("santa", "fe"), ("buenos", "aires"), ("estados", "unidos"),
    ("nueva", "york"), ("pais", "vasco"), ("castilla", "leon"),
    ("republica", "dominicana"), ("el", "salvador"), ("costa", "rica"),
    ("padre", "las", "casas"), ("la", "araucania"),
    ("distrito", "federal"), ("valle", "de", "mexico"), ("america", "latina"),
    ("latinoamerica",), ("santiago", "de", "chile"), ("agua", "negra"),
    ("peninsula", "de", "yucatan"), ("rio", "de", "janeiro"), ("san", "jose"),
    ("santa", "cruz"), ("san", "vicente"), ("san", "pedro"), ("las", "condes"),
    ("espana",), ("costa", "del", "pacifico"), ("caribe",), ("amazonia",),
]

ORGS = """
imss issste cfe pemex banxico segob shcp sre sedena semar fgr ine banobras
infonavit fovissste conacyt conapo inegi profeco condusef
onu fmi bid oea unesco unicef fao cepal mercosur celac oms oit
ypf petrobras pdvsa repsol iberdrola bbva santander telmex
""".split()

_ORGS_MULTI = [
    ("banco", "de", "mexico"), ("suprema", "corte"), ("camara", "de", "diputados"),
    ("banco", "mundial"), ("union", "europea"), ("mision", "hogares"),
]

# Nombre canónico para mostrar (el patrón se pliega aparte).
def _seq_table(single, multi, canon=None):
    table = {}
    for word in single:
        seq = (fold(word),)
        table[seq] = canon(seq) if canon else " ".join(seq)
    for phrase in multi:
        seq = tuple(fold(w) for w in phrase)
        table[seq] = canon(seq) if canon else " ".join(seq)
    return table


PLACE_SEQS = _seq_table(_PLACES_SINGLE, _PLACES_MULTI)
PLACE_MAXLEN = max(len(k) for k in PLACE_SEQS)
ORG_SEQS = _seq_table(ORGS, _ORGS_MULTI)
ORG_MAXLEN = max(len(k) for k in ORG_SEQS)

# -----------------------------------------------------------------------------
# 5. Palabras cerradas (chunker + CFG)
# -----------------------------------------------------------------------------
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre"]

DETS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "mi", "mis", "nuestro", "nuestra", "nuestros", "nuestras",
        "su", "sus", "todo", "toda", "todos", "todas", "cada"}

PREPS = {"a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde",
         "durante", "en", "entre", "hacia", "hasta", "para", "por", "segun",
         "sin", "sobre", "tras"}

ADVS = {"ya", "hoy", "ahora", "tambien", "ademas", "aqui", "alli", "siempre",
        "nunca", "primero", "luego", "despues", "pronto", "finalmente",
        "incluso", "sobre", "mucho", "muy", "mas", "menos", "todos", "juntos"}

CONJ = {"y", "e", "o", "u", "pero", "sino", "aunque", "porque", "pues",
        "mientras", "cuando", "si", "que", "como"}

# -----------------------------------------------------------------------------
# 6. Negación
# -----------------------------------------------------------------------------
NEGATORS = {"no", "nunca", "jamas", "tampoco", "ningun", "ninguna", "ninguno",
            "nadie", "nada", "ni"}

# Atenuadores: bajan la fuerza aunque no nieguen.
HEDGES = {"quizas", "quiza", "tal", "vez", "podriamos", "podria", "podrian",
          "intentaremos", "propondremos", "exploraremos", "evaluaremos",
          "revisaremos", "analizaremos", "esperamos"}
