"""Léxicos del módulo de entidades y relaciones.

Seis bloques, todos escritos a mano:

1. `PERSONS`      — gazetteer de personas (mandatarios, gabinetes, opositores,
   figuras internacionales mencionadas en los discursos).
2. `PERSON_ALIASES` — variantes por las que se conoce a la misma persona.
3. `ORGS` / `ORGS_PHRASES` — organizaciones por sigla y por nombre completo.
4. `PLACES`       — topónimos (se reutiliza el gazetteer del módulo de EI).
5. `CARGOS`       — cargos con su canónico y, cuando se conoce, su ocupante.
6. `SIGLAS`       — expansiones de siglas para coreferencia.

Todo se pliega con `ri_scripts.preprocess.fold` para que el autómata compare en el
mismo espacio que el resto del proyecto. Los años de mandato se guardan como
comentario, no se usan para filtrar: el corpus tiene fechas inconsistentes.
"""

from ri_scripts.preprocess import fold, stem

from ei_scripts import lexicon as ei_lexicon

# -----------------------------------------------------------------------------
# 1. Personas
# -----------------------------------------------------------------------------
_PERSONS_RAW = {
    # --- México --------------------------------------------------------------
    "Andrés Manuel López Obrador": ["AMLO", "López Obrador", "Obrador", "el tabasqueño", "el Peje"],
    "Claudia Sheinbaum": ["Sheinbaum", "Claudia Sheinbaum Pardo"],
    "Marcelo Ebrard": ["Ebrard", "Marcelo Ebrard Casaubón"],
    "Rogelio Ramírez de la O": ["Ramírez de la O"],
    "Rosa Icela Rodríguez": ["Rosa Icela"],
    "Adán Augusto López": ["Adán Augusto"],
    "Olga Sánchez Cordero": ["Sánchez Cordero"],
    "Hugo López-Gatell": ["López-Gatell", "Gatell"],
    "Jorge Alcocer": [],
    "Zoé Robledo": ["Robledo"],
    "Luis Crescencio Sandoval": ["Crescencio Sandoval"],
    "José Rafael Ojeda Durán": ["Ojeda Durán"],
    "Manuel Bartlett": ["Bartlett"],
    "Rocío Nahle": ["Nahle"],
    "Octavio Romero Oropeza": ["Romero Oropeza"],
    "Irma Eréndira Sandoval": ["Eréndira Sandoval"],
    "Arturo Herrera": [],
    "Tatiana Clouthier": ["Clouthier"],
    "Raquel Buenrostro": ["Buenrostro"],
    "Ariadna Montiel": [],
    "Jesús Ramírez Cuevas": [],
    "Ricardo Sheffield": ["Sheffield"],
    "Alejandro Gertz Manero": ["Gertz Manero"],
    "Santiago Nieto": [],
    "Luisa María Alcalde": ["Luisa Alcalde"],
    "María Luisa Albores": [],
    "Víctor Villalobos": [],
    "Juan Ramón de la Fuente": [],
    "Delfina Gómez": ["Delfina Gómez Álvarez"],
    "Evelyn Salgado": [],
    "Alfonso Durazo": [],
    "Miguel Barbosa": [],
    "Cuitláhuac García": [],
    "Layda Sansores": [],
    "Samuel García": [],
    "Enrique Alfaro": [],
    "Diego Sinhue Rodríguez Vallejo": ["Diego Sinhue"],
    "Mauricio Vila": [],
    "Ricardo Monreal": ["Monreal"],
    "Mario Delgado": [],
    "Gerardo Fernández Noroña": ["Fernández Noroña", "Noroña"],
    "Ifigenia Martínez": [],
    "Beatriz Gutiérrez Müller": [],
    "Ernesto Zedillo": [],
    "Felipe Calderón": ["Calderón"],
    "Carlos Salinas de Gortari": ["Salinas de Gortari"],
    "Vicente Fox": [],
    "Enrique Peña Nieto": ["Peña Nieto"],
    "Genaro García Luna": ["García Luna"],
    "Joaquín Guzmán Loera": ["El Chapo", "Chapo Guzmán"],
    "Emilio Lozoya": [],
    "Rosario Robles": [],
    "Javier Duarte": [],
    "Tomás Zerón": [],
    "Jesús Murillo Karam": ["Murillo Karam"],
    "Néstor Camacho": [],
    # --- Argentina -----------------------------------------------------------
    "Javier Milei": ["Milei", "el león"],
    "Victoria Villarruel": ["Villarruel"],
    "Luis Caputo": ["Caputo"],
    "Patricia Bullrich": ["Bullrich"],
    "Diana Mondino": ["Mondino"],
    "Guillermo Francos": ["Francos"],
    "Nicolás Posse": [],
    "Karina Milei": [],
    "Santiago Bausili": [],
    "Federico Sturzenegger": ["Sturzenegger"],
    "Mariano Cúneo Libarona": [],
    "Sandra Pettovello": [],
    "Mario Russo": [],
    "Sergio Massa": ["Massa"],
    "Cristina Fernández de Kirchner": ["Cristina Kirchner", "Cristina Fernández", "CFK"],
    "Alberto Fernández": [],
    "Mauricio Macri": ["Macri"],
    "Axel Kicillof": ["Kicillof"],
    "Horacio Rodríguez Larreta": ["Rodríguez Larreta"],
    "María Eugenia Vidal": ["Vidal"],
    "Elisa Carrió": ["Carrió"],
    "José Luis Espert": ["Espert"],
    "Martín Lousteau": [],
    "Hugo Moyano": ["Moyano"],
    "Pablo Moyano": [],
    # --- Chile ---------------------------------------------------------------
    "Sebastián Piñera Echenique": ["Piñera", "Sebastián Piñera"],
    "Andrés Chadwick": ["Chadwick"],
    "Ignacio Briones": ["Briones"],
    "Hernán Larraín": ["Larraín"],
    "Marcela Cubillos": ["Cubillos"],
    "Joaquín Lavín": ["Lavín"],
    "Evelyn Matthei": ["Matthei"],
    "Gabriel Boric": ["Boric", "Gabriel Boric Font"],
    "Michelle Bachelet": ["Bachelet"],
    "Carolina Tohá": ["Tohá"],
    "Camila Vallejo": ["Vallejo"],
    "Giorgio Jackson": ["Jackson"],
    "Izkia Siches": [],
    "Maya Fernández": [],
    "Nicolás Grau": [],
    "Mario Marcel": ["Marcel"],
    "Rodrigo Hinzpeter": [],
    "Felipe Larraín": [],
    "Alberto Espina": [],
    "Antonio Walker": [],
    "Juan Carlos Jobet": [],
    "Karla Rubilar": [],
    "Jaime Bellolio": [],
    "Raúl Figueroa": [],
    "Enrique Paris": [],
    # --- España --------------------------------------------------------------
    "Pedro Sánchez": ["Sánchez", "Pedro Sánchez Pérez-Castejón", "el presidente del Gobierno"],
    "María Jesús Montero": ["Montero"],
    "Nadia Calviño": ["Calviño"],
    "Yolanda Díaz": ["Díaz"],
    "José Luis Escrivá": ["Escrivá"],
    "Fernando Grande-Marlaska": ["Grande-Marlaska", "Marlaska"],
    "Margarita Robles": ["Robles"],
    "Teresa Ribera": ["Ribera"],
    "Luis Planas": ["Planas"],
    "José Manuel Albares": ["Albares"],
    "Félix Bolaños": ["Bolaños"],
    "Pilar Alegría": ["Alegría"],
    "Diana Morant": ["Morant"],
    "Isabel Rodríguez": [],
    "Mónica García": [],
    "Óscar Puente": ["Puente"],
    "Carlos Cuerpo": ["Cuerpo"],
    "Jordi Hereu": [],
    "Elma Saiz": [],
    "Pablo Bustinduy": [],
    "Sira Rego": [],
    "Ana Redondo": [],
    "Ernest Urtasun": [],
    "Pablo Iglesias": ["Iglesias"],
    "Irene Montero": [],
    "Santiago Abascal": ["Abascal"],
    "Alberto Núñez Feijóo": ["Feijóo", "Núñez Feijóo"],
    "Pablo Casado": ["Casado"],
    "Mariano Rajoy": ["Rajoy"],
    "José Luis Rodríguez Zapatero": ["Zapatero"],
    "Felipe González": [],
    "José María Aznar": ["Aznar"],
    "Carles Puigdemont": ["Puigdemont"],
    "Quim Torra": ["Torra"],
    "Pere Aragonès": ["Aragonès", "Aragones"],
    "Salvador Illa": ["Illa"],
    "Ada Colau": ["Colau"],
    "Manuela Carmena": ["Carmena"],
    "Íñigo Errejón": ["Errejón", "Errejon"],
    "Juan Carlos I": [],
    "Felipe VI": [],
    "Letizia": [],
    "Pablo Hernández de Cos": [],
    "Luis de Guindos": ["De Guindos"],
    "Christine Lagarde": ["Lagarde"],
    # --- Venezuela -----------------------------------------------------------
    "Nicolás Maduro": ["Maduro", "Nicolás Maduro Moros"],
    "Cilia Flores": [],
    "Delcy Rodríguez": ["Delcy"],
    "Jorge Rodríguez": [],
    "Diosdado Cabello": ["Cabello", "Diosdado"],
    "Vladimir Padrino López": ["Padrino López"],
    "Tareck El Aissami": ["El Aissami"],
    "Jorge Arreaza": ["Arreaza"],
    "Félix Plasencia": ["Plasencia"],
    "Yván Gil": [],
    "Pedro Tellechea": ["Tellechea"],
    "Héctor Rodríguez": [],
    "Rafael Ramírez": [],
    "Hugo Chávez": ["Chávez", "Hugo Chávez Frías"],
    "Juan Guaidó": ["Guaidó"],
    "Henrique Capriles": ["Capriles"],
    "Leopoldo López": [],
    "María Corina Machado": ["Machado"],
    "Edmundo González": ["Edmundo González Urrutia"],
    "Tarek William Saab": ["Tarek William"],
    # --- internacionales -----------------------------------------------------
    "Joe Biden": ["Biden"],
    "Donald Trump": ["Trump"],
    "Kamala Harris": ["Harris"],
    "Barack Obama": ["Obama"],
    "Hillary Clinton": ["Clinton"],
    "Vladimir Putin": ["Putin"],
    "Volodímir Zelenski": ["Zelenski", "Zelensky", "Volodymyr Zelensky"],
    "Xi Jinping": ["Xi Jinping"],
    "Luiz Inácio Lula da Silva": ["Lula", "Lula da Silva"],
    "Dilma Rousseff": ["Rousseff"],
    "Jair Bolsonaro": ["Bolsonaro"],
    "Evo Morales": ["Evo"],
    "Luis Arce": [],
    "Gustavo Petro": ["Petro"],
    "Iván Duque": ["Duque"],
    "Álvaro Uribe": ["Uribe"],
    "Lenín Moreno": [],
    "Guillermo Lasso": ["Lasso"],
    "Daniel Noboa": ["Noboa"],
    "Mario Abdo Benítez": ["Abdo Benítez"],
    "Santiago Peña": [],
    "Pedro Castillo": ["Castillo"],
    "Dina Boluarte": ["Boluarte"],
    "Nayib Bukele": ["Bukele"],
    "Daniel Ortega": ["Ortega"],
    "Xiomara Castro": [],
    "Luis Lacalle Pou": ["Lacalle Pou"],
    "Tabaré Vázquez": ["Tabaré"],
    "Jeanine Áñez": ["Áñez"],
    "Fidel Castro": ["Fidel"],
    "Raúl Castro": ["Raúl Castro"],
    "Papa Francisco": ["Francisco", "el Papa", "el papa Francisco"],
    "Angela Merkel": ["Merkel"],
    "Olaf Scholz": ["Scholz"],
    "Emmanuel Macron": ["Macron"],
    "Boris Johnson": ["Johnson"],
    "Rishi Sunak": ["Sunak"],
    "Giorgia Meloni": ["Meloni"],
    "Josep Borrell": ["Borrell"],
    "Ursula von der Leyen": ["von der Leyen", "Von der Leyen"],
    "António Guterres": ["Guterres"],
    "Elon Musk": ["Musk"],
    "George Soros": ["Soros"],
    "Keir Starmer": [],
    "Charles Michel": [],
}

# Nombre canónico -> identificador de nodo
def person_id(canon: str) -> str:
    return "PER:" + canon


PERSONS: dict[tuple[str, ...], str] = {}
for canon, aliases in _PERSONS_RAW.items():
    seq = tuple(fold(w) for w in canon.split())
    PERSONS[seq] = canon
    for alias in aliases:
        PERSONS[tuple(fold(w) for w in alias.split())] = canon

# Alias más largos primero a la hora de resolver solapes.
PERSON_MAXLEN = max(len(k) for k in PERSONS)

# -----------------------------------------------------------------------------
# 2. Organizaciones
# -----------------------------------------------------------------------------
# Siglas: sigla -> nombre completo (para coreferencia sigla <-> expansión).
SIGLAS = {
    "IMSS": "Instituto Mexicano del Seguro Social",
    "ISSSTE": "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado",
    "CFE": "Comisión Federal de Electricidad",
    "PEMEX": "Petróleos Mexicanos",
    "SEGOB": "Secretaría de Gobernación",
    "SHCP": "Secretaría de Hacienda y Crédito Público",
    "SRE": "Secretaría de Relaciones Exteriores",
    "SEDENA": "Secretaría de la Defensa Nacional",
    "SEMAR": "Secretaría de Marina",
    "FGR": "Fiscalía General de la República",
    "INE": "Instituto Nacional Electoral",
    "INEGI": "Instituto Nacional de Estadística y Geografía",
    "CONACYT": "Consejo Nacional de Ciencia y Tecnología",
    "BANXICO": "Banco de México",
    "UNAM": "Universidad Nacional Autónoma de México",
    "IPN": "Instituto Politécnico Nacional",
    "IMSS-BIENESTAR": "IMSS Bienestar",
    "ISSSTE": "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado",
    "BID": "Banco Interamericano de Desarrollo",
    "FMI": "Fondo Monetario Internacional",
    "BM": "Banco Mundial",
    "ONU": "Organización de las Naciones Unidas",
    "OEA": "Organización de los Estados Americanos",
    "CEPAL": "Comisión Económica para América Latina y el Caribe",
    "UE": "Unión Europea",
    "OTAN": "Organización del Tratado del Atlántico Norte",
    "OMS": "Organización Mundial de la Salud",
    "OIT": "Organización Internacional del Trabajo",
    "TMEC": "Tratado entre México, Estados Unidos y Canadá",
    "OPEP": "Organización de Países Exportadores de Petróleo",
    "PDVSA": "Petróleos de Venezuela",
    "YPF": "Yacimientos Petrolíferos Fiscales",
    "BCE": "Banco Central Europeo",
    "ALBA": "Alianza Bolivariana para los Pueblos de Nuestra América",
    "MERCOSUR": "Mercado Común del Sur",
    "CELAC": "Comunidad de Estados Latinoamericanos y Caribeños",
    "UNASUR": "Unión de Naciones Suramericanas",
    "PSOE": "Partido Socialista Obrero Español",
    "PP": "Partido Popular",
    "VOX": "Vox",
    "ERC": "Esquerra Republicana de Catalunya",
    "MORENA": "Morena",
    "PAN": "Partido Acción Nacional",
    "PRI": "Partido Revolucionario Institucional",
    "PRD": "Partido de la Revolución Democrática",
    "PT": "Partido del Trabajo",
    "FMI": "Fondo Monetario Internacional",
    "SAT": "Servicio de Administración Tributaria",
    "CNDH": "Comisión Nacional de los Derechos Humanos",
    "COFEPRIS": "Comisión Federal para la Protección contra Riesgos Sanitarios",
    "IMSS": "Instituto Mexicano del Seguro Social",
    "CRE": "Comisión Reguladora de Energía",
    "CNH": "Comisión Nacional de Hidrocarburos",
}

ORG_PHRASES = [
    "Instituto Mexicano del Seguro Social", "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado",
    "Comisión Federal de Electricidad", "Petróleos Mexicanos", "Secretaría de Gobernación",
    "Secretaría de Hacienda y Crédito Público", "Secretaría de Relaciones Exteriores",
    "Secretaría de la Defensa Nacional", "Secretaría de Marina", "Fiscalía General de la República",
    "Instituto Nacional Electoral", "Instituto Nacional de Estadística y Geografía",
    "Banco de México", "Banco Mundial", "Banco Interamericano de Desarrollo",
    "Fondo Monetario Internacional", "Organización de las Naciones Unidas",
    "Organización de los Estados Americanos", "Unión Europea", "Corte Suprema de Justicia",
    "Suprema Corte de Justicia de la Nación", "Cámara de Diputados", "Cámara de Senadores",
    "Congreso de la Unión", "Poder Judicial", "Poder Legislativo", "Poder Ejecutivo",
    "Guardia Nacional", "Ejército", "Fuerzas Armadas", "Policía Federal",
    "Morena", "Partido Acción Nacional", "Partido Revolucionario Institucional",
    "Partido Popular", "Partido Socialista Obrero Español", "Vox", "Podemos",
    "Ciudadanos", "Esquerra Republicana", "Junts", "Frente de Todos", "La Libertad Avanza",
    "Propuesta Republicana", "Partido Comunista", "Partido Socialista",
    "Central Unitaria de Trabajadores", "Confederación de Trabajadores de México",
    "Universidad Nacional Autónoma de México", "Instituto Politécnico Nacional",
    "Universidad de Chile", "Pontificia Universidad Católica",
    "Repsol", "Iberdrola", "Telefónica", "Telmex", "Grupo Carso", "Cemex", "Bimbo",
    "YPF", "PDVSA", "Petrobras", "Pemex", "Chevron", "ExxonMobil", "Shell",
    "Banco Santander", "BBVA", "HSBC", "JPMorgan",
    "Fondo Monetario", "Tribunal Supremo", "Consejo General del Poder Judicial",
    "Ministerio Público", "Fiscalía General del Estado", "Consejo Nacional Electoral",
    "Asamblea Nacional", "Tribunal Electoral", "Banco Central",
]

ORGS: dict[tuple[str, ...], str] = {}
for sigla, expansion in SIGLAS.items():
    ORGS[(fold(sigla),)] = sigla
for phrase in ORG_PHRASES:
    ORGS[tuple(fold(w) for w in phrase.split())] = phrase
for seq, canon in ei_lexicon.ORG_SEQS.items():
    ORGS.setdefault(seq, canon.upper() if len(canon) <= 6 else canon.title())

ORG_MAXLEN = max(len(k) for k in ORGS)

# Expansión plegada -> sigla, para fusionar "Instituto Mexicano del Seguro Social"
# e "IMSS" en el mismo nodo del grafo.
EXPANSION_TO_SIGLA = {
    tuple(fold(w) for w in expansion.split()): sigla
    for sigla, expansion in SIGLAS.items()
}

# -----------------------------------------------------------------------------
# 3. Lugares (reutiliza el gazetteer de EI)
# -----------------------------------------------------------------------------
PLACES = dict(ei_lexicon.PLACE_SEQS)
PLACE_MAXLEN = max(len(k) for k in PLACES)

# Nombre para mostrar (el gazetteer de EI guarda la forma plegada).
_PLACE_DISPLAY_OVERRIDES = {
    "mexico": "México", "espana": "España", "peru": "Perú", "panama": "Panamá",
    "colombia": "Colombia", "venezuela": "Venezuela", "argentina": "Argentina",
    "chile": "Chile", "brasil": "Brasil", "bolivia": "Bolivia", "ecuador": "Ecuador",
    "uruguay": "Uruguay", "paraguay": "Paraguay", "cuba": "Cuba", "haiti": "Haití",
    "nicaragua": "Nicaragua", "honduras": "Honduras", "guatemala": "Guatemala",
    "costa rica": "Costa Rica", "el salvador": "El Salvador",
    "republica dominicana": "República Dominicana", "estados unidos": "Estados Unidos",
    "ciudad de mexico": "Ciudad de México", "estado de mexico": "Estado de México",
    "nuevo leon": "Nuevo León", "baja california": "Baja California",
    "quintana roo": "Quintana Roo", "san luis potosi": "San Luis Potosí",
    "yucatan": "Yucatán", "michoacan": "Michoacán", "sinaloa": "Sinaloa",
    "ucrania": "Ucrania", "rusia": "Rusia", "china": "China", "nueva york": "Nueva York",
    "jalisco": "Jalisco", "sonora": "Sonora", "guerrero": "Guerrero", "oaxaca": "Oaxaca",
    "chiapas": "Chiapas", "veracruz": "Veracruz", "tabasco": "Tabasco",
    "tamaulipas": "Tamaulipas", "chihuahua": "Chihuahua", "coahuila": "Coahuila",
    "durango": "Durango", "zacatecas": "Zacatecas", "guanajuato": "Guanajuato",
    "queretaro": "Querétaro", "hidalgo": "Hidalgo", "tlaxcala": "Tlaxcala",
    "morelos": "Morelos", "colima": "Colima", "nayarit": "Nayarit",
    "aguascalientes": "Aguascalientes", "campeche": "Campeche", "puebla": "Puebla",
    "monterrey": "Monterrey", "guadalajara": "Guadalajara", "tijuana": "Tijuana",
    "juarez": "Juárez", "cordoba": "Córdoba", "santiago": "Santiago",
    "valparaiso": "Valparaíso", "concepcion": "Concepción", "antofagasta": "Antofagasta",
    "temuco": "Temuco", "madrid": "Madrid", "barcelona": "Barcelona",
    "sevilla": "Sevilla", "malaga": "Málaga", "galicia": "Galicia",
    "andalucia": "Andalucía", "cataluna": "Cataluña", "caracas": "Caracas",
    "maracaibo": "Maracaibo", "zulia": "Zulia", "miranda": "Miranda",
    "bolivar": "Bolívar", "tachira": "Táchira", "buenos aires": "Buenos Aires",
    "mendoza": "Mendoza", "rosario": "Rosario", "tucuman": "Tucumán",
    "misiones": "Misiones", "patagonia": "Patagonia", "ushuaia": "Ushuaia",
    "magallanes": "Magallanes", "atacama": "Atacama", "coquimbo": "Coquimbo",
    "maule": "Maule", "biobio": "Biobío", "araucania": "Araucanía",
    "la araucania": "La Araucanía", "padre las casas": "Padre Las Casas",
    "pais vasco": "País Vasco", "castilla leon": "Castilla y León",
    "asturias": "Asturias", "cantabria": "Cantabria", "aragon": "Aragón",
    "toledo": "Toledo", "granada": "Granada", "salamanca": "Salamanca",
    "canarias": "Canarias", "baleares": "Baleares", "bilbao": "Bilbao",
    "murcia": "Murcia", "zaragoza": "Zaragoza", "valencia": "Valencia",
    "anzoategui": "Anzoátegui", "barquisimeto": "Barquisimeto", "maracay": "Maracay",
    "arica": "Arica", "iquique": "Iquique", "santa fe": "Santa Fe", "salta": "Salta",
}


def place_label(seq: tuple[str, ...]) -> str:
    key = " ".join(seq)
    if key in _PLACE_DISPLAY_OVERRIDES:
        return _PLACE_DISPLAY_OVERRIDES[key]
    return " ".join(w.capitalize() for w in key.split())


PLACE_DISPLAY = {seq: place_label(seq) for seq in PLACES}

# País -> nombre del gobierno (nodo ORGANIZATION que representa al Estado)
COUNTRY_GOVERNMENT = {
    "MEX": "Gobierno de México",
    "ARG": "Gobierno de Argentina",
    "CHL": "Gobierno de Chile",
    "ESP": "Gobierno de España",
    "VEN": "Gobierno de Venezuela",
}

# -----------------------------------------------------------------------------
# 4. Cargos
# -----------------------------------------------------------------------------
# cargo plegado -> (canónico, ocupante sugerido o None)
# Los ocupantes son una aproximación estática (administración cubierta por el
# corpus); cuando el corpus cambia de gobierno el mapping queda desactualizado y
# por eso el cargo se guarda como mención propia, no como verdad.
CARGOS: dict[tuple[str, ...], tuple[str, str | None]] = {}


def _add_cargo(cargo: str, canon: str, holder: str | None = None):
    CARGOS[tuple(fold(w) for w in cargo.split())] = (canon, holder)


for cargo, canon in [
    ("presidente", "presidente"), ("presidenta", "presidente"),
    ("presidente de la republica", "presidente"),
    ("presidente del gobierno", "presidente del gobierno"),
    ("vicepresidente", "vicepresidente"), ("vicepresidenta", "vicepresidente"),
    ("secretario", "secretario"), ("secretaria", "secretario"),
    ("ministro", "ministro"), ("ministra", "ministro"),
    ("canciller", "canciller"), ("gobernador", "gobernador"), ("gobernadora", "gobernador"),
    ("senador", "senador"), ("senadora", "senador"),
    ("diputado", "diputado"), ("diputada", "diputado"),
    ("fiscal general", "fiscal general"), ("fiscal", "fiscal"),
    ("director", "director"), ("directora", "director"),
    ("rector", "rector"), ("rectora", "rector"),
    ("embajador", "embajador"), ("embajadora", "embajador"),
    ("juez", "juez"), ("jueza", "juez"), ("magistrado", "magistrado"),
    ("general", "general"), ("almirante", "almirante"),
    ("comandante", "comandante"), ("delegado", "delegado"),
    ("subsecretario", "subsecretario"), ("subsecretaria", "subsecretario"),
    ("coordinador", "coordinador"), ("coordinadora", "coordinador"),
    ("portavoz", "portavoz"), ("vocero", "vocero"),
]:
    _add_cargo(cargo, canon)

# Ocupantes conocidos: (cargo, país) -> persona.
CARGO_HOLDERS: dict[tuple[str, str], str] = {}


def _holder(cargo: str, country: str, person: str):
    CARGO_HOLDERS[(fold(cargo), country)] = person


_holder("secretario de hacienda", "MEX", "Rogelio Ramírez de la O")
_holder("secretario de relaciones exteriores", "MEX", "Marcelo Ebrard")
_holder("secretaria de gobernacion", "MEX", "Luisa María Alcalde")
_holder("secretario de la defensa", "MEX", "Luis Crescencio Sandoval")
_holder("secretario de marina", "MEX", "José Rafael Ojeda Durán")
_holder("director del imss", "MEX", "Zoé Robledo")
_holder("fiscal general", "MEX", "Alejandro Gertz Manero")
_holder("secretaria de economia", "MEX", "Raquel Buenrostro")
_holder("secretaria de seguridad", "MEX", "Rosa Icela Rodríguez")
_holder("presidente del gobierno", "ESP", "Pedro Sánchez")
_holder("ministra de hacienda", "ESP", "María Jesús Montero")
_holder("ministro de asuntos exteriores", "ESP", "José Manuel Albares")
_holder("ministra de defensa", "ESP", "Margarita Robles")
_holder("ministra para la transicion ecologica", "ESP", "Teresa Ribera")
_holder("ministro del interior", "ESP", "Fernando Grande-Marlaska")
_holder("ministra de trabajo", "ESP", "Yolanda Díaz")
_holder("vicepresidenta", "ESP", "Nadia Calviño")
_holder("ministro de economia", "ARG", "Luis Caputo")
_holder("ministra de seguridad", "ARG", "Patricia Bullrich")
_holder("canciller", "ARG", "Diana Mondino")
_holder("vicepresidenta", "ARG", "Victoria Villarruel")
_holder("vicepresidenta", "VEN", "Delcy Rodríguez")
_holder("ministro de defensa", "VEN", "Vladimir Padrino López")
_holder("fiscal general", "VEN", "Tarek William Saab")
_holder("ministro del interior", "CHL", "Andrés Chadwick")
_holder("ministro de hacienda", "CHL", "Ignacio Briones")

CARGO_MAXLEN = max(len(k) for k in CARGOS)

# -----------------------------------------------------------------------------
# 5. Utilidades
# -----------------------------------------------------------------------------


def is_person_surface_capitalized(surface: str) -> bool:
    """El gazetteer de personas exige mayúscula inicial (evita 'maduro' adjetivo)."""
    return bool(surface) and (surface[0].isupper() or surface.isupper())


def cargo_holder(cargo: str, country: str) -> str | None:
    return CARGO_HOLDERS.get((fold(cargo), country))


def canonical_cargo(cargo: str) -> str | None:
    entry = CARGOS.get(tuple(fold(w) for w in cargo.split()))
    return entry[0] if entry else None


def expansion_of(sigla: str) -> str | None:
    return SIGLAS.get(sigla.upper())


def sigla_stems(text: str) -> set[str]:
    """Stems de las palabras de una expansión, para emparejar siglas con su forma larga."""
    return {stem(fold(w)) for w in text.split() if len(w) > 3}
