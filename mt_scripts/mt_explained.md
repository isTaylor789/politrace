# Módulo 6 — Minería de textos: topic modeling y evolución temporal (MT)

**Proyecto:** Observatorio del discurso presidencial en español
**Unidades del temario cubiertas:** 5.1 (introducción, concepto, aplicaciones), 5.2 (técnicas básicas
de MT: frecuencias, co-ocurrencia, PMI), 5.3 (algoritmos y modelos: LSA, NMF, LDA), 5.4-5.5
(herramientas y software) y 5.6 (más allá de la MT: dinámica temporal de tópicos).

**Restricciones respetadas:** cero IA generativa, cero deep learning, cero transformers y cero
embeddings neuronales. Todo el módulo es determinista y auditable: TF-IDF, SVD truncada, NMF, LDA
(probabilístico clásico), PMI/NPMI, coseno, asignación húngara y reglas léxicas.

**Entregables de este módulo**

| Pieza | Archivo |
|---|---|
| Paquete | `mt_scripts/` (8 módulos + `gold/`) |
| Pipeline | `python mt_scripts/build_topics.py` (~3 min) |
| Notebook de demostración | `MT_demo.ipynb` |
| Artefactos y tablas | `mt_scripts/artifacts/` (`tables.md`, `eval.json`, ...) |
| Este documento | `mt_scripts/mt_explained.md` |

---

## 1. Qué hace y qué no hace

Toma los 4,531 discursos ya indexados por el módulo 1 y descubre **temas latentes** con tres familias
de modelos clásicos, mide su coherencia, los evalúa sin gold grande, y construye **series temporales**
de tópicos por hablante y por año. El resultado alimenta la vista de evolución del dashboard.

| Objetivo | Resultado |
|---|---|
| Vocabulario de MT | 8,000 términos (poda: df ≥ 10, df ≤ 50 %, sin dígitos, sin puesta en escena) |
| Barrido | 3 familias × 5 valores de k = 15 modelos en 187 s totales |
| Modelo elegido | NMF, k = 10 (`c_v` = 0.443, diversidad 0.940, exclusividad 0.682) |
| Tópicos | 10 tópicos con lectura clara (seguridad, exterior, pandemia, protección social, infraestructura, economía, salud, Cataluña, precios de la gasolina) |
| Estabilidad | Jaccard = 0.858 ± 0.175 (5 remuestreos al 80 %) |
| Anclas del módulo 5 | recall = 0.469 (4 casos anuales) |
| Gold manual | pureza = 0.553 (pureza@2 = 0.660, n = 47) |
| Tópicos por hablante | 5 modelos, k = 8 (Maduro k = 6, marcado de baja confianza) |
| Ventanas temporales | 22 modelos hablante-año y 18 transiciones alineadas (deriva) |
| Coste total del pipeline | 187 s |

No se hace: generación abstractiva, etiquetado "de la noticia del día", deduplicación de discursos
(módulo 9 del plan, pendiente), ni comparación entre países (país = persona en este dataset).

---

## 2. Arquitectura

```
      índice de RI (módulo 1)                mt_scripts/matrices.py
 corpus.jsonl · postings.npz · idf.npy   ┌────────────────────────────────┐
 vocabulary.json · index_meta.json  ───► │ X   (conteos)                  │
                                         │ W   (tf·idf sin normalizar)    │
                                         │ W_l2 (tf·idf L2 = 1)           │
                                         └───────────────┬────────────────┘
                                                         │
             ┌───────────────────────────────────────────┼───────────────────────────────┐
             ▼                                           ▼                               ▼
    models.py (5.3)                            coherence.py (5.2)                 labels.py
    LSA · NMF · LDA                       NPMI · c_v · UMass                   stems → formas reales
    topic_word, doc_topic                 candidatos = unión de top-10          + gazetteer manual
             │                                           │                               │
             └──────────────────┬────────────────────────┘                               │
                                ▼                                                        │
                      build_topics.py: barrido + selección 1-SE                           │
                                │                                                        │
        ┌───────────────────────┼──────────────────────────────┬─────────────────────────┘
        ▼                       ▼                              ▼
 timeline.py (5.6)        evaluate.py                   build_topics.py → artifacts/
 vía A: θ por año         estabilidad (Jaccard)          topics_global.json
 vía B: ventanas +        anclas (gold anual)            doc_topics.jsonl
 alineación húngara       pureza (gold manual)           timeline.json
                                                         topics_by_speaker.json
                                                         topics_by_window.json
                                                         eval.json · tables.md
```

| Archivo | Contenido |
|---|---|
| `config.py` | rutas, podas, k, seeds, umbrales, listas |
| `matrices.py` | reconstrucción de X/W/W_l2 desde postings, poda, subespacios, tokens, stems legibles |
| `models.py` | `TopicModel`, LSA/NMF/LDA, diversidad, alineación de tópicos |
| `coherence.py` | NPMI, c_v, UMass, pares más fuertes (sin gensim) |
| `labels.py` | gazetteer de 13 áreas temáticas y reglas de etiquetado |
| `timeline.py` | θ por periodo, ventanas hablante-año, deriva y cadenas |
| `evaluate.py` | anclas, pureza, estabilidad, tablas markdown |
| `build_topics.py` | CLI que orquesta todo y escribe los artefactos |
| `gold/topics_gold.json` | 47 discursos etiquetados a mano (12 áreas) |

---

## 3. Base de RI reutilizada: tres matrices, no una

El módulo 1 deja `w_global.npz` con la matriz TF-IDF de documentos, **filas normalizadas L2** para que
el coseno sea un producto punto. Esa normalización borra la longitud del documento y aplana los
discursos largos (AMLO llega a 20k palabras). Por eso MT no la usa como entrada de los modelos:

$$X_{ij} = \text{tf}_{ij}, \qquad W_{ij} = \text{tf}_{ij}\cdot \text{idf}_j, \qquad
W^{\text{L2}}_{ij} = \frac{W_{ij}}{\sqrt{\sum_{j'} W_{ij'}^2}}$$

| matriz | origen | uso |
|---|---|---|
| `X` | `postings.npz` (CSR término → doc, transpuesta) | LDA, PMI, UMass |
| `W` | `X · idf` (columnas) | NMF, keyness |
| `W_l2` | `normalize(W)` | coseno, SVD, alineación |

La reconstrucción se **verifica**, no se asume:

```
normalize(global_counts(ix) · idf) − w_global
→ max|Δ| = 0.0 ,  nnz = 3,687,905 = nnz(RI)
```

Es decir, MT usa exactamente el mismo vocabulario, el mismo IDF y los mismos `doc_id` que RI; no hay
una segunda indexación que pueda divergir.

---

## 4. Vocabulario de trabajo

El vocabulario de RI (`min_df = 2`, 30,083 términos) es demasiado ruidoso para tópicos: LDA densifica
y los tópicos se llenan de hápax y números. La poda para MT es:

1. `df ≥ 10` — un término de un solo documento no es un tema;
2. `df ≤ 0.50 · N` — si aparece en la mitad de los discursos es plantilla ("presidente", "gobierno");
3. sin dígitos, longitud del stem ≥ 3;
4. fuera la **lista de puesta en escena** (`config.BOILERPLATE_TERMS`, 34 stems): cargos
   (`president`, `señori`, `interlocutor`, `moder`), gentilicios y países de autorreferencia
   (`chil`, `chilen`, `argentin`, `españ`, `español`, `mexic`, `venezuel`), nombres de los cinco
   hablantes (`obrador`, `lopez`, `madur`, `sanchez`, `piñer`, `milei`), ruido de transcripción
   (`aplaus`, `grabacion`, `version`, `estenograf`) y dícticos (`ayer`, `hoy`, `ahor`, `aqui`,
   `moment`, `tem`, `asunt`, `cos`);
5. si quedan más de 8,000, se conservan los de mayor `df · idf` —frecuentes pero distintivos—, no los
   más raros ni los más comunes.

| Magnitud | Valor |
|---|---|
| Documentos | 4,531 |
| Vocabulario RI | 30,083 |
| Vocabulario MT | 8,000 |
| `df` de los términos conservados | 34 – 2,265 |
| No-cero de `X` | 2,579,502 |
| Tokens analizados (para coherencia) | 10,980,218 |
| Stems con forma legible | 44,555 |

### 4.1 Por qué se elimina la puesta en escena

El plan base fija la restricción 1: **5 hablantes, 1 por país → país = persona = ideología**. Si no se
filtra, el modelo agrupa discursos **por quién habla**, no por lo que dice. El tópico dominante de
Piñera queda:

> `chilenos · chile · compatriotas · region · recuperacion · coronavirus · aporte · calidad`

que no es un tema sino una firma estilística. Con la lista de puesta en escena, ese mismo conjunto se
reorganiza en tópicos temáticos (salud/hospitales, pandemia/vacunas, seguridad/carabineros,
igualdad/derechos). El efecto medido en el mismo `k = 10`:

| | sin filtro | con filtro |
|---|---|---|
| pureza (gold manual) | 0.511 | **0.553** |
| pureza@2 | 0.596 | **0.660** |
| `c_v` | 0.439 | **0.441** |

Es una decisión de modelado, no un truco: se elimina aquello que **no es contenido temático** y se
documenta. Lo que se pierde —cuánto habla cada líder de su propio país— lo cubren los módulos 3
(grafo de entidades) y 8 (polarización).

---

## 5. Modelos (temario 5.3)

Los tres reciben el mismo `Space` y devuelven un `TopicModel` con `topic_word` (k × V, filas suman 1),
`doc_topic` (N × k, filas suman 1) y los tópicos en el vocabulario global para poder alinearlos.

### 5.1 LSA — SVD truncada

$$W^{\text{L2}} \approx U_k \Sigma_k V_k^\top, \qquad V_k \in \mathbb{R}^{k \times V}$$

Las filas de `V_k` son los tópicos; como el signo es arbitrario se toma `|V_k|` y se normaliza. La
proyección de los documentos es `U_k Σ_k` (también en valor absoluto y normalizada). Se ajusta con
`TruncatedSVD(n_components=50)`, independiente del `k` de tópicos, para poder comparar con y sin la
truncación. Aporta la línea base no probabilística y la varianza explicada (30.5 % con 50
componentes).

### 5.2 NMF — factorización no negativa

$$\min_{H \ge 0,\, V \ge 0} \|W^{\text{L2}} - H V\|_F^2,
\qquad H \in \mathbb{R}^{N \times k},\; V \in \mathbb{R}^{k \times V}$$

`V` son los tópicos (filas normalizadas a suma 1) y `H` la mezcla de cada documento. Se usa
`init="nndsvda"` (determinista dado el seed), `max_iter = 500`, `tol = 1e-5`, `random_state = 42`.
Los tópicos son aditivos y dispersos: en promedio cada documento activa ~5.7 componentes y su tópico
dominante tiene θ medio 0.57, así que la asignación es blanda, no dura.

### 5.3 LDA — asignación latente de Dirichlet

Modelo generativo clásico:

$$\theta_d \sim \mathrm{Dir}(\alpha),\qquad z_{dn} \sim \mathrm{Cat}(\theta_d),\qquad
w_{dn} \sim \mathrm{Cat}(\phi_{z_{dn}}),\qquad \phi_z \sim \mathrm{Dir}(\beta)$$

Se ajusta por inferencia variacional en línea (`learning_method="online"`, `batch_size = 2048`,
`max_iter = 12`, `n_jobs = -1`). La perplejidad se reporta pero **no** se usa para elegir `k`: baja
al aumentar `k` y no está alineada con la interpretabilidad.

### 5.4 Elección de `k`

La coherencia sube de forma casi monótona con `k`: más tópicos = listas de términos más específicas.
Quedarse con el máximo favorece siempre el `k` mayor y fragmenta el modelo. Se usa la regla de **un
error estándar** (1-SE), habitual en selección de modelos:

1. familia = la de mayor `c_v` con diversidad ≥ 0.60 (un modelo con tópicos solapados no es elegible);
2. dentro de la familia, `SE = σ(c_v por tópico) / √k` del mejor `k`;
3. se elige el **menor** `k` con `c_v ≥ c_v_max − max(SE, 0.05)`.

La banda de 0.05 es del orden del error estándar observado (SE = 0.048 en el barrido): diferencias
menores no son distinguibles con 4,531 documentos. En este corpus:

| | valor |
|---|---|
| `c_v` máxima | 0.492 (NMF, k = 15) |
| SE del mejor | 0.048 |
| umbral | 0.442 |
| k elegido | **10** (`c_v = 0.443`) |

La §11 muestra que la regla no es una excusa: el `k = 15` tiene más coherencia pero peor pureza y
tópicos casi duplicados.

---

## 6. Coherencia desde cero (temario 5.2)

No se usa gensim ni otra caja negra: la métrica que decide el número de tópicos es la misma que se
explica aquí.

### 6.1 Espacio de pares y NPMI

Para cada documento se cuentan los pares de tokens a distancia `d < W` (`W = 10`). El espacio de
muestreo son los pares **ordenados**, de tamaño

$$T = 2\sum_{d=1}^{W-1}(N - d),$$

con `N` el número de tokens del corpus. Con `m_ij` pares ordenados (izquierda `i`, derecha `j`):

$$p_{ij} = \frac{m_{ij}}{T},\qquad p_i = \frac{\sum_j m_{ij}}{T},\qquad
\mathrm{PMI}(i,j) = \log\frac{p_{ij}}{p_i\,p_j},\qquad
\mathrm{NPMI}(i,j) = \frac{\mathrm{PMI}(i,j)}{-\log p_{ij}} \in [-1,1].$$

En la implementación, `M = U + Uᵀ − diag(U)` convierte los conteos no ordenados `U` en ordenados
(cada par `i≠j` cuenta dos veces, una por orden; `i=i` una sola).

### 6.2 c_v

Cada término del tópico se representa por su vector de NPMI restringido a las palabras del tópico y se
promedian los cosenos entre pares:

$$c_v = \frac{2}{n(n-1)}\sum_{i<j} \cos\big(\mathrm{NPMI}(i,\cdot),\ \mathrm{NPMI}(j,\cdot)\big)$$

sobre las `n = 10` palabras de mayor peso. Es una medida de **confirmación semántica**: los términos
de un buen tópico co-ocurren entre sí en contextos parecidos.

### 6.3 UMass

Co-ocurrencia a nivel documento, con los términos ordenados por `df` decreciente:

$$\mathrm{UMass} = \frac{2}{n(n-1)}\sum_{i<j}\log\frac{D(i,j)+1}{D(i)},$$

donde `D(i)` es el `df` del término y `D(i,j)` el número de documentos que contienen ambos. Premia
tópicos cuyas palabras aparecen juntas en los mismos discursos, no sólo cerca. Se calcula con un
producto disperso booleano sobre las filas del índice invertido.

### 6.4 Implementación y coste

- `cooccurrence_counts`: desplaza el vector de índices locales de cada documento (`loc[:-d]` vs
  `loc[d:]`) y acumula pares con un único `bincount` por corpus. 189 candidatos, ventana 10,
  1.04M de pares en ~2 s.
- `doc_cooccurrence`: producto CSR de 189 × 4,531 por su traspuesta → matriz 189 × 189.
- Todo el cálculo de coherencia del barrido tarda **< 3 s**.

Pares con mayor NPMI del corpus (validación cualitativa del PMI):

| término A | término B | NPMI | pares |
|---|---|---|---|
| quintan | roo | 0.697 | 3,726 |
| jueg | panamerican | 0.565 | 135 |
| europe | union | 0.555 | 5,095 |
| jueg | olimp | 0.546 | 127 |
| medall | olimp | 0.543 | 57 |
| may | tren | 0.539 | 5,948 |
| adult | pension | 0.517 | 3,653 |
| astrazenec | pfiz | 0.514 | 1,120 |

Son colocaciones reales del corpus (Quintana Roo, Juegos Panamericanos, Unión Europea, Tren Maya,
pensión para adultos mayores, vacunas AstraZeneca/Pfizer).

---

## 7. Barrido y selección

| familia | k | c_v | UMass | diversidad | exclusividad | tiempo (s) |
|---|---|---|---|---|---|---|
| LSA | 6 | 0.140 | -1.303 | 0.600 | 0.322 | 1.9 |
| LSA | 8 | 0.141 | -1.435 | 0.575 | 0.266 | 2.1 |
| LSA | 10 | 0.128 | -1.596 | 0.540 | 0.225 | 1.8 |
| LSA | 12 | 0.147 | -1.634 | 0.500 | 0.194 | 2.1 |
| LSA | 15 | 0.137 | -1.694 | 0.473 | 0.172 | 1.9 |
| NMF | 6 | 0.198 | -0.962 | 0.917 | 0.708 | 3.0 |
| NMF | 8 | 0.331 | -1.018 | 0.925 | 0.688 | 3.2 |
| **NMF** | **10** | **0.443** | **-1.058** | **0.940** | **0.682** | **4.6** |
| NMF | 12 | 0.444 | -1.053 | 0.925 | 0.653 | 4.7 |
| NMF | 15 | 0.492 | -1.149 | 0.920 | 0.634 | 17.8* |
| LDA | 6 | 0.142 | -0.828 | 0.883 | 0.532 | 19.4 |
| LDA | 8 | 0.158 | -0.814 | 0.800 | 0.426 | 17.3 |
| LDA | 10 | 0.271 | -0.811 | 0.860 | 0.423 | 19.6 |
| LDA | 12 | 0.250 | -0.837 | 0.817 | 0.400 | 19.7 |
| LDA | 15 | 0.284 | -0.930 | 0.760 | 0.384 | 23.6 |

\* NMF con k = 15 alcanza `max_iter = 500` sin converger (aviso de sklearn); no es el elegido.

Lectura:

- **NMF domina** en `c_v` (0.44-0.49) con 3-5 s por ajuste. Es el único que produce tópicos con
  términos que se leen como un tema.
- **LSA** queda en 0.13-0.15: los componentes tienen signo y sus listas mezclan términos ("cient",
  "vacun", "corrupcion" en el mismo componente). Es útil como línea base y para la varianza explicada,
  no para interpretar.
- **LDA** con 12 iteraciones en línea llega a 0.14-0.28: la brecha con NMF no es sólo de ajuste, el
  modelo necesita más iteraciones y el corpus tiene documentos muy largos y heterogéneos. Se reporta
  como contraste probabilístico, no como el modelo principal.
- **Diversidad**: LSA cae a 0.47 en k = 15 (tópicos solapados); NMF se mantiene en 0.92-0.94. La
  diversidad se mide como `|∪ top-10| / (k · 10)`.
- **Exclusividad** media de los top-10: NMF 0.68, LSA 0.17-0.32, LDA 0.38-0.53.

---

## 8. Tópicos del modelo elegido (NMF, k = 10)

Participación media de cada tópico en el corpus y en cada hablante (cada fila de θ suma 1; las
columnas suman 1 entre tópicos). La última columna es la lista de términos de mayor peso en forma
legible.

| T | nombre | partic. | excl. | AMLO | Milei | Maduro | Sánchez | Piñera | términos |
|---|---|---|---|---|---|---|---|---|---|
| 0 | seguridad y justicia | 0.129 | 0.51 | 0.23 | 0.04 | 0.10 | 0.03 | 0.03 | federal, guardia, investigación, ciento, delitos, fiscalía, operación, corrupción, homicidios, marina |
| 1 | política exterior | 0.119 | 0.79 | 0.02 | 0.04 | 0.20 | 0.40 | 0.04 | europea, unión, europa, Ucrania, Putin, consejo, guerra, euros, cumbre, lógicamente |
| 2 | pandemia y crisis sanitaria | 0.054 | 0.66 | 0.06 | 0.00 | 0.07 | 0.03 | 0.07 | vacunas, dosis, pandemia, covid, Pfizer, adultos, población, contagios, sanitaria, virus |
| 3 | salud, educación y protección social | 0.135 | 0.62 | 0.24 | 0.01 | 0.04 | 0.02 | 0.05 | jóvenes, becas, niños, escuelas, adultos, maestros, pensión, corrupción, dinero, créditos |
| 4 | seguridad y justicia | 0.175 | 0.49 | 0.04 | 0.11 | 0.24 | 0.13 | 0.59 | carabineros, compatriotas, niños, región, proteger, mujeres, recuperación, calidad, esfuerzo, colaboración |
| 5 | infraestructura y obras | 0.085 | 0.86 | 0.13 | 0.01 | 0.02 | 0.02 | 0.06 | tren, aeropuerto, maya, obra, tramo, sureste, Cancún, arqueológicas, Roo, Quintana |
| 6 | economía, inflación y fiscal | 0.096 | 0.67 | 0.06 | 0.76 | 0.19 | 0.07 | 0.07 | inflación, PBI, monetaria, libertad, crecimiento, fiscal, tasa, precio, déficit, banco |
| 7 | salud, educación y protección social | 0.064 | 0.79 | 0.11 | 0.00 | 0.04 | 0.01 | 0.04 | hospitales, médicos, IMSS, medicamentos, medicina, atención, enfermeras, doctor, equipo, unidades |
| 8 | democracia, instituciones y electoral | 0.094 | 0.68 | 0.03 | 0.02 | 0.10 | 0.29 | 0.03 | Cataluña, parlamentarios, autonómicas, catalana, euros, popular, vivienda, independencia, ministro, grupo |
| 9 | *centavos · precio · litro* | 0.050 | 0.75 | 0.09 | 0.01 | 0.01 | 0.01 | 0.01 | centavos, precio, litro, gas, margen, PEMEX, gasolina, franquicia, consumidores, ciento |

### 8.1 Etiquetado: dos capas

1. **Presentación.** Los modelos devuelven stems ("inflacion", "segur"). El artefacto
   `stem_surface.json` guarda, por stem, la forma superficial más frecuente del corpus ("inflación",
   "seguridad"), obtenida recorriendo las formas reales y aplicando el mismo stemmer. Esta capa no
   interviene en ningún cálculo.
2. **Interpretación.** `labels.py` tiene un gazetteer de 15 áreas temáticas con anclas (stems). Un
   tópico recibe el nombre manual si tiene **≥ 2 anclas en el top-20 y al menos una en el top-10**
   (así no se etiqueta "economía" por una palabra suelta en la cola). Si no, queda la etiqueta
   automática: sus tres términos de mayor peso. La tabla del reporte muestra cuántas anclas
   dispararon el nombre; el T9 no llegó a 2 y conserva la etiqueta automática, que es lo honesto.

### 8.2 Evidencia

Los documentos con mayor θ en el T6 (economía) son discursos de Milei sobre inflación, PBI, ajuste y
libertad económica; el T7 (salud) son las mañaneras del IMSS y los hospitales; el T5 (infraestructura)
son inauguraciones del Tren Maya y del aeropuerto Felipe Ángeles. `topics_global.json` guarda, por
tópico, los 5 documentos con mayor θ y un extracto del discurso alrededor del primer término del
tópico, para que la interpretación sea auditable sin abrir 4,531 textos.

---

## 9. Evolución temporal, vía A: un modelo, θ agregada por periodo

Un único modelo para todo el corpus; cada documento recibe su θ. Después:

$$\bar\theta_{s,p} = \frac{1}{|D_{s,p}|}\sum_{d \in D_{s,p}} \theta_d,
\qquad \sum_{t} \bar\theta_{s,p,t} = 1,$$

con `s` hablante y `p` año o trimestre. Al ser una **participación** media, la comparación entre
hablantes es proporcional: el desbalance 2,294 : 30 no domina la tabla.

Ventana limpia 2023-2024 (única con 4 de los 5 líderes activos, restricción 3 del plan base):

| T | nombre | AMLO 23 | AMLO 24 | Milei 23 | Milei 24 | Maduro 23 | Sánchez 23 | Sánchez 24 |
|---|---|---|---|---|---|---|---|---|
| 0 | seguridad y justicia | 0.274 | 0.267 | 0.013 | 0.035 | 0.049 | 0.018 | 0.034 |
| 1 | política exterior | 0.015 | 0.014 | 0.002 | 0.028 | 0.829 | 0.575 | 0.375 |
| 2 | pandemia y crisis sanitaria | 0.020 | 0.007 | 0.002 | 0.001 | 0.000 | 0.010 | 0.010 |
| 3 | protección social | 0.181 | 0.195 | 0.003 | 0.008 | 0.000 | 0.014 | 0.017 |
| 4 | seguridad (Chile) | 0.032 | 0.032 | 0.061 | 0.110 | 0.034 | 0.094 | 0.152 |
| 5 | infraestructura | 0.227 | 0.201 | 0.000 | 0.014 | 0.039 | 0.020 | 0.028 |
| 6 | economía | 0.054 | 0.077 | 0.878 | 0.777 | 0.049 | 0.058 | 0.080 |
| 7 | salud | 0.096 | 0.080 | 0.010 | 0.003 | 0.000 | 0.009 | 0.011 |
| 8 | instituciones | 0.025 | 0.046 | 0.031 | 0.017 | 0.000 | 0.195 | 0.291 |
| 9 | precios de la gasolina | 0.077 | 0.080 | 0.000 | 0.006 | 0.000 | 0.008 | 0.003 |

Lectura: Milei concentra 0.78-0.88 de su θ en el tópico económico; Sánchez se reparte entre política
exterior (0.58 → 0.37, con la presidencia de la UE en 2023) e instituciones/Cataluña (0.20 → 0.29);
AMLO se reparte entre protección social, seguridad, infraestructura y su tópico de prensa matutina.
El modelo capta perfiles temáticos distintos, no un promedio del corpus.

Series completas (todas las combinaciones hablante × año y × trimestre) en `timeline.json` y
`artifacts/tables.md`.

**Coste de la vía A**: cero modelos extra, sólo una media de θ. **Límite**: un tópico que nace en 2024
o que cambia de vocabulario no puede manifestarse, porque el vocabulario del tópico es único.

---

## 10. Evolución temporal, vía B: ventanas y alineación de tópicos

Se ajusta un modelo NMF por `(hablante, año)` con ≥ 30 discursos (22 ventanas). Los tópicos de años
consecutivos se emparejan con **asignación óptima húngara** (`linear_sum_assignment` sobre
`-similitud`) y la **deriva** de cada tópico es la similitud entre sus conjuntos de términos.

### 10.1 Por qué conjuntos binarios y no pesos

Al principio se usó el coseno de los vectores de pesos continuos (top-30). El resultado fue un
desastre silencioso: la mediana de la similitud entre ventanas consecutivas era **0.004** y ningún
tópico superaba 0.05, porque las colas de las distribuciones no se solapan y el coseno de dos vectores
dispersos con 2-4 términos compartidos es diminuto. La señal correcta estaba ahí (los emparejamientos
por máximo eran semánticamente correctos), pero el umbral no tenía sentido.

Se usa entonces el coseno de los **conjuntos binarios** de los 50 términos de mayor peso:

$$\mathrm{sim}(t,t') = \frac{|A_t \cap A_{t'}|}{\sqrt{|A_t|\,|A_{t'}|}},$$

es decir, el solapamiento normalizado. Un mismo tema entre dos años comparte 20-29 de 50 términos
(`sim = 0.40-0.58`); dos tópicos ajenos comparten < 10 (`sim < 0.20`). El umbral de emparejamiento es
`sim ≥ 0.25` (≈ 12 de 50 términos compartidos).

### 10.2 Deriva por transición

| hablante | transición | similitud media | emparejados |
|---|---|---|---|
| AMLO | 2019→2020 | 0.505 | 4/8 |
| AMLO | 2020→2021 | 0.454 | 7/8 |
| AMLO | 2021→2022 | 0.406 | 7/8 |
| AMLO | 2022→2023 | 0.480 | 6/8 |
| AMLO | 2023→2024 | 0.456 | 5/8 |
| Milei | 2024→2025 | 0.340 | 6/8 |
| Sánchez | 2018→2019 | 0.387 | 3/8 |
| Sánchez | 2019→2020 | 0.287 | 3/8 |
| Sánchez | 2020→2021 | 0.440 | 1/8 |
| Sánchez | 2021→2022 | 0.345 | 4/8 |
| Sánchez | 2022→2023 | 0.410 | 4/8 |
| Sánchez | 2023→2024 | 0.480 | 3/8 |
| Sánchez | 2024→2025 | 0.436 | 5/8 |
| Sánchez | 2025→2026 | 0.345 | 4/8 |
| Piñera | 2018→2019 | 0.388 | 5/8 |
| Piñera | 2019→2020 | 0.400 | 3/8 |
| Piñera | 2020→2021 | 0.383 | 6/8 |
| Piñera | 2021→2022 | 0.320 | 2/8 |

Globalmente: mediana de similitud 0.260, 54.2 % de tópicos emparejados. Las transiciones con más
tópicos "nuevos" son las de crisis: Sánchez 2020→2021 (COVID: sólo 1 de 8 tópicos se mantiene) y
Piñera 2021→2022 (estallido/seguridad y salida de gobierno). Las más estables son las de AMLO
2020→2022, con 7 de 8 tópicos emparejados.

### 10.3 Cadenas de tópicos

Encadenando los emparejamientos se sigue cada tópico a lo largo de los años. Las cadenas más estables
del corpus:

| hablante | tramo | años | similitud media |
|---|---|---|---|
| Milei | 2024→2025 | 2 | 0.520 |
| AMLO | 2019→2024 | 6 | 0.500 |
| Sánchez | 2018→2026 | 9 | 0.445 |
| Milei | 2024→2025 | 2 | 0.440 |
| Piñera | 2018→2022 | 5 | 0.440 |
| AMLO | 2019→2024 | 6 | 0.404 |

### 10.4 Lo que sólo se ve en la vía B: AMLO por años

| año | T0 | T1 | T2 |
|---|---|---|---|
| 2019 | guardia, investigación, policía, delitos, denuncia (0.23) | médicos, hospitales, IMSS (0.14) | créditos, productores, becas, adultos (0.18) |
| 2020 | epidemia, covid, Gatell, Hugo, coronavirus (0.12) | escuelas, becas, pensión, Oaxaca, alumnos (0.18) | centavos, litro, margen, precio (0.08) |
| 2021 | investigación, fiscalía, denuncia, delitos, candidato (0.18) | centavos, litro, precio, colocación, video (0.09) | Oaxaca, Guerrero, Sembrando, becas, pensión (0.13) |
| 2022 | electoral, video, mentiras, falso, INE (0.20) | centavos, litro, franquicia, margen (0.10) | hospitales, IMSS, Zoé, medicamentos (0.09) |
| 2023 | judicial, cuevas, mentiras, falso, INE (0.18) | centavos, tramo, arqueológicas, precio (0.12) | sucursales, becas, pensión, adultos (0.14) |
| 2024 | judicial, cuevas, falso, mentiras, video (0.18) | centavos, precio, canasta, litro (0.08) | Morelos, virreyes, felicidad, independencia (0.23) |

Se ve el arco completo: seguridad (2019) → COVID (2020) → disputa electoral (2021-2022) → reforma
judicial (2023-2024), con el tópico de precios de la gasolina presente todos los años. Es información
que la vía A no puede dar: en la vía A el COVID es una sola curva de participación; en la vía B se ve
además **cómo se reorganiza el resto del vocabulario alrededor de esa crisis**.

---

## 11. Tópicos por hablante (modelo independiente)

Un modelo por hablante con `k = 8` (Maduro `k = 6` por tamaño). Es la lectura correcta del dataset:
5 hablantes, 1 por país; Maduro queda marcado como **baja confianza** (30 discursos).

| hablante | discursos | tópicos principales (participación · área · términos) |
|---|---|---|
| AMLO | 2,294 | 0.203 protección social (becas, pensión, créditos, Sembrando) · 0.200 instituciones (investigación, judicial, denuncia) · 0.133 seguridad (guardia, delitos, homicidios, marina) |
| Milei | 187 | 0.198 economía (déficit, monetaria, superávit, ajuste) · 0.186 ideología (batalla, casta, occidente, colectivismo) · 0.147 economía monetaria (dinero, tasa, banco central) |
| Maduro | 30 (baja confianza) | 0.263 bloqueo/sanciones (unilateral, coercitivas, bloqueo) · 0.194 electoral (asamblea, comandante, constituyente) · 0.165 OPEP/CLAP |
| Sánchez | 1,136 | 0.191 política exterior (cumbre, multilateral, cooperación) · 0.166 igualdad (mujeres, feminista, deporte) · 0.151 economía (euros, sector, empresas) |
| Piñera | 884 | 0.217 energía/tecnología (energía, comercio, innovación) · 0.193 igualdad y derechos (ley, mujeres, violencia) · 0.166 salud/región (hospitales, Araucanía, agua) |

Comparar los modelos por hablante con el modelo global es informativo: el global obliga a compartir
vocabulario entre los cinco y produce tópicos "promedio" (p. ej. un solo tópico de seguridad para AMLO
y Piñera); el modelo por hablante distingue la seguridad mexicana (guardia, marina, fiscalía) de la
chilena (carabineros, narcotráfico, terrorismo).

---

## 12. Evaluación

No hay gold de tópicos para 4,531 discursos, así que se combinan cuatro señales independientes.

### 12.1 Anclas del gold anual (módulo 5)

`rat_scripts/gold/year_gold.json` lista 7-8 temas clave para 4 casos anuales. Se mide qué fracción
cubren los 3 tópicos dominantes del año (unión de sus top-30 términos).

| caso | discursos | recall | anclas cubiertas | no cubiertas |
|---|---|---|---|---|
| Sánchez 2023 | 129 | 0.500 | 4/8 | Gaza, Marruecos, reindustrialización, CGPJ |
| Sánchez 2024 | 142 | 0.375 | 3/8 | Palestina, Gaza/Israel, DANA, India/China, empleo |
| AMLO 2023 | 308 | 0.500 | 4/8 | IMSS-Bienestar, Acapulco/Otis, Istmo, litio |
| Milei 2024 | 87 | 0.500 | 4/8 | DNU, Pacto de Mayo, Israel/Holocausto, Mercosur |

Recall medio **0.469**. Las anclas no cubiertas son en su mayoría **eventos puntuales** (DANA, Otis,
Acapulco, Pacto de Mayo): un modelo de tópicos global con k = 10 no reserva un tópico para cada evento
de un año; eso lo cubren el módulo 2 (eventos) y el módulo 4 (resumen). El recall mide, por tanto, la
cobertura **temática**, no la de sucesos.

### 12.2 Pureza contra el gold manual

47 discursos etiquetados a mano con 12 áreas (título + inicio del discurso). Se mide la pureza:

$$\text{pureza} = \frac{1}{N}\sum_{a \in \text{áreas}} \max_{t} \#\{d \in a : \text{argmax}_t \theta_d = t\}$$

invariante a la numeración de tópicos. Resultado: **0.553** (pureza@2 = 0.660).

| área temática | n | tópico modal | coincidencia |
|---|---|---|---|
| economía, inflación y fiscal | 5 | T3 / T6 (dividido) | 0.40 |
| política exterior | 5 | T1 | 0.60 |
| infraestructura y obras | 5 | T5 | 0.60 |
| democracia, instituciones y electoral | 5 | T0 | 0.40 |
| pandemia y crisis sanitaria | 5 | T4 / T2 | 0.40 |
| energía, clima y medio ambiente | 4 | T5 | 0.25 |
| empleo y trabajo | 3 | T3 | 0.33 |
| salud, educación y protección social | 3 | T3 | 0.67 |
| seguridad y justicia | 3 | T0 | 0.67 |
| vivienda y territorio | 3 | T8 | 1.00 |
| migración y frontera | 3 | T4 | 0.67 |
| ideología, libertad y valores | 3 | T6 | 1.00 |

Los fallos son interpretables y valen como análisis:

- **economía** se parte entre T6 (Milei: inflación, PBI, déficit) y T3 (AMLO: precios, créditos,
  programas). Es el confundido tema × hablante: cada líder habla de economía con su vocabulario.
- **infraestructura** de Piñera cae en T4 (su tópico "gobierno que protege") porque comparte
  "región/proteger/obra pública" con la agenda social; la de AMLO va a T5 (Tren Maya).
- **democracia** se reparte entre T0 (AMLO: fiscalía, judicial), T8 (Sánchez: Cataluña, parlamento) y
  T6 (Milei: libertad, reforma).
- **vivienda** e **ideología** caen al 100 % en T8 y T6 respectivamente (Sánchez y Milei son casi los
  únicos que hablan de esos temas en el gold).

### 12.3 Estabilidad

5 remuestreos sin reemplazo del 80 % de los documentos, reajustando NMF k = 10 con seeds distintas,
alineando tópicos y midiendo Jaccard entre los top-10:

| corridas | submuestra | Jaccard medio | Jaccard σ | `c_v` σ |
|---|---|---|---|---|
| 5 | 0.80 | **0.858** | 0.175 | 0.0098 |

Por tópico, la estabilidad es alta en los tópicos nítidos (T1 exterior 0.927, T2/T3/T7 0.927, T0 0.908)
y baja en T6 (economía, 0.655) y T5 (infraestructura, 0.733): son los tópicos que compiten por términos
con otros (T6 con T3, T5 con T4). Es exactamente lo que la pureza también señala.

### 12.4 Sensibilidad a `k`

| k | c_v | diversidad | pureza | pureza@2 | anclas |
|---|---|---|---|---|---|
| 6 | 0.198 | 0.917 | 0.638 | 0.766 | 0.344 |
| 8 | 0.331 | 0.925 | 0.511 | 0.681 | 0.406 |
| **10** | **0.443** | **0.940** | **0.553** | **0.660** | **0.469** |
| 12 | 0.444 | 0.925 | 0.532 | 0.638 | 0.469 |
| 15 | 0.492 | 0.920 | 0.468 | 0.553 | 0.531 |

La tabla justifica la regla 1-SE: `k = 15` tiene la mayor coherencia pero la peor pureza (0.468) y
duplica tópicos (dos de exterior, dos de pandemia); `k = 6` tiene la mayor pureza (0.638) pero es
demasiado grueso (anclas 0.344, y tópicos como "seguridad + económica + todo AMLO"). `k = 10` es el
punto de equilibrio y el menor que entra en la banda de indiferencia.

### 12.5 Resumen de la evaluación

| señal | valor | lectura |
|---|---|---|
| `c_v` (k = 10) | 0.443 | en el rango habitual 0.4-0.7 para discursos |
| UMass | -1.058 | coherencia documental razonable (más alto = mejor) |
| diversidad | 0.940 | tópicos casi disjuntos |
| exclusividad | 0.682 | top-10 mayormente exclusivo de su tópico |
| Jaccard estabilidad | 0.858 ± 0.175 | sobrevive a perder el 20 % de documentos |
| anclas | 0.469 | cubre temas, no eventos |
| pureza | 0.553 / 0.660@2 | separa las áreas del gold manual |

---

## 13. Limitaciones

1. **País = persona.** Los tópicos mezclan tema y estilo. Se mitiga con la lista de puesta en escena y
   reportando por hablante, pero no se elimina: "economía" es el tópico de Milei y también el tema de
   Milei. Es una restricción del dataset, no del método.
2. **Desbalance.** Maduro tiene 30 discursos: su modelo es `k = 6`, se marca `low_confidence` y no se
   mezcla en agregados sin nota. Cualquier media del corpus está dominada por AMLO (2,294).
3. **Periodos disjuntos.** La comparación entre líderes se limita a 2023-2024 (restricción 3 del plan).
   Las series largas se leen dentro de cada hablante.
4. **NMF no es probabilístico.** `θ` es una descomposición, no una posterior; "participación" es una
   lectura descriptiva. LDA, que sí lo es, se reporta como contraste y queda por debajo en coherencia
   con el mismo presupuesto de ajuste.
5. **La coherencia depende de la ventana.** `c_v` usa ventana 10; se documenta y se reporta UMass, que
   usa otra unidad (documento), como segunda opinión.
6. **El gold manual es pequeño** (47 discursos, 12 áreas) y se etiquetó con título + inicio del
   discurso. La pureza es una señal, no una verdad; se reporta junto a pureza@2.
7. **Duplicados.** Hay discursos repetidos (módulo 9, pendiente). Inflan ligeramente los tópicos y la
   serie temporal; no se deduplicó aquí para no alterar los `doc_id` de RI.
8. **Ventanas pequeñas.** Piñera 2022 (50 discursos) y Milei 2024-25 (87-69) producen modelos más
   frágiles: el `k` se recorta y la alineación es más ruidosa. Se ve en la tabla de deriva.
9. **Anclas y eventos.** El recall de anclas castiga los eventos puntuales; no es una métrica de
   "detectar la noticia".

---

## 14. Reproducibilidad y coste

```bash
source ../env/bin/activate

python mt_scripts/build_topics.py                  # pipeline completo (~187 s)
python mt_scripts/build_topics.py --force          # reconstruye tokens y mapa stem->superficie
python mt_scripts/build_topics.py \
    --families nmf --k 6,8 --skip-stability        # prueba rápida (~10 s)
```

Todo es determinista: `random_state = 42` en NMF/LDA/SVD, semillas fijas en el remuestreo y orden de
documentos fijo por `doc_id`. Los artefactos intermedios (`tokens.npz`, `stem_surface.json`,
`cooc_npz.npz`) se cachean para que el notebook no recalcule nada.

Coste medido (24 núcleos, sin GPU):

| etapa | tiempo |
|---|---|
| carga del índice + verificación | 2 s |
| tokens + mapa de formas (cacheado) | 0 s (34 s la primera vez) |
| barrido LSA (5 k) | 10 s |
| barrido NMF (5 k) | 33 s |
| barrido LDA (5 k) | 100 s |
| coocurrencia + coherencia | 3 s |
| modelos por hablante (5) | 8 s |
| ventanas + alineación (22 modelos) | 20 s |
| estabilidad (5 reajustes) | 12 s |
| **total** | **187 s** |

---

## 15. Conexión con el resto del proyecto

| módulo | relación |
|---|---|
| 1 — RI | **Base directa**: mismo `doc_id`, vocabulario, IDF y postings. MT no reindexa; verifica que reconstruye `w_global` exactamente. |
| 2 — EI | Los eventos extraídos permiten explicar por qué un tópico sube en un año (p. ej. DANA 2024 no está en los top-10 pero sí en las anclas). |
| 3 — ER | El grafo de entidades da los nombres propios que MT elimina del vocabulario (líderes, países) y que son la mitad del contenido de los tópicos de exterior. |
| 4-5 — RAT | Sus golds anuales (`year_gold.json`) se reutilizan como anclas de evaluación; los resúmenes anuales pueden etiquetarse con los tópicos de MT. |
| 7 — Clasificación y estilometría | Los tópicos son una variable explicativa de estilo; la lista de puesta en escena es, de hecho, un conjunto de features estilométricas. |
| 8 — Polarización | Los tópicos "nosotros/ellos" (patria, casta, pueblo, oligarquía) son la materia prima del módulo 8. |
| 9 — Duplicados | El módulo 9 debería correr antes de MT en una versión final: deduplicar y volver a ajustar. |
| Dashboard | `timeline.json` (hablante × periodo × tópico) y `topics_global.json` son las dos estructuras que consume la vista de evolución. |
