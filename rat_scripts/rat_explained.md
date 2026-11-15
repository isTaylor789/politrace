# Módulos 4 y 5 — Resumen Automático de Textos (RAT)

**Proyecto:** Observatorio del discurso presidencial en español
**Unidades del temario cubiertas:** 4.1 (concepto, retos, unidades), 4.2 (criterios de
extracción), 4.3 (etapas: tópico, unidades, extracción, generación), 4.4 (métodos estadísticos,
selección de oraciones, discursivos y "otros" como LSA), 4.5 (métricas) y 4.6 (multi-documento).
El resumen de **un** documento está en §§3-11; el resumen **multi-documento por año** (módulo 5 del
plan, también 4.6) se detalla en **§12**.

**Restricciones respetadas:** cero IA generativa, cero deep learning, cero transformers y cero
embeddings neuronales. Todo el resumidor es determinista y auditable: IDF, coseno, SVD truncada,
PageRank, MMR y reglas léxicas.

---

## 1. Qué hace y qué no hace

Toma un texto (un discurso del corpus o un texto cualquiera) y devuelve un **resumen extractivo**:
una selección ordenada de oraciones del propio texto, sin reescribirlas. La calibración busca un
resumen **general y legible**, no el de máxima compresión:

| Objetivo | Valor |
|---|---|
| Compresión típica en discursos de 400-900 palabras | 12-20 % |
| Palabras del resumen | 60-180 |
| Oraciones | 3-12 (presupuesto duro de 180 palabras) |
| Cobertura del tópico (definida en §10.3) | 0.69-0.83 |
| Resumen de un año (módulo 5, §12) | ~390 palabras, 10-12 oraciones, 5-8 temas |

No se hace generación abstractiva (no hay modelo de lenguaje) ni se detecta "la noticia del día":
eso queda fuera del alcance de los métodos simbólicos usados.

---

## 2. Arquitectura

```
texto del discurso
   │
   ├─ 1. SEGMENTACIÓN (ei_scripts.segment: punkt + cortes por ; : ,)
   │      └─ 2. FILTRADO DE UNIDADES (4.2): 7 ≤ palabras ≤ 75, ≥ 2 términos,
   │         sin ruido de transcripción, sin repetidas
   │
   ├─ 3. REPRESENTACIÓN TF-IDF con el IDF GLOBAL del índice de RI
   │      └─ X (oraciones × términos), IDF, W normalizada L2, matriz de coseno
   │
   ├─ 4. PUNTUACIÓN (4.4)
   │      estadísticos: TF-IDF (cabeza), centroide, Luhn
   │      grafo:        TextRank (PageRank) y LexRank (grado)
   │      latente:      LSA (SVD truncada)
   │      discursivos:  posición + conectores + título + cifras − penalizaciones
   │
   ├─ 5. COMBINACIÓN: z-score recortado + media ponderada (pesos en `config.py`)
   │
   ├─ 6. SELECCIÓN: MMR (λ = 0.7), filtro de casi-duplicados, presupuesto de palabras
   │      └─ 7. GENERACIÓN: reordenar por posición original y unir
   │
   ├─ 8. MÉTRICAS: ROUGE-1/2/L contra referencias humanas + compresión, cobertura y redundancia
   │
   └─ 9. MULTI-DOCUMENTO POR AÑO (módulo 5, §12): agrupar discursos (k-means sobre coseno) →
      resumir medoides → fusionar con cuotas por grupo → tópicos comunes
```

Archivos:

| Archivo | Contenido |
|---|---|
| `config.py` | rutas, parámetros y pesos |
| `units.py` | segmentación, limpieza, filtros y `Unit` |
| `representation.py` | TF-IDF, IDF global/local, coseno |
| `scorers.py` | TF-IDF, centroide, Luhn, TextRank, LexRank, LSA, híbrido |
| `features.py` | pistas discursivas y penalizaciones |
| `select.py` | MMR, casi-duplicados, render |
| `summarize.py` | API: `summarize_text`, `summarize_doc`, `summarize_docs` |
| `evaluate.py` | ROUGE-N/L, líneas base, banco de pruebas |
| `build_summaries.py` | CLI que genera `artifacts/` |
| `multi.py` | módulo 5: agrupamiento, fusión y `summarize_year` |
| `evaluate_multidoc.py` | ROUGE + recall de tópicos + variantes del módulo 5 |
| `build_multidoc.py` | CLI de artefactos del módulo 5 |
| `gold/summaries_gold.json` | 5 referencias por discurso (módulo 4) |
| `gold/year_gold.json` | 4 casos anuales con referencia y tópicos clave (módulo 5) |

---

## 3. Unidades de resumen (4.2, 4.3.2)

### 3.1 Segmentación y limpieza

Se reutiliza el segmentador de EI (`punkt` para español + corte por `;` `:` `,` cuando una oración
supera 90 palabras). Sobre cada oración se aplica una limpieza determinista:

1. se eliminan marcas invisibles de dirección de texto (`U+200B`-`U+200F`, `U+FEFF`);
2. se recortan etiquetas de hablante: `PRESIDENTE. –`, `MODERADOR:`, `GRACIELA MÁRQUEZ COLÍN,
   SECRETARIA DE ECONOMÍA:`;
3. se recortan cabeceras de transcripción: `Palabras del Presidente … :`, `Discurso del … :`,
   `~~~PRESIDENTE ANDRÉS MANUEL LÓPEZ OBRADOR 2023:`;
4. se recortan saludos y vocativos iniciales: `Muy buenas tardes:`, `Amigas, amigos de Querétaro:`,
   `Excelencias,`;
5. se recortan muletillas orales iniciales (`Este,`, `Bueno,`, `Pues,`, `O sea,`).

### 3.2 Criterios de extracción

Una oración es **unidad candidata** si:

- `7 ≤ palabras ≤ 75` (por debajo no aporta; por encima no se lee dentro de un resumen),
- tiene al menos 2 términos después de quitar stopwords y aplicar stemming Snowball,
- no es ruido: aplausos, risas, cortes técnicos, fechas sueltas, `Muchas gracias a todas y a todos.`,
  pies de página (`Publicado por … en Viernes, 31 de enero de 2020`),
- no está repetida (misma secuencia de términos que otra).

Si el filtro estricto deja menos de 8 oraciones (textos cortos o mal puntuados) se relaja a
`palabras ≥ 4` y sin tope superior. Para textos con menos de 12 unidades el mínimo de oraciones del
resumen baja a `max(1, ⌊n/4⌋)` (en vez de 3) para no devolver casi el texto completo.

### 3.3 Posición y párrafo

Cada unidad guarda su posición relativa `p ∈ [0,1]` (0 = primera oración, 1 = última) y el índice
de párrafo, detectado por líneas en blanco. La posición alimenta el prior discursivo (§5.6); el
párrafo se reporta para trazabilidad.

---

## 4. Representación vectorial (4.4.1)

### 4.1 IDF

Se reutiliza el IDF del índice de RI, calculado sobre los 4,531 discursos:

$$\text{idf}_j = \ln\!\left(\frac{1 + N}{1 + \text{df}_j}\right) + 1,
\qquad N = 4531,\quad \text{df}_j = \#\{\text{docs con el término } j\}$$

Un término fuera del vocabulario global (aparece en un solo documento, `min_df = 2`) recibe el IDF
máximo observado; es la estimación correcta para algo "muy raro". Para textos que **no** están en el
corpus el IDF se calcula localmente sobre las oraciones del propio texto:

$$\text{idf}^{\text{loc}}_j = \ln\!\left(\frac{1 + n}{1 + \text{df}_j}\right) + 1$$

donde `n` es el número de oraciones del texto.

### 4.2 Matriz y coseno

Con conteos `tf_ij` y filas normalizadas:

$$w_{ij} = \frac{\text{tf}_{ij}\cdot \text{idf}_j}
{\sqrt{\sum_{j'} \text{tf}_{ij'}^2\, \text{idf}_{j'}^2}},\qquad
\cos(i,k) = \langle w_i, w_k\rangle$$

La matriz densa de similitud `sim(i,k) = cos(i,k)` se usa en TextRank, LexRank y MMR. Con cientos de
oraciones (máximo ~600 en las mañaneras) es barata: `O(n²)` en memoria, 0.64 s por discurso de 18k
palabras.

---

## 5. Puntuadores (4.4)

Todos devuelven un vector de longitud `n` (número de oraciones). Después se normalizan (§6).

### 5.1 TF-IDF de cabeza

$$s_{\text{tfidf}}(i) = \frac{1}{K}\sum_{j \in \text{top-}K(i)} w_{ij},\qquad K = 5$$

Promediar la cabeza de la distribución (en vez de sumar todo) evita que una oración gane sólo por
ser larga.

### 5.2 Centroide

$$c = \frac{\frac{1}{n}\sum_i w_i}{\left\lVert \frac{1}{n}\sum_i w_i\right\rVert},
\qquad s_{\text{cent}}(i) = \langle w_i, c\rangle$$

Mide cuánto se parece la oración al "promedio" del discurso.

### 5.3 Luhn

Una palabra es **significativa** si su frecuencia documental (número de oraciones donde aparece)
cae en la banda

$$\left\lceil 0.05\,n \right\rceil \;\le\; \text{df}_t \;\le\; \left\lfloor 0.70\,n \right\rfloor$$

Por encima del 70 % es una palabra ubicua (no distingue); por debajo del 5 % es un hápax (ruido).
El peso de cada palabra significativa es su IDF y la oración se puntúa por densidad:

$$s_{\text{luhn}}(i) = \frac{\sum_{t \in \text{sig}(i)} \text{idf}_t}
{\sqrt{\text{span}_i}}$$

donde `span_i` = distancia, en términos, entre la primera y la última palabra significativa de la
oración (dividir por el span premia las oraciones con las palabras clave concentradas).

### 5.4 TextRank

Grafo completo de oraciones con arista ponderada `sim(i,k)`. Se resuelve PageRank por iteración de
potencias:

$$p^{(t+1)} = \frac{1-d}{n}\mathbf{1} + d\,M^\top p^{(t)},\qquad d = 0.85$$

`M` es la matriz de transición por filas (`M_ik = sim(i,k) / Σ_k sim(i,k)`); las filas sin aristas
(nodos colgantes) se reparten uniformemente. Se itera hasta que `‖p^{(t+1)} − p^{(t)}‖₁ < 10⁻⁸`
(máximo 300 iteraciones).

### 5.5 LexRank

Misma idea pero grafo **binarizado** y centralidad de grado:

$$A_{ik} = \mathbb{1}\left[\cos(i,k) \ge \theta\right],\quad \theta = 0.10,
\qquad s_{\text{lex}}(i) = \frac{\deg(i)}{n-1}$$

Es la variante "de grado" del LexRank original (la versión con PageRank sobre `A` da casi lo mismo
porque `A` es simétrica).

### 5.6 Componente discursivo

Tres señales, con pesos internos `0.50 / 0.35 / 0.15`:

**a) Prior de posición** (el planteamiento está al inicio; el cierre, al final):

$$\pi(p) = 0.85\cdot\max\!\left(0, 1-\frac{p}{0.30}\right)
+ 0.15\cdot\max\!\left(0, 1-\frac{1-p}{0.12}\right)$$

**b) Conectores y fórmulas** (cada categoría cuenta una vez):

| Categoría | Peso | Ejemplos |
|---|---:|---|
| síntesis | 1.00 | en resumen, en conclusión, en definitiva, por último, para finalizar |
| compromiso | 0.95 | vamos a, me comprometo, anuncio que, proponemos, tenemos que, hay que |
| relevancia | 0.65 | es importante, lo fundamental, la prioridad, quiero destacar |
| causalidad | 0.60 | por lo tanto, en consecuencia, por eso, debido a, gracias a |
| gobierno | 0.55 | mi gobierno, hemos logrado, hemos hecho, nuestro pueblo |

La puntuación satura para que cinco conectores no valgan cinco veces uno:

$$\text{cue}(i) = 1 - e^{-\sum_{c} w_c\,\mathbb{1}[\,\text{categoría } c \text{ presente}\,]}$$

**c) Solapamiento con el título** (el título ya declara el tópico, 4.3.1):

$$\text{ttl}(i) = \frac{|T \cap \text{términos}(i)|}{|T|},\qquad T = \text{términos del título}$$

Con eso:

$$\text{disc}(i) = 0.88\left[0.50\,\pi(p_i) + 0.35\,\text{cue}(i) + 0.15\,\text{ttl}(i)\right]
+ 0.12\,\min\!\left(1, \frac{4\cdot \#\text{cifras}_i}{\#\text{palabras}_i}\right)$$

y finalmente **penalizaciones multiplicativas** de legibilidad:

| Situación | Factor |
|---|---:|
| empieza a media idea (`Y…`, `Pero…`, `Que…`, `Muestra que…`) | ×0.75 |
| cortesía / despedida (`muchas gracias`, `que Dios`, `feliz Navidad`, `quiero agradecer`) | ×0.55 |
| fórmula hueca (`es una cosa`, `les quiero decir`, `como ustedes saben`) | ×0.80 |

Las penalizaciones no descartan la oración: si no hay alternativa, puede seguir siendo elegida.

### 5.7 LSA

SVD truncada de la matriz TF-IDF:

$$W \approx U_k\Sigma_k V_k^\top,\qquad
k = \min\!\left(100,\ \lceil 0.30\,n\rceil,\ n-1,\ V-1\right)$$

La cota proporcional es necesaria: con 14 oraciones una SVD de 13 componentes reconstruye la matriz
casi entera y la puntuación se vuelve ruido (empíricamente, ROUGE-1 caía de 0.28 a 0.13). La
puntuación de la oración `j`:

$$s_{\text{lsa}}(j) = \frac{\sum_{i=1}^{k}\sigma_i\,u_{ji}^2}{\sum_{i=1}^{k}\sigma_i} \in [0,1]$$

Es la masa de la oración en los componentes, ponderada por el valor singular.

---

## 6. Combinación híbrida

Cada componente se estandariza de forma robusta y se reescala a `[0,1]`:

$$z_c(i) = \text{clip}\!\left(\frac{x_c(i)-\mu_c}{\sigma_c}, -3, 3\right),\qquad
\hat z_c(i) = \frac{z_c(i)-\min z_c}{\max z_c - \min z_c}$$

El recorte a ±3σ impide que un único valor atípico aplaste al resto. La combinación es una media
ponderada (los pesos suman 1):

$$\boxed{S(i) = \sum_c \lambda_c\,\hat z_c(i)}$$

| Componente | Peso λ |
|---|---:|
| centroide | 0.28 |
| TextRank | 0.24 |
| discursivo | 0.26 |
| LSA | 0.10 |
| Luhn | 0.12 |

Los pesos se fijaron con la ablación de §10.5 y con una búsqueda pequeña sobre las 5 referencias.
LSA se deja en 0.10: es el componente con menos influencia (subirlo a 0.16 no cambia las oraciones
elegidas en los 5 discursos) y aporta señal independiente (r ≈ 0.15 con el centroide).

---

## 7. Selección y generación (4.3.3, 4.3.4)

### 7.1 Número de oraciones

$$k = \text{clip}\left(\left\lceil 0.15\,n\right\rceil,\; m,\; 12\right),
\qquad m = \max\!\left(1, \min\!\left(3, \left\lfloor n/4\right\rfloor\right)\right)$$

y se corta antes si se supera el presupuesto de **180 palabras** (no se corta si aún no se llegó al
mínimo `m`).

### 7.2 MMR

Selección greedy (Carbonell & Goldstein, 1998):

$$\text{MMR}(i \mid S) = \lambda\,\text{rel}(i) - (1-\lambda)\max_{j \in S}\cos(i,j),
\qquad \lambda = 0.70$$

con `rel` = `S` normalizado a `[0,1]` (min-max). En cada paso se elige el argmax entre las no
seleccionadas. Se añaden dos salvaguardas:

1. **casi-duplicados**: si `Jaccard(términos(i), términos(j)) > 0.80` para algún `j ∈ S`, la oración
   se descarta y se prueba la siguiente;
2. **presupuesto**: si añadir la oración excede las 180 palabras (y ya hay `m` oraciones), se
   descarta y se prueba la siguiente, en vez de detenerse (así no se pierden oraciones cortas
   buenas).

Para el resumen multi-documento se añade una **cuota por documento**: como máximo
`⌈k / n_docs⌉` oraciones de cada discurso.

### 7.3 Generación

Las oraciones elegidas se reordenan por su posición original en el discurso, se asegura puntuación
final y se unen con un espacio. No se reescribe nada.

---

## 8. Resumen multi-documento simple (4.6)

> Esta sección describe `summarize_docs`, la fusión mínima para listas pequeñas de discursos
> (5-10). La versión completa del módulo 5 —agrupamiento temático, cuotas proporcionales, tópicos
> comunes y evaluación— está en **§12**.

Problema: resumir 129 discursos de un año en 200 palabras. Promediar todo produce una sopa de
oraciones inconexas y un discurso largo acapara la selección. Estrategia de **fusión**
(*summarize-then-fuse*):

1. **Selección de documentos**: se eligen los `k_docs = 5` discursos más cercanos al centroide del
   conjunto, usando las filas de `w_global` de RI:
   $$\text{sim}(d) = \left\langle w_d,\ \frac{\bar w}{\lVert \bar w\rVert}\right\rangle,
   \qquad \bar w = \frac{1}{|D|}\sum_{d\in D} w_d$$
2. **Resumen individual**: cada discurso se resume con el pipeline completo y
   `k = 3` oraciones.
3. **Fusión**: las oraciones de esos resúmenes se vuelven a puntuar juntas (misma representación,
   mismo IDF global, mismo híbrido) y se seleccionan con MMR, con
   $$k_{\text{multi}} = \min\!\left(12,\ \max\!\left(5, \left\lceil 0.55\,|U|\right\rceil\right)\right),
   \quad \text{presupuesto} = 225 \text{ palabras},$$
   y cuota por documento.
4. **Tópicos comunes**: términos presentes en al menos `⌈0.6·n_docs⌉` discursos, ordenados por
   número de discursos y luego por frecuencia total. Es la "identificación de tópico común" de 4.6.

---

## 9. API

```python
from rat_scripts import summarize_text, summarize_doc, summarize_docs, summarize_year

resumen = summarize_text(mi_texto, title="...", method="hybrid")   # texto cualquiera
resumen = summarize_doc(1065)                                      # discurso del corpus
resumen = summarize_docs(doc_ids_de_2023, k_docs=5)                # fusión simple
anual = summarize_year(ix, 2023, "Pedro Sánchez")                  # módulo 5 (§12)

print(resumen.text)
print(resumen.stats)        # compresión, cobertura, redundancia, ...
```

Métodos disponibles: `hybrid` (por omisión), `centroid`, `tfidf`, `luhn`, `textrank`, `lexrank`,
`lsa`, `discourse` y `lead` (línea base: primeras k oraciones).

CLI:

```bash
source ../env/bin/activate
python rat_scripts/build_summaries.py     # módulo 4
python rat_scripts/build_multidoc.py      # módulo 5
# -> rat_scripts/artifacts/{eval.json, summaries_demo.jsonl, tables.md}
# -> rat_scripts/artifacts/multidoc/{eval_multidoc.json, summaries_year.jsonl, tables_multidoc.md}
```

---

## 10. Evaluación y resultados (4.5)

### 10.1 Referencias

No existía un corpus de resúmenes humanos, así que se escribieron **5 referencias** (una por
hablante, 62-77 palabras cada una) sobre discursos de 400-440 tokens: `doc_id` 1065, 2381, 2493,
2877 y 4432 (`gold/summaries_gold.json`). Las referencias son abstractivas (parafrasean), mientras
el sistema es extractivo; por eso ROUGE queda en 0.28, no en 0.6. Sirven para **comparar métodos**,
no para certificar calidad absoluta.

### 10.2 ROUGE

ROUGE-N con conteo recortado (*clipping*, Lin 2004) sobre términos analizados (stem, sin stopwords):

$$\text{ROUGE-N} = \sum_{g \in \text{n-gramas}(r)} \min\left(c_p(g), c_r(g)\right),
\qquad
P = \frac{\text{ROUGE-N}}{|p|},\quad R = \frac{\text{ROUGE-N}}{|r|}$$

$$F_1 = \frac{2PR}{P+R}$$

ROUGE-L usa la longitud `LCS(p, r)`:

$$P_L = \frac{\text{LCS}}{|p|},\qquad R_L = \frac{\text{LCS}}{|r|},\qquad F_L = \frac{2P_LR_L}{P_L+R_L}$$

### 10.3 Métricas intrínsecas (sin referencia)

- **Compresión** = palabras del resumen / palabras del original.
- **Cobertura** = fracción del peso TF-IDF de los 20 conceptos principales del documento que aparece
  en el resumen:
  $$\text{cov} = \frac{\sum_{t \in \text{top-}20} w_t\,\mathbb{1}[t \in \text{resumen}]}
  {\sum_{t \in \text{top-}20} w_t}$$
- **Redundancia** = coseno medio entre pares de oraciones seleccionadas.

### 10.4 Resultados por método

Media de los 5 discursos (misma `k` y mismo presupuesto para todos; `random` promedia 3 semillas):

| Método | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Compresión | Cobertura | Redundancia | Pal/oración |
|---|---:|---:|---:|---:|---:|---:|---:|
| lead (primeras k) | 0.189 | 0.049 | 0.129 | 0.151 | 0.565 | 0.042 | 27.8 |
| tfidf | 0.171 | 0.049 | 0.130 | 0.090 | 0.478 | 0.031 | 17.9 |
| centroide | 0.236 | 0.086 | 0.177 | 0.194 | 0.713 | 0.229 | 35.9 |
| Luhn | 0.232 | 0.059 | 0.148 | 0.202 | 0.723 | 0.119 | 41.0 |
| TextRank | 0.236 | 0.086 | 0.177 | 0.194 | 0.713 | 0.229 | 35.9 |
| LexRank | 0.266 | 0.096 | 0.188 | 0.176 | 0.745 | 0.153 | 33.9 |
| LSA | 0.176 | 0.041 | 0.119 | 0.130 | 0.556 | 0.018 | 24.8 |
| **híbrido** | **0.285** | **0.110** | **0.202** | 0.189 | **0.756** | 0.156 | 35.8 |
| random | 0.181 | 0.042 | 0.118 | — | — | — | — |

Lectura:

- El híbrido gana en las tres variantes de ROUGE y en cobertura, con compresión comparable a las
  líneas base.
- TextRank y centroide dan **exactamente** los mismos números: sus puntuaciones correlacionan
  r = 0.99 y seleccionan las mismas oraciones en los 5 discursos.
- LSA y TF-IDF solos son débiles; TF-IDF de cabeza premia oraciones cortas con una palabra rara
  (en el discurso del hospital elige "Muchas gracias a todas y a todos", que además ya se filtra
  como cortesía).
- `lead` no es un rival trivial: en discursos protocolares el inicio resume mucho. El híbrido le
  gana en los 5 documentos, pero en Milei por sólo 0.004 de ROUGE-1.

**Por documento (híbrido vs lead):**

| doc_id | Hablante | lead R1 | híbrido R1 | híbrido R2 | híbrido RL | Cobertura |
|---:|---|---:|---:|---:|---:|---:|
| 1065 | AMLO | 0.157 | 0.389 | 0.189 | 0.296 | 0.795 |
| 2381 | Milei | 0.222 | 0.226 | 0.077 | 0.132 | 0.709 |
| 2493 | Maduro | 0.063 | 0.194 | 0.045 | 0.134 | 0.693 |
| 2877 | Sánchez | 0.235 | 0.318 | 0.154 | 0.258 | 0.826 |
| 4432 | Piñera | 0.267 | 0.298 | 0.087 | 0.191 | 0.757 |

### 10.5 Ablación

Quitar un componente del híbrido y renormalizar los pesos restantes:

| Variante | ROUGE-1 | ROUGE-2 | ROUGE-L | Cobertura |
|---|---:|---:|---:|---:|
| completo | 0.285 | 0.110 | 0.202 | 0.756 |
| sin centroide | 0.296 | 0.094 | 0.184 | 0.743 |
| sin TextRank | 0.296 | 0.094 | 0.184 | 0.743 |
| sin discursivo | 0.247 | 0.093 | 0.175 | 0.764 |
| sin LSA | 0.277 | 0.110 | 0.202 | 0.749 |
| sin Luhn | 0.270 | 0.108 | 0.205 | 0.745 |

El componente **discursivo** es el que más aporta (sin él ROUGE-L cae 13 % relativo y ROUGE-2 un
15 %). Centroide y TextRank son redundantes entre sí (quitar cualquiera da exactamente la misma
tabla). Luhn aporta poco pero estabiliza; LSA casi no mueve ROUGE y se conserva por la cobertura
conceptual del temario (4.4.6).

Sensibilidad observada (media de los 5 discursos):

| λ de MMR | ROUGE-1 | ROUGE-2 | ROUGE-L | Cobertura | Redundancia |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 0.280 | 0.097 | 0.187 | 0.790 | 0.100 |
| 0.5 | 0.284 | 0.097 | 0.187 | 0.765 | 0.125 |
| 0.6-0.7 | 0.285 | 0.110 | 0.202 | 0.756 | 0.156 |
| 0.8 | 0.285 | 0.114 | 0.203 | 0.729 | 0.175 |
| 0.9 | 0.263 | 0.107 | 0.201 | 0.718 | 0.193 |

| ratio | ROUGE-1 | ROUGE-2 | ROUGE-L | Cobertura | Oraciones |
|---:|---:|---:|---:|---:|---:|
| 0.08 | 0.290 | 0.097 | 0.199 | 0.647 | 3.0 |
| 0.12 | 0.295 | 0.108 | 0.199 | 0.729 | 4.0 |
| **0.15** | 0.285 | 0.110 | 0.202 | 0.756 | 4.8 |
| 0.18 | 0.283 | 0.114 | 0.205 | 0.784 | 5.2 |
| 0.22 | 0.281 | 0.110 | 0.199 | 0.801 | 5.6 |

Correlación entre componentes (Pearson medio sobre los 5 discursos):

|  | centroide | TextRank | LexRank | LSA | Luhn | discursivo |
|---|---:|---:|---:|---:|---:|---:|
| centroide | 1.00 | 0.99 | 0.92 | 0.15 | 0.65 | −0.04 |
| TextRank | 0.99 | 1.00 | 0.91 | 0.18 | 0.65 | −0.04 |
| LexRank | 0.92 | 0.91 | 1.00 | 0.09 | 0.57 | −0.01 |
| LSA | 0.15 | 0.18 | 0.09 | 1.00 | 0.28 | −0.00 |
| Luhn | 0.65 | 0.65 | 0.57 | 0.28 | 1.00 | −0.00 |
| discursivo | −0.04 | −0.04 | −0.01 | −0.00 | −0.00 | 1.00 |

El bloque discursivo es ortogonal al resto: por eso aporta aunque su peso sea sólo 0.26.

### 10.6 Multi-documento

Pedro Sánchez, 2023 (129 discursos disponibles). Documentos elegidos: 3225, 3157, 3278, 3271 y 3193.
Resultado: 5 oraciones (222 palabras) a partir de 15 oraciones fusionadas, compresión total 0.75 %
sobre 29,594 palabras, cobertura 0.76, redundancia 0.23; reparto por discurso 1/1/2/0/1 (la cuota
es un tope, no un mínimo). Tópicos comunes: `europe` (5 discursos), `españ` (5), `cre` (5), `pais`
(5), `president` (5), `union` (5) — coherente con el año de la presidencia española de la UE.

### 10.7 Coste

| Tipo de discurso | Tiempo por discurso | Compresión | Cobertura |
|---|---:|---:|---:|
| ≤ 900 tokens (n=40) | 33 ms | 0.124 | 0.733 |
| 1k-3k tokens (n=30) | 149 ms | 0.046 | 0.723 |
| > 6k tokens (n=10) | 430 ms | 0.013 | 0.681 |

El coste lo domina la matriz de similitud `O(n²)` y la SVD `O(n·V·k)`; ambos son aceptables para
los discursos del corpus (máximo ~600 oraciones).

---

## 11. Decisiones de calibración

| Decisión | Valor | Razón |
|---|---:|---|
| Compresión objetivo | 15 % | Con 12 % ROUGE-1 sube 0.010 pero la cobertura cae de 0.756 a 0.729 y el resumen baja a 4 oraciones; con 18-22 % el resumen se alarga sin ganar ROUGE-1. |
| Tope de oraciones | 12 | Evita que un discurso de 600 oraciones produzca una lista. |
| Presupuesto | 180 palabras | Un "resumen general" de un discurso de 400-900 palabras ocupa 60-165 palabras. |
| λ de MMR | 0.70 | Entre 0.4 y 0.8 el ROUGE-1 varía sólo 0.005; con λ=0.9 la redundancia sube de 0.156 a 0.193 y ROUGE-1 cae a 0.263. 0.70 maximiza ROUGE-2 sin disparar redundancia. |
| Jaccard de duplicados | 0.80 | Detecta pares casi literales de las mañaneras sin borrar paráfrasis. |
| LSA | k ≤ 0.30·n, peso 0.10 | Sin la cota proporcional, LSA era contraproducente en documentos cortos (R1 0.13 → 0.20 al corregirla). Con peso 0.10 su efecto es marginal. |
| IDF | global de RI | Coherente con el buscador y evita recalcular 4,531 documentos. |

La búsqueda de pesos se hizo sobre las mismas 5 referencias, así que estos números están algo
sobreajustados a esa muestra. Los valores elegidos son redondos y las diferencias entre variantes
razonables son pequeñas (0.01-0.02 de ROUGE), así que no se persiguió el óptimo.

---

## 12. Módulo 5 — resumen multi-documento por año (4.6)

Implementación completa de 4.6 en `rat_scripts/multi.py`, con evaluación propia en
`evaluate_multidoc.py` y artefactos en `artifacts/multidoc/`.

### 12.1 Problema

Un año de un hablante son 87-308 discursos y 100.000-320.000 palabras. El reto (4.6.1) no es sólo
comprimir: hay que **elegir de qué se habla** (agrupar), **no repetir** (MMR), **no dejar que un
discurso o un tema acapare** (cuotas) y **detectar el tópico común** (frecuencia). El resultado es
un resumen de ~390 palabras, 10-12 oraciones y 5-8 temas.

### 12.2 Pipeline

```
discursos del año (y hablante, si se indica)
  │
  ├─ 1. AGRUPAMIENTO: k-means sobre las filas L2 de w_global (coseno)
  │      k = clip(round(√(n/4)), 3, 8)  ·  grupos de 1 doc → al centroide más cercano
  │      etiquetas: df·idf dentro del grupo, sin términos genéricos del año ni números
  │
  ├─ 2. RESUMEN INDIVIDUAL: 3 medoides por grupo → 2 oraciones cada uno
  │      (posición neutralizada: p = 0.5)
  │
  ├─ 3. FUSIÓN: representación TF-IDF de las oraciones (IDF global)
  │      híbrido del módulo 4 + penalizaciones de trámite/ceremonia
  │      MMR (λ = 0.7) con CUOTA proporcional por grupo y presupuesto de 400 palabras
  │
  └─ 4. TÓPICOS COMUNES: términos del tema de varios grupos, ordenados por
         nº de grupos y peso df·idf
```

### 12.3 Agrupamiento

Vectores de documento: las filas de `w_global` (TF-IDF con IDF del corpus, norma L2 = 1). Para
vectores unitarios,

$$\lVert u-v\rVert^2 = 2 - 2\cos(u,v),$$

así que k-means con distancia euclidiana maximiza el coseno (k-means esférico):

$$\min_{\{\mu_c\}} \sum_{c}\sum_{d\in c}\lVert w_d-\mu_c\rVert^2 .$$

Número de grupos:

$$k = \text{clip}\left(\text{round}\left(\sqrt{n/4}\right),\;3,\;8\right).$$

Con 129 discursos → 6 grupos; 308 → 8; 87 → 5. Los grupos de un solo documento se reasignan al
centroide más cercano (no se resume un discurso suelto como si fuera un tema).

**Etiquetas de grupo.** Para cada grupo `g` se calcula, por término,

$$\text{peso}_g(t) = \text{df}_g(t)\cdot \text{idf}(t),$$

donde `df_g(t)` = nº de documentos del grupo que contienen `t`. Se excluyen los términos
**genéricos del año** (aparecen en ≥ 40 % de los documentos: `president`, `gobiern`, el nombre del
hablante, la plantilla de la mañanera) y los que contienen dígitos. Se usan `df` y no `tf` porque
un discurso muy largo repitiendo una palabra dominaba el perfil del año; `df·idf` mide "frecuente
en el conjunto y distintivo en el corpus".

**¿IDF global o matriz por presidente?** Se usa el IDF **global** de RI, no una matriz por
presidente:

1. el resumen anual puede cruzar hablantes (2023-2024 tiene 4 líderes activos) y con IDF por
   hablante la similitud entre oraciones de dos presidentes no es comparable;
2. estimar el IDF sobre los documentos que se están resumiendo es circular, y con 30 documentos
   (Maduro) el rango de IDF se comprime y dominan los términos exclusivos;
3. la matriz global ya es barata y estable (N = 4,531, `min_df = 2`).

Medido en los 4 casos anuales, las tres opciones de IDF para las **oraciones** dan resultados
equivalentes en recall de tópicos (global 0.562, mezclado 0.562, local 0.562) y el global es igual o
mejor en ROUGE (§12.5). El IDF **de cluster** se puede reconstruir desde `postings.npz` sin
re-tokenizar (se usó para las etiquetas y los tópicos), pero no mejora la selección.

### 12.4 Resúmenes individuales y fusión

**Medoides.** Para cada grupo se eligen los documentos más cercanos a su centroide:

$$d^\* = \operatorname*{arg\,max}_{d\in g}\ \cos(w_d,\ \mu_g),\qquad
\mu_g = \frac{\sum_{d\in g} w_d}{\lVert \sum_{d\in g} w_d\rVert}.$$

Se toman 3 medoides por grupo y se resume cada uno con 2 oraciones (mismo pipeline del módulo 4).
La posición se fija en `p = 0.5` para anular el prior de inicio/fin: el comienzo de un discurso
medoide no es más importante que su desarrollo.

**Cuotas.** Sea `k = min(14, max(n_grupos, ⌈0.45·|U|⌉))` el objetivo de oraciones y `n_g` el tamaño
del grupo:

$$q_g = \left\lfloor \frac{k\,n_g}{n}\right\rfloor
\quad\text{con}\quad
q_g = \max(1,q_g)\ \text{si } \frac{n_g}{n}\ge 0.05,
\qquad q_g = 0\ \text{si } \frac{k\,n_g}{n} < 0.5,$$

y los lugares restantes se reparten por **mayor resto** (método de Hare) hasta sumar `k`. Así el
tema mayoritario aporta más oraciones, los temas significativos tienen al menos una y los
anecdóticos (una ceremonia de 3-4 discursos) no ocupan un lugar fijo. La selección es MMR con
`λ = 0.7`, filtro de casi-duplicados (Jaccard 0.8) y presupuesto de 400 palabras.

**Tópicos comunes.** Cada grupo aporta su conjunto de 30 términos con mayor `df·idf`; un término es
"común" si aparece en ≥ 2 grupos y se ordena por nº de grupos y luego por su peso `df·idf` en el año.

### 12.5 Evaluación

Cuatro casos anuales con **referencia humana** (~70 palabras) y **7-8 tópicos clave** cada uno
(`gold/year_gold.json`): Sánchez 2023 y 2024, AMLO 2023 y Milei 2024. Métricas:

* **ROUGE-1/2/L** (definidas en §10.2). Ojo: el resumen tiene ~390 palabras y la referencia ~70, así
  que el F1 está penalizado por longitud; sirve para **comparar variantes**, no como valor absoluto.
* **Recall de tópicos clave**: `|tópicos cubiertos| / |tópicos|`. Un tópico puede exigir varios
  stems a la vez (`all`) y aceptar alternativas (`any`), p. ej. `{all: [europe], any: [rotatori, consej]}`.
* **Similitud con el centroide del año**: coseno entre el centroide de las oraciones elegidas y el
  centroide de los documentos del año (ambos en TF-IDF global, sin términos genéricos). Valores
  bajos por construcción (el resumen es una nube dispersa frente a un centroide ancho); se usa para
  comparar.
* **Cobertura temática**: proporción de documentos del año cuyo grupo temático tiene ≥ 1 oración en
  el resumen (ponderada por tamaño).
* Compresión, redundancia y longitud.

**Resultados (media de los 4 casos):**

| Variante | ROUGE-1 | ROUGE-2 | ROUGE-L | Tópicos | Sim. centroide | Temas | Palabras | Redundancia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **base (IDF global)** | 0.104 | 0.031 | 0.058 | 0.562 | 0.204 | 0.959 | 390 | 0.067 |
| IDF mezclado (α=0.5) | 0.105 | 0.030 | 0.060 | 0.562 | 0.205 | 0.959 | 392 | 0.056 |
| IDF local | 0.100 | 0.030 | 0.059 | 0.562 | 0.208 | 0.959 | 392 | 0.049 |
| sin agrupamiento (1 grupo) | 0.095 | 0.021 | 0.073 | 0.344 | 0.147 | 1.000 | 345 | 0.144 |
| sin cuotas | 0.094 | 0.023 | 0.061 | 0.406 | 0.230 | 0.647 | 393 | 0.089 |
| cuotas mínimas (1 por grupo) | 0.113 | 0.033 | 0.065 | 0.625 | 0.203 | 0.991 | 390 | 0.063 |
| aleatorio (mismo material) | 0.081 | 0.015 | 0.046 | 0.510 | 0.236 | 0.908 | 396 | 0.038 |

Lectura:

* **El agrupamiento es lo que más aporta**: sin él, el recall de tópicos cae de 0.56 a 0.34, porque
  el presupuesto se llena con muchas oraciones del mismo tema. Las cuotas también importan:
  quitarlas baja el recall a 0.41 y la cobertura temática a 0.65.
* Las tres opciones de IDF empatan en recall; el global es el más simple y queda como opción por
  omisión.
* El **azar** sobre el mismo material cubre tópicos frecuentes (0.51) porque el material ya viene
  filtrado por grupo y medoide, pero produce resúmenes sin estructura y peor ROUGE (0.081 vs 0.104).
  Es un suelo exigente, no un rival trivial.
* Las **cuotas mínimas** (todas las temas con al menos 1 oración) ganan en las métricas (0.625), pero
  meten alguna oración de ceremonia (una entrega de medallas en Sánchez 2024). Se prefiere la
  variante proporcional con umbral del 5 %: pierde tópicos de 3-13 discursos (Marruecos, Acapulco)
  pero el resumen se lee mejor.

**Detalle por caso (variante base):**

| Caso | Grupos | Docs usados | Oraciones | Palabras | ROUGE-1 | ROUGE-L | Tópicos | Temas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sánchez 2023 | 6 | 9 | 11 | 381 | 0.106 | 0.077 | 0.50 | 0.98 |
| Sánchez 2024 | 6 | 9 | 11 | 377 | 0.087 | 0.038 | 0.50 | 0.97 |
| AMLO 2023 | 8 | 10 | 12 | 400 | 0.115 | 0.062 | 0.62 | 0.92 |
| Milei 2024 | 5 | 7 | 10 | 400 | 0.108 | 0.054 | 0.62 | 0.97 |

Tópicos cubiertos por caso: Sánchez 2023 acierta UE, Ucrania, Gaza y CGPJ (falla vivienda,
reindustrialización, Cataluña y Marruecos); Sánchez 2024 acierta Palestina, DANA, India/China y
financiación; AMLO acierta Tren Maya, IMSS-Bienestar, Sembrando Vida, seguridad y Poder Judicial;
Milei acierta ajuste fiscal, inflación, recorte del gasto, Pacto de Mayo e Israel.

### 12.6 Calibración

| Decisión | Valor | Razón |
|---|---:|---|
| `k` de grupos | `clip(round(√(n/4)), 3, 8)` | Con k=7-8 en Milei (87 docs) los grupos se fragmentan y el resumen se vuelve retórico; con 5 se separan ideología, economía, libro, Israel y Mercosur. |
| Documentos por grupo | 3 | Con 4 (más candidatas) el ROUGE-1 anual no mejora y el coste se duplica. |
| Oraciones por documento | 2 | Aporta el mínimo material por medoide; el resto lo decide la fusión. |
| Presupuesto | 400 palabras | Con 450-500 el recall de tópicos sube a 0.59 pero ROUGE-1 baja a 0.09 (la referencia es de 70 palabras). 400 mantiene el resumen legible. |
| Cuotas | proporcional + 5 % | Evita que un grupo acapare y que una ceremonia ocupe un lugar fijo. |
| Términos genéricos | df ≥ 40 % del año | Quita plantilla (`president`, `gobiern`, nombre del hablante) de etiquetas, cobertura y tópicos. |
| Afinidad con el centroide del año | peso 0 | Probada con peso 0.15: no mejoró la cobertura y en Milei reforzó el grupo retórico mayoritario, perdiendo la frase del ajuste fiscal. Queda como opción (`year_weight=`). |
| IDF | global | Ver §12.3. |

### 12.7 Coste

| Fase | Sánchez 2023 (129 docs) | AMLO 2023 (308 docs) | Milei 2024 (87 docs) |
|---|---:|---:|---:|
| `prepare_year` (agrupar + 18 resúmenes individuales) | 3.5 s | 10.7 s | 3.2 s |
| `fuse_year` (representación + MMR) | 0.13 s | 0.13 s | 0.11 s |

El banco completo (4 casos × 7 variantes, incluida la ablación sin agrupamiento) tarda ~36 s; el
coste lo domina leer y resumir los medoides, no la fusión.

### 12.8 Limitaciones del módulo 5

1. **Las etiquetas son stems**, no títulos editoriales: `reindustrializ · innov · apuest` es el grupo
   económico de Sánchez 2023; `fentanil · arqueolog` es una mañanera de AMLO. Sirven para auditar el
   agrupamiento, no para publicar.
2. **AMLO agrupa peor**: las mañaneras son un formato homogéneo y el grupo mayor (107 de 308) es la
   conferencia genérica. El resumen anual de AMLO mezcla programas, seguridad y Acapulco, pero
   también frases de trámite.
3. **Los temas pequeños se pierden**: con cuota proporcional, un tema de 3 discursos (Marruecos) o
   de 13 (Acapulco) puede quedar fuera; en AMLO la oración de Acapulco (~60 palabras) tampoco cabe
   en el presupuesto. La variante "cuotas mínimas" los recupera a costa de ruido ceremonial.
4. **ROUGE no es justo con resúmenes largos**: contra referencias de 70 palabras, un resumen de 390
   tiene F1 bajo por precisión. Se reporta para comparar variantes, y el recall de tópicos es la
   métrica de decisión.
5. **Sólo 4 casos anuales** y referencias escritas por una persona: las diferencias de 0.01-0.03 no
   son significativas.
6. **No hay normalización por hablante** en el resumen: cada año se resume por separado (restricción
   del dataset), pero no se comparan puntuaciones absolutas entre presidentes.

---

## 13. Limitaciones

1. **Extractivo puro**: no reescribe. Las oraciones que empiezan a media idea (`"Muestra que la
   vacuna…"`) se penalizan pero pueden colarse; los cierres protocolares ("Que Dios los bendiga")
   también.
2. **El hecho noticioso no se detecta**: en el discurso de Milei el caso de Nahuel Gallo (la noticia
   del día) pierde frente a oraciones más "centrales" léxicamente. Sin supervisión, la saliencia
   léxica no distingue noticia de retórica. Es la limitación más visible del método.
3. **Cinco referencias** es una muestra pequeña; ROUGE con referencias abstractivas castiga la
   selección de oraciones por sinonimia. Las diferencias de 0.01-0.03 no son significativas.
4. **Transcripciones**: las mañaneras de AMLO (18k palabras, oraciones de 60 palabras) producen
   resúmenes del ~1 %: el presupuesto es el límite, no el ratio. El filtro de 75 palabras descarta
   tiradas muy largas, que a veces contienen el contenido.
5. **Cobertura conservadora en textos cortos**: se mide sobre los 20 términos de mayor peso; con 10
   oraciones cada término aparece en una sola, así que 0.15 no implica mal resumen.
6. **Sesgo del IDF global**: un discurso que repite términos comunes del corpus presidencial
   ("pueblo", "gobierno", "país") tiene menos términos con IDF alto; no se corrige por hablante.
7. **Resumen anual**: las limitaciones propias del módulo 5 (etiquetas por stems, agrupamiento débil
   de las mañaneras, temas pequeños que se pierden) están en §12.8.

---

## 14. Reproducir

```bash
source ../env/bin/activate
python rat_scripts/build_summaries.py          # módulo 4: eval.json + summaries_demo.jsonl + tables.md
python rat_scripts/build_multidoc.py           # módulo 5: artifacts/multidoc/*
jupyter nbconvert --to notebook --execute --inplace RAT_demo.ipynb   # demo reproducible
```

Artefactos del módulo 4: `rat_scripts/artifacts/eval.json` (métricas completas),
`summaries_demo.jsonl` (los 5 resúmenes híbridos con sus oraciones y componentes) y `tables.md`
(todas las tablas de §§10-11). Del módulo 5: `artifacts/multidoc/eval_multidoc.json`,
`summaries_year.jsonl` (4 resúmenes anuales con grupos y oraciones) y `tables_multidoc.md` (tablas de
§12).
