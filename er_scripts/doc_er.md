# Módulo 3 — Grafo de entidades y relaciones

**Proyecto:** Observatorio del discurso presidencial en español
**Unidades del temario cubiertas:** EI 3.3 (reconocimiento de entidades, identificación y
clasificación de nombres) y EI 3.5 (extracción de relaciones), con la visualización de grafo
como entregable.

**Restricciones respetadas:** cero IA generativa, cero deep learning, cero transformers, cero
embeddings neuronales. Todo es simbólico y auditable: gazetteers escritos a mano, un
Aho-Corasick (reutilizado del módulo de EI), expresiones regulares, reglas de coreferencia,
patrón sujeto–verbo–objeto y NetworkX para el grafo.

---

## 1. Qué se extrae

**Menciones.** Cada aparición de una entidad con su span de caracteres en el discurso:

| Campo | Valores | Ejemplo |
|---|---|---|
| `type` | `PER`, `ORG`, `LOC`, `CARGO` | `PER` |
| `surface` | forma tal como aparece | `el presidente del Gobierno` |
| `canonical` | nodo canónico | `PER:Pedro Sánchez` |
| `resolved` | nodo tras coreferencia | `PER:Pedro Sánchez` |
| `source` | `gazetteer`, `nombre`, `sigla`, `cargo_de`, `cargo_explicito`, `cargo_inferido`, `coref` | `coref` |

**Relaciones.** Aristas tipadas con peso y evidencia:

| Campo | Ejemplo |
|---|---|
| `source`, `target` | `PER:Zoé Robledo`, `ORG:IMSS` |
| `type` | `ocupa_cargo`, `visita`, `coocurre`, `acusa`, … |
| `weight` | 12 ocurrencias en el corpus |
| `docs` | 7 documentos distintos |
| `evidence` | hasta 3 oraciones con `doc_id` y `sent_idx` |

**Nodos del grafo.** `{id, label, type, mentions, docs, degree, weighted_degree, betweenness,
community}`. El grafo se exporta a `graph.json` (nodos/aristas para d3/pyvis) y `graph.graphml`
(Gephi/NetworkX).

---

## 2. Arquitectura

```
discurso
  │
  ├─ segmentación + tokenización con offsets        (reutiliza ei_scripts.segment)
  │
  ├─ MENCIONES  (mentions.py)
  │    1. gazetteer: personas / organizaciones / lugares   ← Aho-Corasick (619 patrones)
  │    2. cabeceras de organización: "Secretaría de la Defensa"      ← regex con anclas
  │    3. rachas de mayúsculas → nombres propios nuevos             ← patrón + listas negras
  │    4. siglas y acrónimos: IMSS, CFE, OTAN, CESFAM
  │    → resolución de solapes: span más largo, luego prioridad PER > ORG > LOC > CARGO
  │      (excepción: un CARGO puede contener una ORG: "director del IMSS")
  │
  ├─ COREFERENCIA  (coref.py)
  │    hablante · gobierno del país · cargo→ocupante (explícito/inferido) · sigla↔expansión
  │
  ├─ RELACIONES  (relations.py)
  │    coocurrencia en ventana (recall) + sujeto-verbo-objeto (precisión) + cargo
  │
  └─ GRAFO  (graph.py): agregación, proyección no dirigida, métricas, export
```

---

## 3. Menciones

### 3.1 Capa 1 — gazetteer con Aho-Corasick

Se construye **un solo** autómata con 658 patrones:

| Léxico | Patrones | Ejemplos |
|---|---|---|
| Personas (nombre + alias) | 383 | `AMLO`, `López Obrador`, `el tabasqueño`, `CFK`, `Zapatero`, `Boric` |
| Organizaciones (siglas y nombres) | 136 | `IMSS`, `CFE`, `PEMEX`, `OTAN`, `PSOE`, `Cámara de Diputados` |
| Lugares | 139 | `México`, `Ciudad de México`, `España`, `Padre Las Casas`, `La Araucanía` |
| **Total** | **658** | 1,019 estados, 1,018 aristas, profundidad máxima 11 |

Coste de búsqueda `O(n + z)` sobre ~23.7 millones de tokens: una pasada por el corpus.

**Reglas de precisión** (sin ellas el gazetteer sobre-genera):

1. **Mayúscula obligatoria** para personas y organizaciones. Sin esto, `maduro` (adjetivo),
   `ciudadanos`, `podemos` o `firma` se leerían como entidades: en español la capitalización es
   la única pista simbólica disponible.
2. **Apellido suelto**: una persona de un solo token no se acepta si está pegada a otra palabra
   capitalizada (`… Rodríguez Vallejo` no debe producir "Vallejo" → *Camila Vallejo*).
3. **Topónimo en mayúsculas**: un token todo en mayúsculas que es un topónimo del gazetteer no se
   trata como sigla (`AGUASCALIENTES` → lugar, no organización).
4. `~~~PREGUNTA`, `INTERVENCIÓN`, `INAUDIBLE`: marcadores de transcripción descartados.

### 3.2 Capa 2 — rachas de mayúsculas

Se buscan secuencias de palabras capitalizadas admitiendo conectores (`de`, `del`, `la`, `las`,
`los`, `y`, `e`) sólo si van seguidos de otra capitalizada:

```
("Diego", "Sinhue", "Rodríguez", "Vallejo") → PER
("Secretaría", "de", "la", "Defensa")        → ORG
```

Clasificación de la racha, en orden: sigla/acrónimo → cabecera de organización →
**nombre de persona** (≥ 2 tokens de contenido). Se rechaza si:

* el primer token es un arranque imposible (`El`, `Pero`, `Hoy`, meses, días, cargos);
* el primer token es un documento o proceso (`Acuerdo`, `Declaración`, `Revolución`, `Plan`, `Ley`);
* la racha completa está en la lista negra de no-personas (`Tren Maya`, `América Latina`,
  `Estado de derecho`, …), obtenida revisando las etiquetas más frecuentes de esta capa;
* todos los tokens son genéricos (`Nacional`, `General`, `Federal`);
* contiene un cargo u honorífico (`Secretario General`, `Fiscal Nacional`).

### 3.3 Capa 3 — cabeceras de organización

`Secretaría|Instituto|Comisión|Consejo|Banco|Fondo|Ministerio|Corte|Partido|Sindicato|…` seguido
obligatoriamente de un complemento capitalizado:

```
"la Secretaría de la Defensa Nacional"        → ORG:Secretaría de la Defensa Nacional
"la Comisión Federal de Electricidad"         → ORG:Comisión Federal de Electricidad
"el Banco Central"                            → ORG:Banco Central
"una secretaría de gobierno"                  → (no, minúscula: no es nombre propio)
```

La cabecera va **capitalizada a propósito**: si se aceptara en minúscula, `el banco`, `la
secretaría` serían organizaciones en cada oración.

---

## 4. Coreferencia por reglas

| Regla | Patrón | Resuelve a | Límite declarado |
|---|---|---|---|
| 1. Hablante | `el presidente`, `el mandatario`, `el Ejecutivo` con artículo/posesivo de 1ª persona | el hablante del discurso | no resuelve `su presidente` ni `la presidenta` |
| 1b. Gobierno | `el gobierno`, `mi gobierno`, `el gobierno de Chile` | `ORG:Gobierno de <país>` | si lleva adjetivo de nacionalidad (`el gobierno francés`) no se resuelve al país del hablante |
| 2. Cargo → ocupante | `el secretario de Hacienda` | ocupante de `CARGO_HOLDERS[(cargo, país)]` | tabla estática: no distingue periodos |
| 3. Sigla ↔ expansión | `Instituto Mexicano del Seguro Social` | `ORG:IMSS` | sólo si la expansión está en el diccionario |
| 4. Nombre adyacente | `el presidente Boric` | gana el nombre propio; la regla 1 no pisa | — |

El cargo resuelto se marca como **explícito** (el nombre aparece junto al cargo: `el secretario de
Hacienda, Rogelio Ramírez de la O`) o **inferido** (sólo aparece el cargo y el ocupante sale de la
tabla). Son relaciones distintas porque la segunda es una inferencia del sistema, no una
afirmación del discurso.

---

## 5. Relaciones

### 5.1 Coocurrencia en ventana

Para cada oración se deduplica por nodo y se toman como máximo `M = 10` entidades. Se emiten
pares con distancia en tokens menor o igual a `W = 12`:

$$\#\text{pares} \le \binom{M}{2} = \frac{M(M-1)}{2} = 45$$

así que la extracción es lineal en el número de oraciones (nunca cuadrática en el corpus).

### 5.2 Patrón sujeto–verbo–objeto

Para cada forma verbal relacional del léxico se toma la entidad más cercana a la izquierda
(sujeto) y a la derecha (objeto) dentro de la oración:

$$\text{sujeto} \xrightarrow{\;v\;} \text{objeto} \;\Rightarrow\; (a, \text{tipo}(v), b)$$

**Las formas verbales se generan por regla morfológica, no por stem.** El stem confunde sustantivo
con verbo: `el apoyo` y `apoyó` comparten stem, y con stemming aparecerían "apoyos" en todas
partes. Se generan 3ª persona y 1ª plural y se **excluye la 1ª singular** presente, que es la más
ambigua (`apoyo`, `gobierno` son también sustantivos):

$$\text{formas}(v)=\begin{cases}
\text{raíz}+\{a, an, ó, aron, aba, aban, ando, amos\} & v \text{ en } -ar\\
\text{raíz}+\{e, en, ió, ieron, ía, ían, iendo, emos, imos\} & v \text{ en } -er/-ir
\end{cases}$$

Resultado: 222 formas reconocidas para 30 verbos relacionales (inaugurar, entregar, visitar,
recibir, nombrar, designar, felicitar, agradecer, criticar, acusar, denunciar, apoyar, rechazar,
sancionar, amenazar, invitar, reunirse, firmar, viajar, gobernar, encabezar, anunciar, condenar,
exigir, pedir, decretar, promulgar, destituir, expulsar, saludar).

### 5.3 Cargo

Si el cargo se resolvió a persona y **dentro de su span** hay una organización
(`director del IMSS`), se emite `ocupa_cargo` (explícito) o `cargo_inferido`.

### 5.4 Agregación

Las ocurrencias no se guardan una por una: se acumulan por arista `(a, b, tipo)` con
`weight` = ocurrencias, `docs` = documentos distintos y hasta 3 oraciones de evidencia
(`MAX_EVIDENCE`). Sólo se publican aristas con `weight ≥ 2`.

---

## 6. Grafo y métricas

Para las métricas se proyecta el multigrafo dirigido a un grafo **no dirigido ponderado**
(peso = suma de ocurrencias de todos los tipos). Las métricas se calculan sobre los **nodos
conectados**: la mayoría de las entidades aparece una sola vez y un nodo aislado distorsiona
densidad y modularidad.

$$D=\frac{2|E|}{|V|(|V|-1)},\qquad
Q=\frac{1}{2m}\sum_{ij}\left[A_{ij}-\frac{k_ik_j}{2m}\right]\delta(c_i,c_j)$$

Con intermediación aproximada por muestreo de `k = 64` fuentes (`betweenness_centrality(k=64)`).
El export a GraphML guarda `label`, `type`, `mentions`, `docs`, `degree`, `betweenness`,
`community` por nodo y `weight` + `rel_types` por arista.

---

## 7. Resultados sobre el corpus

Corrida completa: `python -m er_scripts.build_graph` → **211 s** sobre 4,531 discursos y 993,494
oraciones.

### 7.1 Volumen

| Métrica | Valor |
|---|---|
| Oraciones con al menos una entidad | 371,705 (37.4 %) |
| Menciones | 643,811 |
| Ocurrencias de relación | 257,063 |
| Oraciones con relación | 115,191 (11.6 %) |
| Aristas publicadas (peso ≥ 2) | 18,659 |
| Nodos | 46,869 |
| Nodos con ≥ 1 relación | 4,145 |
| Nodos con un solo documento | 34,776 (74 %) |

### 7.2 Menciones por tipo y por fuente

| Tipo | Menciones | % |
|---|---|---|
| `PER` | 231,466 | 36.0 |
| `LOC` | 214,522 | 33.3 |
| `ORG` | 158,992 | 24.7 |
| `CARGO` | 38,831 | 6.0 |

| Fuente | Menciones | % |
|---|---|---|
| gazetteer | 380,300 | 59.1 |
| racha de mayúsculas | 174,736 | 27.1 |
| cargo (`cargo_de`) | 38,831 | 6.0 |
| coreferencia | 24,587 | 3.8 |
| siglas | 20,101 | 3.1 |
| ocupante inferido | 4,643 | 0.7 |
| ocupante explícito | 613 | 0.1 |

### 7.3 Relaciones

Sólo se publican aristas con peso ≥ 2 (18,659 de las ~83 mil distintas). Por tipo:

| Tipo | Aristas | Ocurrencias |
|---|---|---|
| `coocurre` | 18,425 | 190,049 |
| `recibe` | 30 | — |
| `apoya` | 29 | — |
| `visita` | 25 | — |
| `encabeza` | 20 | — |
| `pide_a` | 19 | — |
| `entrega` | 19 | — |
| `denuncia` | 16 | — |
| `firma_con` | 12 | — |
| `gobierna` | 9 | — |
| resto (21 tipos verbales) | 47 | — |
| **verbales (total)** | **226** | **576** |
| `cargo_inferido` + `ocupa_cargo` | 8 | 1,221 |

El **99.3 % de las ocurrencias publicadas es coocurrencia**: es la capa de *recall*, barata y sin
sesgo de vocabulario, pero no dice **qué** une a las entidades. Las relaciones verbales —que sí lo
dicen— son 226 aristas y 576 ocurrencias, tres órdenes de magnitud menos. Las aristas de cargo
tienen peso medio 153 porque la inferencia estática dispara en cada mención del ministerio, no en
cada afirmación sobre la persona.

### 7.4 Entidades más mencionadas

| Nodo | Tipo | Menciones | Documentos |
|---|---|---|---|
| Andrés Manuel López Obrador | PER | 56,064 | 2,298 |
| México | LOC | 39,538 | 2,369 |
| España | LOC | 18,639 | 1,873 |
| secretario | CARGO | 16,034 | 2,075 |
| Estados Unidos | LOC | 15,652 | 2,224 |
| Gobierno de México | ORG | 11,186 | 1,942 |
| presidente | CARGO | 7,011 | 2,503 |
| Guardia Nacional | ORG | 6,793 | 1,186 |
| Gobierno de España | ORG | 6,657 | 932 |
| PEMEX | ORG | 6,612 | 1,091 |

### 7.5 Métricas del grafo

| Métrica | Valor |
|---|---|
| Densidad | 0.0016 |
| Grado medio (nodos conectados) | 6.78 |
| Componentes conexas | 186 |
| Tamaño del componente gigante | 8.0 % de los nodos |
| Comunidades (modularidad voraz) | 228 |
| Modularidad | 0.547 |

| Nodo | Grado | Intermediación |
|---|---|---|
| Andrés Manuel López Obrador | 633 | 0.139 |
| México | 620 | 0.104 |
| secretario | 542 | 0.103 |
| presidente | 486 | 0.108 |
| Gobierno de México | 456 | 0.046 |

Las comunidades separan ejes temáticos/geográficos coherentes (una comunidad de cargos
institucionales, otra de estados mexicanos, otra de regiones españolas, otra de mandatarios y
organismos internacionales). La modularidad 0.55 > 0.3 indica estructura comunitaria real, no
ruido.

### 7.6 Por hablante

| Hablante | Menciones | Entidades dominantes |
|---|---|---|
| AMLO | 518,158 | AMLO (56,019), México (39,286), secretario (15,806) |
| Pedro Sánchez | 86,126 | España (16,994), Gobierno de España (6,657), Europa (4,351) |
| Sebastián Piñera | 26,114 | Chile (5,036), Gobierno de Chile (538), presidente (395) |
| Javier Milei | 9,604 | Argentina (2,187), PBI (800), Banco Central (737) |
| Nicolás Maduro | 3,809 | Venezuela (743), Estados Unidos (137), Hugo Chávez (104) |

El 84 % de las menciones son de AMLO porque aporta el 79 % de las oraciones del corpus. El perfil
de Milei es el más "economicista" (Banco Central, PBI) y el de Sánchez el más "europeísta".

---

## 8. Evaluación

### 8.1 Gold de menciones

`er_scripts/gold/gold_sentences.csv`: **40 oraciones** (8 por hablante) con **74 entidades**
etiquetadas a mano y emparejamiento **estricto** por nodo canónico:

| Tipo | TP | FP | FN | P | R | F1 | IC95 P |
|---|---|---|---|---|---|---|---|
| `ORG` | 15 | 4 | 6 | 0.789 | 0.714 | 0.750 | [0.57, 0.92] |
| `PER` | 12 | 2 | 3 | 0.857 | 0.800 | 0.828 | [0.60, 0.96] |
| `LOC` | 31 | 0 | 7 | 1.000 | 0.816 | 0.899 | [0.89, 1.00] |
| **micro** | **58** | **6** | **16** | **0.906** | **0.784** | **0.841** | [0.81, 0.96] |

Ablaciones:

| Configuración | P | R | F1 |
|---|---|---|---|
| gazetteer + patrones + coreferencia (estricto) | 0.906 | 0.784 | 0.841 |
| sin coreferencia (estricto) | 0.905 | 0.770 | 0.832 |
| con coreferencia (laxo) | 0.906 | 0.784 | 0.841 |

La coreferencia aporta +1.4 puntos de recall en esta muestra (poco, porque el gold contiene pocas
elipsis: la muestra es de 40 oraciones y las reglas de hablante sólo actúan cuando aparece
`el presidente`/`el gobierno`). El criterio laxo no cambia nada: los nodos canónicos ya coinciden
con la etiqueta cuando el emparejamiento estricto acierta.

Intervalo de Wilson al 95 %:

$$\frac{\tilde p+\frac{z^2}{2n}\pm z\sqrt{\frac{\tilde p(1-\tilde p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}$$

### 8.2 Relaciones

Muestra de **53 aristas** (`gold_relations.csv`) con su oración de evidencia; se juzga si las dos
entidades son correctas **y** si la oración sostiene la relación:

| Grupo | n | aciertos | precisión |
|---|---|---|---|
| `coocurre` | 30 | 21 | 0.700 |
| verbales (todos los tipos) | 21 | 6 | 0.286 |
| cargo (inferido/explícito) | 2 | 2 | 1.000 |
| **total** | **53** | **29** | **0.547** |

Lectura honesta: **la coocurrencia es fiable (0.70) y las relaciones verbales no (0.29)**. El
patrón sujeto–verbo–objeto elige la entidad más cercana, y en oraciones largas de transcripción la
más cercana no siempre es el sujeto. Los dos aciertos de cargo son el caso controlado
(`Raquel Buenrostro ↔ Secretaría de Economía`), donde la regla es determinista.

### 8.3 Tipología de errores

**Falsos positivos de mención** (los 6 de la muestra): `Sistema de Salud`, `Sistema Penal` leídos
como ORG; `Secretario General de Interpol` como PER (es un cargo sin nombre); acrónimos
desconocidos que crean nodo propio (`SAR`, `CESCOF`).

**Falsos negativos**: entidades de un solo token que no están en el gazetteer (`Mises`,
`Kissinger`, `Santeliz`), topónimos menores (`Agua Negra`, `Rengo`, `Navidad`) y organizaciones con
nombre metafórico (`Revolución Bolivariana`).

**Relaciones mal tipadas**: `Ucrania --condena--> Putin` (la condena es del hablante, no de
Ucrania), `España --pide_a--> Unión Europea` (quien pide son otros líderes), `AMLO --anuncia_a-->
México` (anuncia **en** México). El patrón no distingue sujeto de complemento circunstancial
cuando la oración es larga.

---

## 9. Limitaciones

1. **Gold pequeño y de un anotador** (40 oraciones, 74 entidades): los intervalos son anchos.
2. **El gold se revisó una vez** leyendo las oraciones completas (la muestra se generó truncada a
   400 caracteres). Mejora la calidad del gold pero puede favorecer al sistema: se declara como
   sesgo de anotación.
3. **Coreferencia acotada**: sin pronombres personales lejanos (`él`, `ella`), sin desambiguación
   de homónimos y con tabla cargo→ocupante **estática**: no distingue quién ocupaba el cargo en
   cada periodo (Piñera y Boric comparten `ministro de Hacienda`).
4. **Cargos como nodos**: `CARGO:secretario` es el cuarto nodo más mencionado. Es correcto
   conceptualmente (el rol existe sin nombre propio) pero infla el grado de los nodos genéricos.
5. **Relaciones verbales ruidosas** (precisión 0.29 en la muestra): es el precio de un patrón
   simbólico sin análisis sintáctico completo.
6. **Coocurrencia ≠ relación semántica**: dice que dos entidades aparecen juntas, no qué las une.
7. **Cola larga**: 74 % de los nodos aparece en un solo documento; el grafo interpretable es el de
   los ~4 mil nodos conectados.
8. **Siglas**: sólo se fusionan si la expansión está en el diccionario.

---

## 10. Reproducibilidad

```bash
source ../env/bin/activate

python -m er_scripts.build_graph                    # 211 s → menciones, relaciones, grafo, stats
python -m er_scripts.build_graph --limit 200        # prueba rápida
python -m er_scripts.build_graph --no-coref --tag ablacion
python -m er_scripts.build_graph --no-patterns --tag solo_gazetteer
python -m er_scripts.evaluate --markdown             # métricas + artifacts/tables.md
python -m er_scripts.evaluate --sample-relations 30  # genera muestra para etiquetar
```

La libreta `ER_demo.ipynb` recorre el pipeline sobre ejemplos, muestra las tres capas de
detección, las reglas de coreferencia, las relaciones tipadas, el grafo con sus métricas y la
evaluación contra el gold.

---

## 11. Conexión con el resto del proyecto

* **RI (módulo 1):** el grafo se construye sobre el mismo `doc_id`; `mentions_by_doc.jsonl` es un
  índice documento → entidades que se puede cruzar con BM25 (`buscar "becas" y ver qué entidades
  acompañan`).
* **EI (módulo 2):** los eventos de promesa quedan enriquecidos con estas entidades; los
  gazetteers de lugar y organización son compartidos (`ei_scripts/lexicon.py` es la fuente de los
  topónimos y de parte de las organizaciones).
* **Dashboard:** `graph.json` (nodos/aristas con tipo, peso y comunidad) alimenta directamente la
  visualización; `relations.jsonl` conserva la evidencia para el panel de detalle.
