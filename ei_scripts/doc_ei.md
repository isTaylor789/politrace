# Módulo 2 — Extracción de anuncios y promesas

**Proyecto:** Observatorio del discurso presidencial en español
**Unidades del temario cubiertas:** EI 3.1 (concepto y diferencias RI/EI), 3.2 (expresiones
regulares, autómatas, gramáticas independientes del contexto), 3.3 (extracción de entidades) y
3.4 (extracción de eventos) — con la parte de relaciones de 3.5 usada de forma instrumental
(coocurrencia de entidades en el marco).

**Restricciones respetadas:** cero IA generativa, cero deep learning, cero transformers,
cero embeddings neuronales. Todo el sistema es simbólico, determinista y auditable:
un autómata Aho-Corasick, expresiones regulares, una CFG con parser CYK, gazetteers escritos a
mano y MinHash/LSH.

---

## 1. Qué se extrae exactamente

Se define **anuncio/promesa** como una oración en la que el hablante (o su gobierno, en primera
persona) se compromete a una acción futura. Operativamente se exige:

1. un **disparador** léxico de acto de habla (`vamos a`, `me comprometo a`, `anuncio que`,
   `entregaremos`, …);
2. una **acción prometible** cercana al disparador (perífrasis verbal local);
3. que la oración **no** sea pregunta, cita textual ni negación;
4. que la estructura encaje en el **molde de evento** de la gramática.

Cada evento es un registro con este esquema:

| Campo | Tipo | Ejemplo |
|---|---|---|
| `doc_id`, `speech_id` | int, str | `2769`, `ESP-SANCHEZ-20200104-…` |
| `speaker`, `country`, `date` | str | `Pedro Sánchez`, `ESP`, `2020-01-04` |
| `sent_idx`, `char_start`, `char_end` | int | posición exacta de la oración en el discurso |
| `sentence` | str | texto de la oración (evidencia citable) |
| `trigger`, `trigger_class`, `strength` | str, str, float | `vamos a`, `future_1p`, `0.90` |
| `action` | str | `construir un hospital` |
| `event_type`, `type_evidence` | str, str | `salud`, `hospital` |
| `frame.fecha` | {texto, valor} | `antes de diciembre de 2024` → `??-12` |
| `frame.monto`, `frame.cantidad`, `frame.porcentaje` | {texto, valor} | `3,000 millones de pesos` → `3000000000` |
| `frame.lugar`, `frame.organizacion`, `frame.beneficiario` | str | `en Tijuana`, `IMSS`, `a los jóvenes` |
| `polarity` | str | `afirmado` / `atenuado` |
| `chunks`, `cyk`, `cyk_tree` | list, bool, str | segmentación en chunks y árbol de parse |
| `confidence`, `method` | float, str | `0.86`, `ac+regex+cyk` |
| `dup_group`, `dup_rep` | int, bool | grupo de casi duplicados |

---

## 2. Arquitectura

```
texto del discurso
   │
   ├─ segmentación (punkt + corte por ; : , en oraciones > 90 palabras)
   │      └─ oraciones con offsets de carácter sobre el texto crudo
   │
   ├─ tokenización dual: superficie (mayúsculas/acentos) y plegada (minúsculas sin acentos)
   │
   ├─ AHO-CORASICK sobre tokens plegados
   │      · 456 disparadores (multi-token y de un token)
   │      · 117 topónimos + 46 organizaciones (gazetteers)
   │      · futuros sintéticos generados por regla morfológica
   │      └─ DFA de negación sobre la ventana previa al disparador
   │
   ├─ REGEX de slots sobre texto crudo: fecha, monto, cantidad, %, lugar, beneficiario
   │
   ├─ CHUNKER  →  TRIG | VTRIG | ACC | SN | LOC | ORG | BENEF | PLAZO | MONTO | MISC
   │      └─ CFG (CNF) + CYK: ¿los chunks forman un marco de promesa válido?
   │
   ├─ PLANTILLA DE EVENTO + score de confianza (suma ponderada de evidencias)
   │
   └─ DEDUPLICACIÓN MinHash + LSH (mismo hablante, shingles de 3 palabras)
```

Archivos: `config.py` (parámetros), `lexicon.py` (léxicos y gazetteers), `segment.py`,
`automaton.py`, `slots.py`, `grammar.py`, `extract.py`, `dedup.py`, `build_events.py`
(CLI), `evaluate.py` (métricas). Artefactos en `ei_scripts/artifacts/`.

---

## 3. Segmentación y tokenización

`punkt` para español sobre el texto crudo; las transcripciones sin puntuación (mañaneras de AMLO)
se parten además por `;`, `:` y, si aún queda una oración de más de `MAX_SENT_TOKENS = 90`
palabras, por `,` agrupando bloques.

Dos vistas sincronizadas por offsets: la **cruda** (para mostrar y para los patrones que dependen
de mayúsculas o acentos) y la **plegada** con `ri_scripts.preprocess.fold` — minúsculas, sin
acentos, conservando la ñ y **la longitud**, de modo que el offset de un token es idéntico en
ambas vistas.

---

## 4. Aho-Corasick: disparadores y gazetteers

### 4.1 Definición

El autómata se construye con las tres funciones clásicas sobre el alfabeto de tokens plegados:

* `goto(q, a)`: estado siguiente desde `q` con el símbolo `a` (diccionario de aristas).
* `fail(q)`: el sufijo propio más largo de la cadena de `q` que también es prefijo de algún
  patrón; se calcula por BFS en orden de profundidad.
* `out(q)`: patrones que terminan en `q`. Por herencia, `out(q) ⊇ out(fail(q))`, lo que permite
  detectar patrones que son sufijo de otros.

Búsqueda: se recorre la secuencia de `n` tokens siguiendo `goto`/`fail` y se emiten las salidas;
coste `O(n + z)` para `z` coincidencias, **independiente del número de patrones**.

Comparación con la alternativa ingenua: 456 disparadores × ~23.7 millones de tokens ≈
1.1 × 10¹⁰ comparaciones de token con `str.find`; con el autómata, una sola pasada de
~23.7 millones de transiciones (medido: 146 s para todo el corpus, incluido el resto del
pipeline; 27.0 millones de tokens contando puntuación).

### 4.2 Inventario construido

| Léxico | Patrones | Ejemplos |
|---|---|---|
| Disparadores de compromiso | 28 | `me comprometo a`, `mi compromiso es`, `damos nuestra palabra`, `tenemos el compromiso` |
| Futuro perifrástico | 5 | `voy a`, `vamos a`, `va a`, `van a` |
| Futuro sintético, 1ª persona | 194 | `construiremos`, `entregaré` |
| Futuro sintético, 3ª persona | 194 | `entregará`, `inaugurarán` (exigen sujeto atribuible) |
| Anuncio performativo | 17 | `anuncio que`, `informo que`, `doy a conocer`, `planteamos` |
| Instrucción / acto administrativo | 8 | `he girado instrucciones`, `he ordenado`, `he pedido` |
| Marco temporal (no es promesa por sí solo) | 7 | `antes de que termine`, `en los próximos` |
| Deóntico (señal débil) | 3 | `hay que`, `tenemos que`, `debemos` |
| Topónimos (gazetteer) | 117 | `México`, `Tijuana`, `estado de México`, `Padre Las Casas` |
| Organizaciones (gazetteer) | 46 | `IMSS`, `CFE`, `Pemex`, `Suprema Corte` |
| **Total** | **619** | 727 estados, 726 aristas, profundidad máxima 11 |

Los futuros sintéticos no están listados a mano: se generan por regla morfológica a partir de la
lista de verbos de acción.

$$\text{futuro}(v)=\begin{cases}
v[0{:}{-}2]+\{\text{aré},\text{aremos}\} & v \text{ termina en } ar\\
v[0{:}{-}2]+\{\text{eré},\text{eremos}\} & v \text{ termina en } er/ir
\end{cases}$$

La 3ª persona (`entregará`) se acepta sólo si la oración tiene sujeto atribuible, porque en los
verbos en `-ar` es indistinguible del imperfecto de subjuntivo (`entregara`).

### 4.3 Reglas de resolución de solapes

1. Gana el patrón **más largo** (`me comprometo a` > `me comprometo`).
2. `vamos a ver / vamos a platicar`: la perífrasis seguida de verbo de relleno no es promesa.
3. `el anuncio de…`: el sustantivo no es acto de habla (se descarta si le precede un determinante).
4. `va a / van a` y futuros de 3ª persona: exigen **atribución** al hablante o a su gobierno
   (`vamos`, `nosotros`, `gobierno`, `mi`, `voy`).

---

## 5. DFA de negación

Cuatro estados, leído hacia atrás desde el disparador, ventana `NEGATION_WINDOW = 6` palabras.
La puntuación fuerte (`. ; : ! ?`) **bloquea** la ventana; la coma no.

| estado | negador | puntuación fuerte | otro |
|---|---|---|---|
| limpio | negado | bloqueado | limpio |
| negado | doble | bloqueado | negado |
| doble | negado | bloqueado | doble |
| bloqueado | bloqueado | bloqueado | bloqueado |

`doble` equivale a limpio (la doble negación se cancela):

| oración | estado final | polaridad |
|---|---|---|
| `No vamos a aumentar los impuestos` | negado | **negado** |
| `No es cierto que no vamos a construir el hospital` | doble | afirmado |
| `Quizá podríamos construir el hospital` | limpio | atenuado (hedge) |
| `Vamos a construir el hospital. No vamos a cerrar escuelas` | bloqueado | afirmado |

Si la polaridad es `negado` el evento se descarta (configurable) y si es `atenuado` la confianza
se multiplica por 0.7.

---

## 6. Slots por expresiones regulares

Se aplican sobre el texto crudo, con el nombre del patrón guardado en `evidence`.

| Slot | Patrón (resumen) | Ejemplo → valor |
|---|---|---|
| `DATE` | `\d{1,2} de (enero\|…\|diciembre)( de \d{4})?` | `antes de diciembre de 2024` → `??-12`, `2024` |
| `DATE` | `en los próximos \d+ (días\|meses\|años)` | `en los próximos 100 días` |
| `DATE` | `a más tardar`, `antes de que termine el sexenio`, `(19\|20)\d{2}` | `2024` |
| `MONEY` | `\d[\d.,]*\s*(mil\|millón\|millones)?\s*(de)?\s*(pesos\|dólares\|euros)` | `3,000 millones de pesos` → `3000000000` |
| `MONEY` | `\d[\d.,]*\s*(mdp\|mmdp\|mdd)` | `250 mdp` → `250 MDP` |
| `PCT` | `\d+([.,]\d+)?\s*(por ciento\|%)` | `35 por ciento` → `35%` |
| `QTY` | `\d[\d.,]*\s+(mil\|millones de)?\s*(becas\|viviendas\|empleos\|…)` | `15 mil becas` → `15000` |
| `PLACE` | `en (el\|la)? [A-ZÁÉÍÓÚÑ]\w+` | `en Tijuana` |
| `BENEF` | `para (los\|las\|todos los) \w+( \w+){0,3}` | `a los jóvenes` |

Normalización numérica: `\d{1,3}([.,]\d{3})+` se interpreta como separador de miles
(`1.290.000` → `1290000`); en otro caso la coma es decimal (`3,5` → `3.5`). Los spans se resuelven
sin solapes con prioridad `DATE > MONEY > PCT > QTY > PLACE > BENEF`.

---

## 7. Gramática independiente del contexto + CYK

### 7.1 Por qué el molde y no el español completo

Parsear oraciones completas de español con una CFG escrita a mano exige una gramática de miles de
reglas y una ambigüedad inmanejable. Lo que se necesita aquí es decidir si la oración **contiene
un marco de promesa**, y eso se puede decidir sobre una secuencia de *chunks*: el chunker
colapsa el léxico en etiquetas con papel semántico (anclas del autómata y de las regex) y la CFG
verifica la estructura.

### 7.2 Gramática en forma binaria (CNF)

```
S    -> PRE S2 | S2
S2   -> TRIG REST | VTRIG MODS
REST -> ACC | ACC MODS | MODS ACC | MODS TAIL
TAIL -> ACC MODS
MODS -> MOD | MOD MODS
MOD  -> SN | LOC | ORG | BENEF | PLAZO | MONTO | MISC
PRE  -> SN | LOC | ORG | PLAZO | MONTO | MISC
```

`ACC` es **obligatorio**: sin verbo de acción no hay promesa. `VTRIG` cubre el caso en que el
propio disparador es el verbo (`Entregaremos`, `Construiremos`). `PRE` admite contexto previo
(`En Tijuana vamos a construir…`).

### 7.3 Chunker

Cada token recibe, si aplica, un ancla (disparador, slot o gazetteer) con prioridad
`TRIG/VTRIG 100 > PLAZO 65 > MONTO 60 > LOC 50 > ORG 45 > BENEF 40 > ACC 30 > SN 20 > MISC 0`.
Los tramos sin ancla se etiquetan `ACC` si contienen un verbo de acción, `SN` si contienen un
sustantivo del lexicón temático y `MISC` en el resto. Los chunks contiguos de la misma etiqueta se
fusionan (dos disparadores, dos fechas).

### 7.4 CYK

Algoritmo clásico: `table[i][j]` = conjunto de no terminales que cubren los chunks `i..j`,
rellenando por longitud y probando todos los puntos de corte `k`, más cierre unario.

$$\text{coste}=O(n^3\cdot|P|),\qquad n=\text{nº de chunks}\ (\le 40),\ |P|=24\ \text{reglas}$$

Con un backpointer por `(símbolo, i, j)` se reconstruye **un** árbol de ejemplo; el árbol se
guarda en el evento como texto (`cyk_tree`) y sirve de auditoría del parse.

Ejemplo real (oración del corpus):

```
Vamos a construir un hospital en Tijuana antes de diciembre de 2024.
chunks: TRIG:vamos a | ACC:construir un hospital | LOC:en Tijuana | PLAZO:antes de diciembre …

S
  S2
    TRIG
    REST
      ACC
      MODS
        MOD
          LOC
        MODS
          MOD
            PLAZO
```

---

## 8. Plantilla de evento y score de confianza

### 8.1 Score

$$\text{conf}=\frac{\sum_i w_i\,e_i}{\sum_i w_i}\cdot\varphi(\text{polaridad}),\qquad
\varphi=\begin{cases}1 & \text{afirmado}\\ 0.7 & \text{atenuado}\\ 0.15 & \text{negado}\end{cases}$$

| Evidencia $e_i$ | Peso $w_i$ | Valor |
|---|---|---|
| Disparador | 0.40 | fuerza de su clase: `commit` 1.0, `future_1p` 0.90, `future_1p_syn` 0.90, `future_3p_syn` 0.85, `announce` 0.80, `instruct` 0.70, `goal` 0.45, `deontic` 0.25 |
| Acción | 0.20 | 1.0 si es de **contenido** (`construir`, `entregar`, `invertir`…), 0.5 si es **de discurso** (`informar`, `presentar`, `anunciar`) |
| Slot duro | 0.15 | 1.0 si hay fecha, monto, cantidad o porcentaje |
| CYK | 0.15 | 1.0 si la CFG acepta la oración |
| Lugar | 0.10 | 1.0 si hay LOC en el marco |

La normalización por los pesos **activos** hace comparables las ablaciones: al apagar un
componente, su peso sale del denominador y el máximo sigue siendo 1.0.

Los valores posibles son discretos (indicadores con pesos fijos), y eso se ve en la distribución:
0.71 (disparador + acción de contenido + CYK, 5,761 eventos), 0.61 (acción de discurso, 2,993),
0.81 (+ lugar, 660), 0.86 (+ slot duro, 355)…

### 8.2 Tipo de evento

El tipo se vota con los sustantivos del lexicón (`NOUN_STEMS`, indexado por **stem** Snowball, lo
que absorbe plurales) que aparecen en los chunks de contenido. Taxonomía cerrada de 8 clases:
`infraestructura`, `salud`, `educacion`, `social`, `economia_empleo`, `seguridad`,
`politica_institucional`, `otro`.

---

## 9. Deduplicación: MinHash + LSH

Los presidentes repiten la misma promesa en discursos distintos; comparar todos los pares es
`O(n²)` (≈6 × 10⁷ pares con 11 mil eventos).

**Shingles**: conjuntos de 3-gramas de palabras plegadas, hasheados con BLAKE2b a 64 bits y
reducidos módulo `p = 2³¹ − 1`.

**Firma MinHash** con `k = 96` permutaciones y la familia universal de hashes lineales
$h_i(x) = (a_i x + b_i) \bmod p$:

$$\hat J(A,B)=\frac{1}{k}\sum_{i=1}^{k}\big[\min_{a\in A}h_i(a)=\min_{b\in B}h_i(b)\big],
\qquad \mathbb{E}[\hat J]=J(A,B),\qquad
\sigma(\hat J)=\sqrt{\tfrac{s(1-s)}{k}}$$

Con `k = 96` y `s = 0.8`: error típico 0.041.

**Banding LSH** con `b = 12` bandas de `r = 8` filas:

$$P_{\text{colisión}}(s)=1-\left(1-s^{r}\right)^{b}$$

| Jaccard $s$ | 0.5 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|
| P(candidato) | 0.046 | 0.510 | 0.890 | 0.999 |

Los candidatos se **verifican** con el Jaccard exacto (umbral 0.80), así que un falso candidato
cuesta tiempo, nunca un error de agrupación. La deduplicación se restringe al **mismo hablante** y
a oraciones de al menos 8 tokens. La agrupación se cierra con Union-Find y se conserva como
representante el evento más antiguo.

Resultado en el corpus: **96 grupos, 241 eventos** (2.2 % del total), 2,315 colisiones LSH y 403
pares verificados. La mayoría de los grupos son discursos repetidos de la misma gira
(`doc_id` consecutivos, dos eventos por discurso).

---

## 10. Resultados sobre el corpus

Corrida completa: `python -m ei_scripts.build_events` → **146 s**, 11,052 eventos.

### 10.1 Embudo de filtros (993,494 oraciones)

| Etapa | Oraciones | % del total |
|---|---|---|
| Oraciones segmentadas | 993,494 | 100.0 |
| Descartadas por cortas (< 4 palabras) | 60,617 | 6.1 |
| Descartadas por ser pregunta (`?`) | 79,588 | 8.0 |
| Descartadas por ser cita (comillas tipográficas) | 2,835 | 0.3 |
| Sin ningún disparador | 756,922 | 76.2 |
| **Con disparador** | **93,532** | **9.4** |
| Rechazadas por la CFG/CYK | 73,272 | 7.4 |
| Aceptadas por la CFG | 20,260 | 2.0 |
| Rechazadas por no tener acción | 7,532 | 0.8 |
| Rechazadas por negación | 769 | 0.1 |
| Rechazadas por confianza | 907 | 0.1 |
| **Eventos publicados** | **11,052** | **1.1** |

El 76 % de las oraciones no tiene ni un disparador. De las 93,532 que sí lo tienen, el sistema
descarta el **88 %** (82,480) y publica 11,052. La CFG aporta la mayor parte del filtrado
(73,272); el resto se reparte entre la regla de acción cercana (7,532), la confianza (907) y la
negación (769).

### 10.2 Por hablante (normalizado: el corpus está muy desbalanceado)

| Hablante | Eventos | Oraciones | Eventos / 1000 oraciones | Confianza media |
|---|---|---|---|---|
| Andrés Manuel López Obrador | 9,265 | 789,289 | 11.74 | 0.688 |
| Pedro Sánchez | 1,077 | 127,486 | 8.45 | 0.706 |
| Sebastián Piñera Echenique | 556 | 41,426 | 13.42 | 0.717 |
| Javier Milei | 144 | 31,833 | 4.52 | 0.711 |
| Nicolás Maduro | 10 | 3,460 | 2.89 | 0.676 |

La tasa por mil oraciones es la comparación honesta: Piñera y AMLO prometen mucho más por
oración que Milei (cuyos textos son clases de economía) y Maduro (pocos discursos, muy
ceremoniales). El 84 % de los eventos son de AMLO simplemente porque aporta el 79 % de las
oraciones del corpus.

### 10.3 Distribuciones

| Tipo de evento | Eventos |
|---|---|
| `otro` | 4,442 |
| `social` | 1,504 |
| `economia_empleo` | 1,206 |
| `salud` | 1,132 |
| `politica_institucional` | 962 |
| `infraestructura` | 822 |
| `seguridad` | 560 |
| `educacion` | 424 |

| Año (del discurso) | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| Eventos | 435 | 2,489 | 1,836 | 1,594 | 1,687 | 1,598 | 1,161 | 177 | 74 |

Disparadores más frecuentes: `vamos a` 6,432 (58 %), `voy a` 1,233, `va a` 987,
`vamos a seguir` 630, `van a` 306, `anuncio` 93, `informo que` 91, `planteamos` 84.

Cobertura de slots (sobre 11,052 eventos): lugar 1,481 (13.4 %), beneficiario 439 (4.0 %),
fecha 305 (2.8 %), cantidad 242 (2.2 %), organización 190 (1.7 %), monto 111 (1.0 %).
Sólo el **6.2 %** de los eventos tiene un slot "duro" (fecha, monto, cantidad o porcentaje): las
promesas de los discursos son mayoritariamente cualitativas.

Lugares más citados: México (212), Estados Unidos (63), Sonora (33), Guerrero (30), Oaxaca (28).

### 10.4 Ejemplos reales

| Hablante | Fecha | Conf. | Oración (recortada) |
|---|---|---|---|
| AMLO | 2020-08-04 | 1.00 | "Mi compromiso es terminar estas obras antes de que concluya nuestro gobierno…" → `fecha=antes de que concluya`, `lugar=Nayarit` |
| AMLO | 2022-06-21 | 1.00 | "En Sinaloa tenemos el compromiso de terminar la presa de Santa María…" → `cantidad=10 mil millones`, `lugar=En Sinaloa` |
| Piñera | 2019-01-17 | 0.96 | "Vamos a proteger muchas áreas… son más de 1.300 hectáreas." → `cantidad=1.300 hectáreas`, `lugar=Atacama` |
| Sánchez | 2018-06-19 | 0.96 | "les voy a dar una serie de datos señorías…" → `cantidad=3.283 personas`, `lugar=en el Mediterráneo` |
| Milei | 2025-06-12 | 0.86 | "Entonces voy a arrancar en 1956." → `fecha=1956` (**falso positivo**, ver §11) |

---

## 11. Evaluación

### 11.1 Gold manual

`ei_scripts/gold/gold_seed.csv`: **205 oraciones etiquetadas a mano** a partir de una muestra
estratificada generada por `--sample 25`:

| Estrato | Oraciones | Positivas | Para qué sirve |
|---|---|---|---|
| Aleatorio (25 por hablante) | 125 | 10 | estimar **prevalencia** y recall insesgado |
| Aceptado por el sistema | 40 | 21 | medir **precisión** con casos difíciles |
| Con disparador y rechazado | 40 | 2 | medir **recall perdido** |

Criterio de etiquetado: la oración expresa un anuncio, promesa o compromiso de acción atribuible
al hablante (incluye actos performativos ya realizados, como un nombramiento anunciado). Se
excluyen preguntas, citas y promesas negadas. Un solo anotador: el gold es una **semilla**, mide
acuerdo con un criterio explícito, no verdad absoluta.

Prevalencia estimada en el estrato aleatorio: **0.080** (10/125; IC95 Wilson [0.044, 0.141]).

### 11.2 Ablaciones

Cada configuración enciende un componente, sin umbral de confianza, para medir el aporte de cada
filtro por separado:

| Configuración | TP | FP | FN | P | R | F1 | IC95 P | IC95 R |
|---|---|---|---|---|---|---|---|---|
| A · autómata | 30 | 15 | 8 | 0.667 | 0.789 | 0.723 | [0.52, 0.79] | [0.64, 0.89] |
| B · + negación | 29 | 13 | 9 | 0.690 | 0.763 | 0.725 | [0.54, 0.81] | [0.61, 0.87] |
| C · + slots regex | 29 | 13 | 9 | 0.690 | 0.763 | 0.725 | [0.54, 0.81] | [0.61, 0.87] |
| D · + CFG/CYK (completo) | 25 | 4 | 13 | **0.862** | 0.658 | **0.746** | [0.69, 0.94] | [0.50, 0.79] |

Lectura:

* El **autómata solo** es el mejor recall (0.789) y el peor precision (0.667): los disparadores
  por sí mismos sobre-generan.
* El **DFA de negación** quita un falso positivo (una promesa negada: *"Nada de que vamos a
  condonar impuestos…"*) sin perder ningún positivo.
* Los **slots** no cambian la decisión: no son un filtro, son el **contenido** del marco. Su
  efecto se ve en la cobertura de §10.3 y en el score (suben 0.15 la confianza).
* La **CFG/CYK** es el filtro más caro y el más rentable: elimina 9 falsos positivos a cambio de 1
  positivo (P +0.17, R −0.11).

### 11.3 Niveles de evidencia (curva operativa del sistema completo)

| Nivel | Umbral | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| Cualquier evidencia | 0.00 | 25 | 4 | 13 | 0.862 | 0.658 | 0.746 |
| Acción de contenido | 0.65 | 18 | 3 | 20 | 0.857 | 0.474 | 0.610 |
| + CYK | 0.71 | 17 | 2 | 21 | 0.895 | 0.447 | 0.596 |
| + Lugar explícito | 0.81 | 3 | 1 | 35 | 0.750 | 0.079 | 0.143 |
| + Fecha/monto/cantidad | 0.86 | 1 | 0 | 37 | 1.000 | 0.026 | 0.051 |

El corte útil es el de "cualquier evidencia" (0.55 en `config.MIN_CONFIDENCE`); exigir
cuantificación deja el sistema en precisión 1.0 pero con recall 0.03, porque **las promesas de
los discursos casi nunca traen número**.

### 11.4 Por estrato de muestreo (sistema completo)

| Estrato | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| Aleatorio | 4 | 0 | 6 | 1.000 | 0.400 | 0.571 |
| Aceptado | 21 | 4 | 5 | 0.840 | 0.808 | 0.824 |
| Rechazado | 0 | 0 | 2 | — | 0.000 | — |

La estimación **insesgada** para el corpus es el estrato aleatorio: precisión 4/4 = 1.0
(IC95 [0.51, 1.0]) y recall 4/10 = 0.40 (IC95 [0.17, 0.69]). Es decir: en texto libre el sistema
es conservador y recupera aproximadamente 2 de cada 5 promesas, con muy pocos falsos positivos.

### 11.5 Por hablante

| Hablante | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| Andrés Manuel López Obrador | 8 | 0 | 2 | 1.000 | 0.800 | 0.889 |
| Javier Milei | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Nicolás Maduro | 2 | 4 | 4 | 0.333 | 0.333 | 0.333 |
| Pedro Sánchez | 8 | 0 | 3 | 1.000 | 0.727 | 0.842 |
| Sebastián Piñera Echenique | 4 | 0 | 4 | 1.000 | 0.500 | 0.667 |

Maduro concentra todos los falsos positivos: retórica con `vamos a` sin acción de gobierno
(*"vamos a enviarle un aplauso"*, *"vamos a cuidarnos"*, *"vamos a iniciar un minuto de
silencio"*). Son actos de habla performativos, no promesas verificables: el sistema no distingue
un aplauso de un hospital porque ambos son NPs con `vamos a`.

### 11.6 Tipología de errores

Falsos negativos típicos:

* promesa con sujeto en tercera persona sin marcador atributivo: *"este gobierno está
  comprometido con levantar el 5.º pilar del Estado del Bienestar"*;
* compromiso sin verbo: *"lo que les traslado es que acuerdo habrá"*;
* nominalización en lugar de verbo: *"proponemos la creación de un fondo rotatorio"*.

Falsos positivos típicos:

* `voy a` + verbo de contenido usado como estructura expositiva: *"Entonces voy a arrancar en
  1956"*, *"lo voy a poner de manera más fundamentalista"*;
* atribución ambigua donde la oración habla de terceros: *"van a pagar por haber apretado a la
  gente"*.

Estabilidad del etiquetado entre configuraciones: κ(A, B) = **0.956** y κ(A, D) = **0.739**, es
decir, los filtros cambian decisiones de forma sustantiva (no son cosméticos) pero no aleatoria.

---

## 12. Limitaciones

1. **Gold de un anotador y pequeño**: 205 oraciones, 38 positivas. Los intervalos de confianza son
   anchos y las fronteras semánticas ("¿`vamos a informar` es promesa?") son discutibles.
2. **Prevalencia baja**: en texto libre sólo ~8 % de las oraciones contiene una promesa; con 125
   oraciones aleatorias la estimación de recall tiene ±25 puntos.
3. **Recall limitado por el léxico**: lo que no tiene disparador no se detecta. Ampliar el léxico
   sube recall y baja precisión (ver §11.2).
4. **Slots de superficie**: no se resuelve el programa presupuestal ni la coherencia
   monto/unidad; `fecha` relativa no se ancla a la fecha del discurso.
5. **Atribución**: la regla de sujeto atribuible protege la precisión y penaliza el estilo
   impersonal (*"se va a construir"*), frecuente en Piñera.
6. **Fechas del corpus hasta 2026**: cualquier evaluación de cumplimiento es ilustrativa.
7. **Maduro** (10 eventos, 3,460 oraciones) no da para comparación por hablante.

---

## 13. Reproducibilidad

```bash
source ../env/bin/activate

python -m ei_scripts.build_events                  # 146 s → artifacts/{events,events_by_doc,stats}
python -m ei_scripts.build_events --limit 300      # prueba rápida
python -m ei_scripts.build_events --no-cyk --no-slots --tag ablacion
python -m ei_scripts.evaluate --markdown           # métricas + artifacts/tables.md
```

La libreta `EI_demo.ipynb` recorre el pipeline completo sobre un discurso y después sobre el
corpus (funnel, distribuciones, dedup, evaluación y cruce con BM25 del módulo de RI).

---

## 14. Conexión con el resto del proyecto

* **RI (módulo 1)**: los eventos comparten exactamente el `doc_id` del índice invertido, así que
  se puede recuperar con BM25 y filtrar por `event_type`, `frame.lugar` o `frame.fecha`. El propio
  `events_by_doc.jsonl` es un índice invertido con otro esquema de campos
  (`tipo/lugar/año → eventos`).
* **Entidades y relaciones (módulo 3)**: los gazetteers de lugares y organizaciones y los spans de
  beneficiario son el insumo directo del grafo.
* **Resumen (módulo 4)**: los eventos dan un criterio de selección de oraciones ("resume sólo lo
  que promete").
* **Origen de las cifras**: `artifacts/stats.json` (volumen y distribuciones),
  `artifacts/eval.json` (métricas), `artifacts/tables.md` (tablas de esta página).
