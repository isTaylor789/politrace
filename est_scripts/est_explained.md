# Módulo 7-lite — Estilometría y clasificación de hablante

**Proyecto:** Observatorio del discurso presidencial en español
**Unidades del temario cubiertas:** 5.2 (técnicas de MT: representación, frecuencias, medidas) y
3.6 / 5.3 (algoritmos clásicos de aprendizaje: Naive Bayes, SVM, regresión logística).
**Restricciones respetadas:** sin IA generativa, sin deep learning, sin transformers, sin embeddings
neuronales. Todo son rasgos contables, TF-IDF y clasificadores lineales clásicos.

**Qué es y qué no es.** Este módulo mide **cómo está escrito** un discurso y si eso basta para
identificar a quién lo pronunció. No infiere personalidad, ideología, emociones ni rasgos psicológicos:
un discurso (y menos una transcripción) no permite eso, y afirmarlo sería pseudociencia. Lo que sí
permite es cuantificar la **distintividad textual** de cada líder y, sobre todo, medir cuánto de esa
distintividad es forma y cuánto es tema o periodo.

**Qué añade sobre el módulo 6.** El 6 describe temas latentes; este módulo aporta tres cosas que el 6
no puede dar: (1) una métrica **predictiva** con datos retenidos (macro-F1 out-of-fold), (2) la
separación explícita **forma / contenido / periodo** mediante ablación, que convierte el confundido
país = persona en un número, y (3) un **perfil estilométrico** por líder interpretable sin entrenar nada.

---

## 1. Corpus y validación

| | |
|---|---|
| Documentos | 4,525 (se descartan 6 con < 120 palabras) |
| Clases | 5 hablantes; AMLO 2,293 (50.7 %), Sánchez 1,133, Piñera 883, Milei 186, Maduro 30 |
| Accuracy de la clase mayoritaria | 0.507 → **no se usa accuracy como métrica principal** |
| Folds | 5, `StratifiedGroupKFold` |
| Grupos de casi-duplicados | 4,470 (mayor grupo: 5 documentos) |
| Umbral de duplicado | coseno TF-IDF ≥ 0.80 |

**Por qué grupos.** Los discursos se repiten (el mismo acto publicado dos veces, la misma alocución en
dos giras). Si un duplicado cae en train y su copia en test, el F1 mide memoria, no generalización. Se
agrupan los documentos con coseno ≥ 0.80 (vecindario exacto sobre `w_global`) y los textos idénticos por
hash del texto plegado; `StratifiedGroupKFold` mantiene cada grupo en un solo fold y conserva la
proporción de clases.

**Por qué macro-F1.** Es la media no ponderada del F1 por clase: Maduro (30 documentos) pesa lo mismo
que AMLO (2,293). Un clasificador que ignore a Maduro tendría accuracy 0.51 y macro-F1 ~0.20.

---

## 2. Rasgos estilométricos (≈50 por documento)

Tradición de estilometría: Mosteller & Wallace (1964) con las palabras de función de los *Federalist
Papers*, y Burrows (2002) con el análisis de componentes principales sobre frecuencias léxicas. Aquí
todo es contable y explicable.

### 2.1 Riqueza y longitud

- **TTR** = tipos / tokens.
- **MTLD** (McCarthy, 2005): se recorre el texto acumulando TTR; cada vez que cae por debajo de
  `t = 0.72` se cierra un factor y se reinicia el conteo. Al final:

  $$\text{MTLD} = \frac{N}{\#\text{factores} + \frac{1-\text{TTR}_{\text{resto}}}{1-t}}$$

  Se calcula en las dos direcciones y se promedia. Es robusto al tamaño del texto, a diferencia de TTR
  (que cae mecánicamente al alargar el documento).
- **Yule's K** (Yule, 1944): `K = 10⁴ (Σ i²·V_i − N)/N²`, con `V_i` = nº de tipos con frecuencia `i`.
  Mide repetición léxica; más alto = más repetitivo.
- **Hápax**: proporción de tipos que aparecen una sola vez.
- **Longitud**: palabras, oraciones, palabras/oración, desviación de la longitud de oración, proporción
  de oraciones cortas (≤ 8 palabras) y largas (≥ 35), letras/palabra, palabras ≥ 7 letras.
- **Legibilidad**: Fernández-Huerta `206.84 − 0.60·P − 1.02·F` y Szigriszt-Pazos
  `206.835 − 62.3·(S/Pal) − Pal/Orac`, con `P` = sílabas/100 palabras, `F` = oraciones/100 palabras.
  Las sílabas se aproximan contando grupos vocálicos (documentado: no hay silabificador léxico).

### 2.2 Puntuación y tipografía (por 1,000 palabras)

Comas, puntos, punto y coma, dos puntos, interrogación, exclamación, comillas (rectas y angulares),
paréntesis, guiones, puntos suspensivos, saltos de línea, ratio de mayúsculas no iniciales (proxy de
nombres propios y siglas) y ratio de dígitos.

### 2.3 Deixis y marcadores

Frecuencia por 1,000 palabras de pronombres de primera singular (*yo, me, mi…*), primera plural
(*nosotros, nos…*), segunda (*tú, usted, ustedes…*) y tercera (*él, ella, le, les, se…*);
`(yo − nosotros)/(yo + nosotros)`; muletillas orales (*bueno, entonces, mire, digamos…*); vocativos
(*amigas, compañeros, señorías, compatriotas…*); atenuadores (*creo, quizá, me parece…*); y frases
(*o sea*, *por supuesto*, *sin embargo*, *tal vez*…).

---

## 3. Diseño experimental: cinco bloques y una ablación

El aporte metodológico es no creerse el primer F1 que salga. Se entrena el mismo conjunto de modelos con
bloques que miden cosas distintas:

| bloque | dimensión | qué mide | papel |
|---|---|---|---|
| `estilo` | 47 | riqueza, oración, puntuación, deixis | forma |
| `funcion` | 160 | palabras de función (artículos, preposiciones, pronombres, auxiliares) | forma |
| `estilo_controlado` | 207 | estilo + función | **forma (cabeza de serie)** |
| `caracteres` | 23,886 | TF-IDF de n-gramas de carácter 3-4 | diagnóstico de fuga |
| `caracteres_enmascarados` | 23,848 | ídem, con nombres y gentilicios sustituidos por `X` | diagnóstico de fuga |
| `tema` | 8,000 | TF-IDF de palabras del módulo 6 | contenido |
| `fecha` | 3 | año, mes, trimestre | periodo |
| `todo` | 8,210 | forma + contenido + periodo | cota superior |

Modelos: SVM lineal (`class_weight='balanced'`), regresión logística y Naive Bayes (complementario en
bloques de conteo/TF-IDF, gaussiano en el bloque denso de rasgos). En los bloques mixtos con rasgos de
signo no se aplica NB (requiere no negatividad).

**Enmascarado.** Los n-gramas de carácter capturan "milei", "obrador", "ucrania", "tren maya". Para
separar forma de contenido se sustituyen por `X` los nombres de los cinco hablantes y los países y
gentilicios de autorreferencia (*Argentina, chileno, España, mexicano, Venezuela…*). Si el F1 no baja,
el n-grama no estaba usando esos nombres.

---

## 4. Resultados de la ablación

| bloque | tipo | modelo | macro-F1 | ± σ | accuracy |
|---|---|---|---|---|---|
| estilo | forma | SVM | 0.839 | 0.034 | 0.967 |
| estilo | forma | NB | 0.733 | 0.016 | 0.911 |
| estilo | forma | LR | 0.850 | 0.019 | 0.964 |
| función | forma | SVM | 0.892 | 0.025 | 0.964 |
| función | forma | NB | 0.784 | 0.028 | 0.922 |
| función | forma | LR | 0.890 | 0.027 | 0.967 |
| **estilo_controlado** | **forma** | **SVM** | **0.940** | **0.004** | **0.985** |
| estilo_controlado | forma | LR | 0.935 | 0.014 | 0.986 |
| caracteres | diagnóstico | SVM | 0.998 | 0.001 | 0.999 |
| caracteres_enmascarados | diagnóstico | SVM | 0.997 | 0.002 | 0.998 |
| caracteres_enmascarados | diagnóstico | NB | 0.740 | 0.016 | 0.971 |
| tema | contenido | SVM | 0.988 | 0.008 | 0.997 |
| tema | contenido | NB | 0.821 | 0.036 | 0.981 |
| fecha | periodo | SVM | 0.311 | 0.015 | 0.436 |
| fecha | periodo | NB | 0.331 | 0.008 | 0.536 |
| todo | cota superior | SVM | 0.990 | 0.011 | 0.998 |

Lectura, en orden de importancia para el reporte:

1. **La forma controlada basta para identificar al hablante** (macro-F1 0.940, σ 0.004, accuracy 0.985).
   No hace falta hablar del tema: la manera de construir oraciones, puntuar, usar *yo/nosotros* y
   palabras de función separa a los cinco líderes. Es el resultado central del módulo.
2. **El contenido también** (0.988). Saber *de qué habla* cada líder casi lo identifica: es el
   confundido país = persona medido. No es un fallo del clasificador, es una propiedad del dataset.
3. **Los n-gramas de carácter son un atajo sucio** (0.997). Y **enmascarar los nombres no los arregla**
   (0.997): el enmascarado cubre nombres completos y gentilicios de autorreferencia, pero sobreviven
   apellidos sueltos (*López*), derivados (*mexiquense*), topónimos y consignas. Al mirar los pesos del
   SVM aparecen exactamente esos fragmentos: `lope/opez/andr` para AMLO, `bert` (*libertad*) para
   Milei, `riot/riota` (*patriota*) para Piñera, `euro/urope` para Sánchez. Por eso no se reportan como
   "estilo": se reportan como diagnóstico de fuga.
4. **El periodo solo no identifica** (0.311): los mandatos se solapan (AMLO 2018-2024, Sánchez
   2018-2026, Piñera 2018-2022), así que el año no es un atajo suficiente.
5. **La cota superior** (0.990) no mejora apenas al contenido: añadir estilo y periodo a un clasificador
   que ya ve el tema no aporta. Es la señal de que el tema es la variable dominante del corpus.

### 4.1 El mejor modelo controlado, por clase

| clase | precisión | recall | F1 | documentos |
|---|---|---|---|---|
| AMLO | 0.996 | 0.998 | 0.997 | 2,293 |
| Milei | 0.930 | 0.925 | 0.927 | 186 |
| Maduro | 0.806 | 0.833 | 0.820 | 30 |
| Sánchez | 0.977 | 0.976 | 0.977 | 1,133 |
| Piñera | 0.984 | 0.980 | 0.982 | 883 |

| real \ pred | AMLO | Milei | Maduro | Sánchez | Piñera |
|---|---|---|---|---|---|
| AMLO | 2,288 | 1 | 0 | 3 | 1 |
| Milei | 2 | 172 | 0 | 11 | 1 |
| Maduro | 2 | 2 | 25 | 0 | 1 |
| Sánchez | 4 | 9 | 3 | 1,106 | 11 |
| Piñera | 2 | 1 | 3 | 12 | 865 |

Los errores son los esperables: los 25 documentos de Maduro aciertan 25/30 con 30 ejemplos; Milei y
Sánchez se confunden entre sí (9 y 11 casos) porque ambos producen discursos institucionales densos, y
Sánchez y Piñera comparten el registro de comparecencia. AMLO casi no tiene errores porque su registro
(mañanera, oral, repetitivo) es el más distinto de todos.

### 4.2 Curva de aprendizaje

| fracción de train | macro-F1 | ± σ |
|---|---|---|
| 10 % | 0.805 | 0.031 |
| 25 % | 0.884 | 0.020 |
| 50 % | 0.929 | 0.028 |
| 75 % | 0.925 | 0.021 |
| 100 % | 0.946 | 0.013 |

Con ~10 % de los discursos (≈360) ya se alcanza 0.80: la señal estilística es fuerte y redundante. La
curva se aplana alrededor de 0.94, lo que sugiere que el techo está en la definición de "estilo" y no
en la cantidad de datos.

---

## 5. Control de ventana 2023-2024

En la ventana limpia (3 líderes con ≥ 30 discursos: AMLO 556, Sánchez 271, Milei 90; 917 documentos):

| bloque | macro-F1 | ± σ |
|---|---|---|
| estilo_controlado | 0.992 | 0.005 |
| caracteres_enmascarados | 1.000 | 0.000 |
| tema | 0.994 | 0.007 |
| fecha | 0.348 | 0.026 |

Dos lecturas, y conviene dar las dos:

- **La fecha deja de identificar** (0.348): el control funciona, el periodo ya no es un atajo.
- **La forma sube a 0.992** al restringir a tres líderes que en 2023-2024 hablan en registros
  extremos: mañanera (AMLO), mitin/cadena (Milei) y comparecencia parlamentaria (Sánchez). Esto no
  invalida el resultado: lo interpreta. La distintividad estilística es real, pero está pegada al
  **género y a la fuente** de cada transcripción. Un discurso de Milei en el Congreso se parecería más
  a Sánchez que a un mitin suyo; el dataset no tiene suficientes cruces de líder × género para
  separarlo del todo.

El `caracteres_enmascarados` con F1 = 1.000 es la señal de alarma: en 917 documentos y 3 clases, los
n-gramas encuentran fragmentos compartidos (consignas, fórmulas de apertura, citas) que identifican sin
ser estilo. Se reporta como cota superior contaminada, nunca como evidencia de estilo.

---

## 6. Perfiles estilométricos por líder (medianas)

| rasgo | AMLO | Milei | Maduro | Sánchez | Piñera |
|---|---|---|---|---|---|
| palabras | 8,429 | 3,417 | 1,808 | 1,920 | 1,096 |
| palabras/oración | 23.5 | 24.9 | 31.6 | 25.7 | 32.4 |
| oraciones ≤ 8 palabras | 0.234 | 0.125 | 0.106 | 0.147 | 0.078 |
| TTR | 0.226 | 0.318 | 0.383 | 0.332 | 0.368 |
| MTLD | 66.9 | 73.8 | 58.8 | 71.8 | 66.0 |
| Yule's K | 113.8 | 123.3 | 143.8 | 133.2 | 135.3 |
| háрах | 0.599 | 0.664 | 0.669 | 0.634 | 0.645 |
| letras/palabra | 4.77 | 4.76 | 5.08 | 4.93 | 4.86 |
| comas /1000 | 96.2 | 74.7 | 89.0 | 82.6 | 84.6 |
| interrogación /1000 | 4.25 | 1.03 | 0.00 | 0.71 | 0.00 |
| exclamación /1000 | 0.07 | 0.30 | 0.92 | 0.00 | 0.00 |
| yo /1000 | 5.38 | 3.69 | 2.45 | 5.28 | 2.87 |
| nosotros /1000 | 6.08 | 10.22 | 14.49 | 9.40 | 13.79 |
| (yo−nosotros)/(yo+nosotros) | −0.07 | −0.43 | −0.76 | −0.28 | −0.66 |
| muletillas /1000 | 10.27 | 3.82 | 2.33 | 4.29 | 1.46 |
| vocativos /1000 | 0.70 | 0.33 | 2.04 | 1.56 | 2.26 |
| atenuadores /1000 | 2.13 | 2.12 | 0.38 | 4.39 | 0.86 |
| Fernández-Huerta | 63.4 | 61.5 | 44.7 | 57.4 | 53.0 |
| "o sea" /1000 | 0.55 | 0.83 | 0.00 | 0.00 | 0.00 |

Lectura (descriptiva, no psicológica):

- **AMLO**: discursos larguísimos (mañaneras), oraciones cortas y muchas, alta oralidad (muletillas
  10.3/1000), preguntas retóricas (4.25/1000), TTR bajo (repite fórmulas) y *yo/nosotros* equilibrado.
- **Milei**: oralidad diferente ("o sea", 0.83/1000), léxico más rico (MTLD 73.8, háрах 0.664), casi sin
  mayúsculas no iniciales (1.0σ por debajo del corpus) y menos comas: sus transcripciones son más
  "habladas" y menos editadas.
- **Maduro**: las oraciones más largas (31.6 palabras), palabras más largas (5.08 letras), la
  legibilidad más baja (44.7), predominio claro de *nosotros* sobre *yo* (−0.76) y muchos vocativos.
- **Sánchez**: el mayor uso de atenuadores (4.39/1000), léxico formal largo, pocas muletillas.
- **Piñera**: oraciones largas, alta riqueza léxica, muchos vocativos, cero exclamaciones y cero
  interrogaciones en las medianas: registro escrito/leído.

Rasgos más distintivos por z-score respecto al corpus (de `tables.md`):

| hablante | más alto | más bajo |
|---|---|---|
| AMLO | muletillas (+0.7σ), oraciones cortas (+0.6σ), longitud (+0.6σ) | TTR (−0.5σ), Yule's K (−0.5σ), palabras largas (−0.5σ) |
| Milei | "o sea" (+0.9σ), MTLD (+0.7σ), háрах (+0.7σ) | mayúsculas (−1.0σ), comas (−0.9σ) |
| Maduro | palabras/oración (+1.6σ), σ de oración (+1.6σ), letras/palabra (+1.5σ) | legibilidad (−2.2σ), *yo/nosotros* (−1.1σ) |
| Sánchez | atenuadores (+0.9σ), palabras largas (+0.6σ) | "o sea" (−0.5σ), longitud (−0.5σ) |
| Piñera | palabras/oración (+1.0σ), *nosotros* (+0.9σ), TTR (+0.8σ) | muletillas (−1.1σ), oraciones cortas (−1.0σ) |

---

## 7. Discusión: qué mide y qué no

1. **No mide personalidad.** Un texto mide forma de hablar en un contexto. Afirmar extraversión,
   neuroticismo o liderazgo a partir de 30 transcripciones es invalidable; el módulo no lo intenta.
2. **La forma está contaminada por género y fuente.** Las mañaneras de AMLO son transcripciones orales
   de 8,000 palabras; las comparecencias de Sánchez son textos editados de 2,000. Parte de la señal
   estilística es del **género**, no del individuo. La ventana 2023-2024 lo hace evidente (F1 sube al
   restringir a tres registros extremos).
3. **El tema es el atajo dominante** (0.988). Es el confundido país = persona del plan base, ahora
   cuantificado: quien sepa de qué habla un discurso acierta el hablante el 98.8 % de las veces.
4. **Los n-gramas de carácter no son estilo.** 0.997 con nombres enmascarados significa que capturan
   consignas, lugares y citas. Se reportan como diagnóstico, no como resultado.
5. **Casi-duplicados parciales.** El agrupamiento por coseno ≥ 0.80 no detecta pasajes compartidos
   dentro de documentos largos; MinHash+LSH por shingles (módulo 9) lo haría mejor. El F1 de n-gramas
   está inflado por esto.
6. **Maduro (30 documentos).** Su F1 (0.82) tiene intervalos amplios; se reporta por clase y con la
   advertencia, nunca agregado sin nota.
7. **Sin etiquetas de género.** No hay gold de "tipo de discurso" (comparecencia, mitin, cumbre…). Ese
   era el alcance completo del módulo y se dejó fuera a propósito: exigía etiquetar a mano cientos de
   documentos y la etiqueta se filtraría desde el título. La ablación forma/contenido/periodo cubre la
   pregunta metodológica principal sin ese coste.

---

## 8. Reproducir y coste

```bash
source ../env/bin/activate

python est_scripts/build_models.py            # ~2 min con features cacheadas
python est_scripts/build_models.py --force    # recalcula features (~5 min)
python est_scripts/build_models.py --skip-window
```

| etapa | tiempo |
|---|---|
| features (primera vez: estilo + función + carácter + tema + duplicados) | 4 min 46 s |
| carga de la caché | 1.5 s |
| ablación (16 combinaciones × 5 folds) | ~90 s |
| curva de aprendizaje | 12 s |
| control de ventana | 15 s |
| **total en caché** | **~116 s** |

Artefactos: `style_profiles.csv` (rasgos por documento), `eval.json` (ablación, mejor modelo, ventana,
curva), `oof_predictions.jsonl` (predicciones out-of-fold del mejor modelo controlado), `tables.md`.

Determinista: `random_state = 42` en folds, SVM, LR, NB y en la submuestra de la curva.

---

## 9. Conexión con el resto del proyecto

| módulo | relación |
|---|---|
| 1 — RI | Reutiliza `w_global` para los grupos de duplicados y el espacio de tema (no reindexa). |
| 6 — MT | El bloque `tema` **es** el espacio TF-IDF del 6: aquí se usa como control de contenido, no como resultado. |
| 4-5 — RAT | Los rasgos de longitud y riqueza coinciden con los que el resumidor usa para filtrar oraciones; aquí se miden como estilo. |
| 8 — Polarización | El perfil *yo/nosotros/ellos* de este módulo es el insumo directo del análisis de encuadre. |
| 9 — Duplicados | El agrupamiento por coseno es un proxy; MinHash+LSH por shingles debería sustituirlo. |
| 10 — Perfil léxico | El perfil estilométrico y el perfil de términos distintivos son complementarios: forma vs léxico. |
| Dashboard | `style_profiles.csv` + `oof_predictions.jsonl` permiten una vista "firma estilística por líder" y marcar discursos atípicos. |
