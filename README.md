# Observatorio del discurso presidencial en español

Qué dicen, cómo lo dicen y cómo cambia lo que dicen cinco presidentes hispanohablantes — medido con
métodos clásicos de procesamiento de lenguaje natural.

---

## Antes de empezar

Este es un proyecto **académico**. Su propósito es entender **cómo funcionan los métodos matemáticos y
algorítmicos** aplicados al seguimiento, análisis, resumen y extracción de información en discursos
públicos.

**No es un proyecto político y no emite juicios morales, sociales ni ideológicos** sobre ninguno de los
oradores. No evalúa la calidad de sus gobiernos, la veracidad de sus afirmaciones ni sus posiciones. Los
resultados que verás son mediciones sobre **texto**: frecuencias, estructuras, similitudes y series
temporales. Cuando el sistema dice que un líder usa más el plural que otro, describe una propiedad del
discurso, no una virtud ni un defecto de la persona.

Tampoco compara países. El dataset tiene un líder por país, así que país, persona e ideología quedan
confundidos en los mismos datos; toda conclusión se limita al **estilo y al discurso de un hablante**.

---

## De qué va esto

Los discursos presidenciales son un volumen de texto imposible de leer completo: **4,531 discursos** y
decenas de millones de palabras, con el mismo vocabulario gastado una y otra vez durante años.

Este proyecto aborda ese problema con las herramientas de la lingüística computacional clásica. El hilo
conductor es una pregunta práctica: **¿qué se puede saber de miles de discursos sin leerlos uno por
uno?**

La respuesta se organiza en cinco operaciones:

| Pregunta | Respuesta del sistema |
|---|---|
| ¿Dónde se dijo esto? | Un buscador con índice invertido y ranking (BM25, TF-IDF) |
| ¿Qué se promete y quién aparece? | Extracción de compromisos con responsable, lugar y plazo, y un grafo de entidades y relaciones |
| ¿De qué va este discurso? | Resúmenes extractivos por selección de oraciones |
| ¿De qué se habla y cómo cambia? | Temas latentes descubiertos sin etiquetas previas y su evolución año con año |
| ¿Cómo habla cada líder? | Perfiles estilométricos con rasgos que no dependen del tema |

En el camino aparecen los problemas reales de la disciplina: cómo se preprocesa texto en español, cómo
se mide si un sistema de recuperación sirve, cómo se evalúa un resumen cuando no existe un resumen
"correcto", cómo se elige el número de temas de un corpus, y cómo se distingue el estilo de alguien de
lo que ese alguien decidió decir.

### Una restricción que define el proyecto

Nada de IA generativa. Nada de redes neuronales, transformers ni embeddings neuronales. **Todo el
sistema es simbólico y estadístico**: expresiones regulares, autómatas, gramáticas, índices invertidos,
TF-IDF, BM25, SVD, NMF, LDA, TextRank, MinHash, SVM.

No es una limitación incómoda: es lo que hace que cada número sea **auditable**. Cualquier cifra del
proyecto se puede reconstruir desde el término hasta la oración y el discurso que la originó. Un modelo
opaco diría "de esto se habla mucho"; este sistema puede mostrar exactamente dónde y por qué.

---

## El dataset

4,531 discursos en español, un líder por país, 2015–2026.

| País | Líder | Discursos | Periodo |
|---|---|---:|---|
| México | Andrés Manuel López Obrador | 2,294 | 2018–2024 |
| España | Pedro Sánchez | 1,136 | 2018–2026 |
| Chile | Sebastián Piñera | 884 | 2018–2022 |
| Argentina | Javier Milei | 187 | 2023–2026 |
| Venezuela | Nicolás Maduro | 30 | 2015–2023 |

El desbalance es severo y está presente en todo el proyecto: cualquier promedio del corpus es, en
realidad, un promedio de AMLO. Por eso cada resultado se reporta **por hablante**, nunca solo agregado.
Composición, fuentes y detalles: [`unified/README.md`](unified/README.md).

---

## Los siete módulos

| # | Módulo | Técnicas | Documento |
|---|---|---|---|
| 1 | Índice y recuperación | Índice invertido, TF-IDF, BM25, evaluación con juicios de relevancia | — |
| 2 | Anuncios y promesas | Aho-Corasick, DFA de negación, regex, gramática + CYK, MinHash/LSH | [`doc_ei.md`](ei_scripts/doc_ei.md) |
| 3 | Grafo de entidades | Gazetteer, patrones de mención, coreferencia por reglas, relaciones por coocurrencia | [`doc_er.md`](er_scripts/doc_er.md) |
| 4 | Resumen extractivo | TF-IDF, centroide, Luhn, TextRank, LexRank, LSA, MMR, ROUGE | [`rat_explained.md`](rat_scripts/rat_explained.md) |
| 5 | Resumen multi-documento | k-means esférico, medoides, MMR con cuotas, tópicos comunes | [`rat_explained.md`](rat_scripts/rat_explained.md) |
| 6 | Temas y evolución temporal | LSA, NMF, LDA, coherencia NPMI/`c_v`, alineación de tópicos | [`mt_explained.md`](mt_scripts/mt_explained.md) |
| 7 | Estilometría | 50 rasgos de forma, SVM/Naive Bayes/regresión logística, ablación de fuga | [`est_explained.md`](est_scripts/est_explained.md) |

Los módulos 1 al 7 cubren el temario completo del curso: recuperación de información, extracción de
información, resumen automático y minería de textos.

---

## Cómo está organizado

```
proyecto/
├── unified/              dataset unificado y su documentación
│
├── ri_scripts/           M1 · índice, vocabulario, postings, buscadores
├── ei_scripts/           M2 · autómata, slots, gramática CYK, eventos
├── er_scripts/           M3 · menciones, coreferencia, grafo de relaciones
├── rat_scripts/          M4-M5 · puntuadores, selección, resumen anual
├── mt_scripts/           M6 · tópicos, coherencia, series temporales
├── est_scripts/          M7 · rasgos estilométricos y clasificación
│
├── RI_demo.ipynb         libretas de demostración, una por módulo
├── EI_demo.ipynb
├── ER_demo.ipynb
├── RAT_demo.ipynb
├── MT_demo.ipynb
├── EST_demo.ipynb
│
└── architecture/         plan de acción y alcance del proyecto
```

Cada carpeta de módulo sigue el mismo patrón: un `config.py`, el código del pipeline, un `artifacts/`
con los resultados ya calculados y, cuando aplica, un documento `.md` con las fórmulas, los parámetros
y la evaluación.

---

## Por dónde entrar

**Si solo quieres ver qué hace:** abre cualquiera de las libretas `*_demo.ipynb`. Ya están ejecutadas y
cargan los artefactos precalculados, así que se leen sin correr nada.

**Si quieres ejecutarlo:**

```bash
source ../env/bin/activate

python ri_scripts/build_all.py        # índice y vectores (base para el resto)
python ei_scripts/build_events.py     # eventos y promesas
python er_scripts/build_graph.py      # grafo de entidades
python rat_scripts/build_summaries.py # resúmenes individuales
python rat_scripts/build_multidoc.py  # resúmenes anuales
python mt_scripts/build_topics.py     # temas y series temporales
python est_scripts/build_models.py    # perfiles estilométricos
```

Todo es determinista, sin GPU y sin servicios externos; el pipeline completo toma unos minutos.

**Si quieres entender el método:** cada módulo tiene su documento y termina conectando con los demás.
Para el diseño general y las decisiones de fondo, [`architecture/plan_accion.md`](architecture/plan_accion.md).

---

## Qué esperar y qué no

**Sí:** mediciones trazables, comparaciones entre métodos con sus métricas, evaluación honesta con
gold pequeño e intervalos de confianza, y las limitaciones declaradas junto a los resultados.

**No:** comprensión semántica profunda. El sistema recupera por forma, no por significado; resume
seleccionando oraciones en vez de redactando; y no detecta ironía ni matices que no estén escritos como
regla. Esas ausencias están documentadas, no escondidas.

---

*Proyecto académico de PLN. Métodos simbólicos y estadísticos exclusivamente. Sin IA generativa.*
