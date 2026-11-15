# Plan de acción — Observatorio del discurso presidencial en paises hispanohablantes

> **RFC-001 · Alcance cerrado**
> Documento de planeación y registro del sistema construido. Define objetivo, alcance,
> decisiones de diseño y criterios de aceptación.
>
> Las referencias técnicas están en los documentos por módulo y los artefactos en
> `*_scripts/artifacts/`.

| Campo | Valor |
|---|---|
| Identificador | RFC-001 |
| Estado | Aceptado — alcance congelado |
| Versión | 1.0 |
| Fecha | 2026-11-15 |
| Alcance | Módulos 1 a 7 |
| Dataset | `unified/presidentes_latam_es.parquet` (`unified/README.md`) |
| Documentos por módulo | `ei_scripts/doc_ei.md`, `er_scripts/doc_er.md`, `rat_scripts/rat_explained.md`, `mt_scripts/mt_explained.md`, `est_scripts/est_explained.md` |

---

## 1. Resumen

Este proyecto construye un **observatorio del discurso presidencial en español**: un sistema que
recupera, estructura, sintetiza y monitorea lo que cinco presidentes dicen a lo largo del tiempo,
usando exclusivamente **métodos simbólicos, estadísticos y algoritmos clásicos de PLN**.

El dataset es de 4,531 discursos de cinco líderes hispanohablantes (Argentina, Chile, España, México y
Venezuela), con un rango temporal de 2015 a 2026.

El proyecto **cubre el temario completo** de la asignatura (recuperación de información, extracción de
información, resumen automático y minería de textos) y se detiene en el **módulo 7**.

El entregable son las **libretas de demostración** de cada módulo, acompañadas de la documentación
técnica que sostiene el reporte escrito.

---

## 2. Contexto y motivación

### 2.1 El problema

El discurso político presidencial es un objeto de estudio con tres propiedades incómodas:

1. **Es masivo.** 4,531 discursos y decenas de millones de palabras. Nadie los lee completos.
2. **Es repetitivo.** Las mismas fórmulas, los mismos compromisos, los mismos nombres.
3. **Es cambiante.** Lo que un líder decía en 2019 no es lo que dice en 2024.

Un observatorio responde a eso con tres operaciones: **encontrar** el pasaje relevante, **estructurar**
lo que se promete y a quién se menciona, y **ver cómo cambia** el discurso en el tiempo.

### 2.2 Por qué métodos simbólicos y estadísticos

La restricción del proyecto es explícita: sin IA generativa, sin deep learning, sin transformers y sin
embeddings neuronales.

Esto **no es una limitación asumida a regañadientes: es la decisión de diseño central**. El producto de
un observatorio son afirmaciones verificables del tipo *"este líder habla más de seguridad que aquel"*.
Una afirmación de ese tipo necesita **trazabilidad**: poder volver del número al término, de la oración
al discurso y del discurso al documento fuente.

| Requisito del observatorio | Método simbólico/estadístico | Modelo neuronal |
|---|---|---|
| Trazabilidad de cada cifra | Total: se reconstruye desde el índice | Opaca |
| Reproducibilidad bit a bit | Sí, con semillas fijas | Deriva entre versiones |
| Coste de operación | Sin GPU, sin API, laptop | GPU y/o servicio externo |
| Superficie de error | Conocida y enumerable | Difusa |
| Generalización semántica | **Débil** | Fuerte |

Se elige la primera columna y se declara la última fila como limitación. El observatorio no pretende
entender el significado; pretende **medir forma, frecuencia, estructura y evolución** con precisión
auditable.

---

## 3. Objetivo

### 3.1 Objetivo general

Construir un instrumento de **análisis y monitoreo** del discurso presidencial en español que permita
describir qué dice cada líder, cómo lo dice, a quién y qué se compromete a hacer, y cómo cambia ese
discurso en el tiempo — con cada resultado **trazable a su evidencia textual** y sin recurrir a modelos
generativos ni neuronales.

### 3.2 Objetivos específicos

| # | Objetivo | Pilar | Módulo |
|---|---|---|---|
| O1 | Localizar discursos y pasajes por consulta, con ranking y evaluación | Recuperación | 1 |
| O2 | Extraer compromisos y anuncios con sus participantes y plazos | Extracción | 2 |
| O3 | Construir la red de entidades y relaciones de cada discurso y del corpus | Extracción | 3 |
| O4 | Producir resúmenes extractivos fieles de un discurso individual | Síntesis | 4 |
| O5 | Producir la síntesis anual de la narrativa de un líder | Síntesis | 5 |
| O6 | Descubrir temas latentes y medir su deriva temporal | Monitoreo | 6 |
| O7 | Caracterizar el estilo de cada líder con rasgos no temáticos | Caracterización | 7 |

### 3.3 No objetivos

Declarados para acotar el sistema a lo que efectivamente resuelve:

- **No** clasificar ideología, intención, veracidad ni sentimiento de los hablantes.
- **No** comparar países. El dataset confunde país, persona e ideología (§7.1).
- **No** generar texto en lenguaje natural nuevo. Todo resumen es **extractivo**: selecciona oraciones
  existentes.
- **No** predecir el futuro ni modelar audiencias, efectos o recepción.
- **No** resolver el español general. Se modelan **dominios acotados**: lo que aparece en este dataset.

---

## 4. Alcance

### 4.1 Módulos cubiertos: 1 a 7

Cinco pilares funcionales sobre una misma base documental:

```
                    ┌──────────────────────────────────────────┐
                    │  DATASET UNIFICADO · 4,531 discursos     │
                    │  presidentes_latam_es.parquet            │
                    └───────────────────┬──────────────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │  M1 · Índice y recuperación   │  ← fuente única de
                        │  preproceso · postings · IDF  │    verdad para M5, M6, M7
                        └───┬───────────┬───────────┬───┘
                            │           │           │
              ┌─────────────▼──┐  ┌─────▼──────┐  ┌─▼─────────────┐
              │ M2 · Promesas  │  │ M3 · Grafo │  │ M4 · Resumen  │
              │ autómata+CFG   │  │ entidades  │  │ extractivo    │
              └────────────────┘  └────────────┘  └─┬─────────────┘
                                                     │
                                          ┌──────────▼───────────┐
                                          │ M5 · Multi-documento │
                                          │ anual (MMR+k-means)  │
                                          └──────────┬───────────┘
                                                     │
                        ┌────────────────────────────┴─────────────┐
                        │ M6 · Tópicos y deriva temporal           │
                        │ M7 · Estilometría y perfil               │
                        └──────────────────────────────────────────┘
```

<br>

| Pilar | Módulos | Pregunta que responde |
|---|---|---|
| **A · Recuperación** | M1 | ¿Dónde se dijo esto? |
| **B · Extracción** | M2, M3 | ¿Qué se promete y quién aparece? |
| **C · Síntesis** | M4, M5 | ¿De qué va este discurso y este año? |
| **D · Monitoreo** | M6 | ¿De qué se habla y cómo cambia? |
| **E · Caracterización** | M7 | ¿Cómo habla cada líder? |

### 4.2 Cobertura del temario

| Unidad | Contenido | Cubierta por | Estado |
|---|---|---|---|
| 2.1 | Antecedentes, concepto, PLN en RI, aplicaciones | M1 | Completa |
| 2.2.1–2.2.2 | Preproceso e indexación | M1 | Completa |
| 2.2.3–2.2.4 | Recuperación y recuperación semántica | M1 | Completa |
| 2.2.5 | Medidas de evaluación | M1, M4 | Completa |
| 3.1 | Concepto y diferencias RI/EI | M2 | Completa |
| 3.2.1–3.2.3 | Expresiones regulares, autómatas, GIC | M2 | Completa |
| 3.3.1–3.3.2 | Reconocimiento, identificación y clasificación de nombres | M3 | Completa |
| 3.4.1 | Reconocimiento de eventos | M2 | Completa |
| 3.5 | Extracción de relaciones | M3 | Completa |
| 3.6 | Algoritmos de aprendizaje de máquinas | M7 | Completa |
| 4.1–4.3 | Concepto, criterios, etapas del RAT | M4 | Completa |
| 4.4.1–4.4.6 | Métodos estadísticos, selección, discursivos, plantillas | M4 | Completa |
| 4.5.1 | Métricas de evaluación | M4 | Completa |
| 4.6.1–4.6.2 | Multi-documento: problemática y métodos | M5 | Completa |
| 5.1 | Introducción, concepto, aplicaciones | M6 | Completa |
| 5.2 | Técnicas básicas de MT | M6 | Completa |
| 5.3 | Algoritmos y modelos | M6 | Completa |
| 5.4–5.5 | Herramientas y software de MT | M6 | Completa |
| 5.6 | Más allá de la MT | M6 (dinámica temporal) | Completa |

**Cobertura del temario: 100%.**

---

## 5. Qué se construyó

Cada módulo es un paquete ejecutable con pipeline propio, artefactos persistidos, libreta de
demostración y documento con fórmulas y evaluación.

| Módulo | Paquete | Libreta | Documento | Artefactos |
|---|---|---|---|---|
| M1 · Recuperación | `ri_scripts/` | `RI_demo.ipynb` | — | `artifacts/` |
| M2 · Promesas | `ei_scripts/` | `EI_demo.ipynb` | `doc_ei.md` | `artifacts/` |
| M3 · Grafo | `er_scripts/` | `ER_demo.ipynb` | `doc_er.md` | `artifacts/` |
| M4–M5 · Resumen | `rat_scripts/` | `RAT_demo.ipynb` | `rat_explained.md` | `artifacts/` + `artifacts/multidoc/` |
| M6 · Tópicos | `mt_scripts/` | `MT_demo.ipynb` | `mt_explained.md` | `artifacts/` |
| M7 · Estilometría | `est_scripts/` | `EST_demo.ipynb` | `est_explained.md` | `artifacts/` |

### 5.1 M1 · Índice y recuperación

Índice invertido propio sobre los 4,531 discursos, con postings comprimidos, vocabulario, IDF y matriz
TF-IDF global. Tres modos de recuperación —booleano, TF-IDF y BM25 con parámetros `k1`/`b` persistidos
en `index_meta.json`— y evaluación con juicios de relevancia manuales.

**Rol arquitectónico:** es la **fuente única de verdad**. `corpus.jsonl`, `idf.npy` y `w_global.npz`
alimentan directamente M5, M6 y M7. Ningún módulo reconstruye el dataset ni el vocabulario.

### 5.2 M2 · Extracción de anuncios y promesas

Definición operativa de *promesa*: oración en primera persona donde el hablante se compromete a una
acción futura. Cadena simbólica completa:

`Aho-Corasick (619 patrones, 727 estados)` → `DFA de negación` → `slots por regex` → `chunker` →
`CFG en CNF + CYK` → `plantilla de evento` → `MinHash + LSH` para casi duplicados.

Salida: **11,052 eventos** con esquema de 20+ campos, cada uno con posición de carácter, oración de
evidencia y score de confianza por suma ponderada de evidencias.

### 5.3 M3 · Grafo de entidades y relaciones

Tres capas de menciones (gazetteer vía autómata, rachas de mayúsculas, cabeceras de organización),
coreferencia por reglas, y relaciones por coocurrencia en ventana, patrón sujeto–verbo–objeto y cargo.
Grafo exportado en `graph.json` y `graph.graphml` para visualización con networkx.

### 5.4 M4 · Resumen extractivo

Siete puntuadores independientes (TF-IDF de cabeza, centroide, Luhn, TextRank, LexRank, LSA,
componente discursivo), combinados en un **híbrido** con selección por MMR. La comparación entre métodos
usa ROUGE-1/2/L sobre referencias humanas y métricas intrínsecas (compresión, cobertura, redundancia).

### 5.5 M5 · Resumen multi-documento por año

Agrupamiento k-means esférico sobre la matriz TF-IDF global, resumen por medoides, fusión con MMR y
cuotas por grupo, y detección de tópicos comunes por frecuencia documental. Produce la **narrativa
anual** de un líder: ~390 palabras, 10–12 oraciones, 5–8 temas.

### 5.6 M6 · Tópicos y evolución temporal

Vocabulario de 8,000 términos con poda explícita de *puesta en escena*. Barrido de tres familias de
modelos (LSA, NMF, LDA) × cinco valores de `k`. Coherencia implementada desde cero (NPMI, `c_v`,
UMass). Dos vías de evolución temporal: agregación de θ por periodo, y ventanas independientes con
alineación de tópicos por asignación húngara para medir **deriva**.

### 5.7 M7 · Estilometría y clasificación de hablante

Cincuenta rasgos por documento en cuatro bloques —forma, tema, periodo y contenido diagnóstico— con
diseño de ablación de cinco bloques. El objetivo es separar **cómo habla** de **de qué habla** y de
**cuándo habló**, controlando explícitamente la fuga de información.

El dataset **no tiene etiquetas de tipo de discurso**, por lo que esta entrega es **estilometría +
clasificación de hablante**, que cubre 3.6 (algoritmos de aprendizaje) y 5.2 (técnicas básicas de MT).

---

## 6. Decisiones de diseño

Registro de las decisiones que condicionan todos los módulos. Cada una tiene consecuencia observable.

| ID | Decisión | Alternativa descartada | Consecuencia |
|---|---|---|---|
| **D-1** | Métodos simbólicos y estadísticos únicamente | Modelos neuronales | Todo es auditable y reproducible; se renuncia a generalización semántica |
| **D-2** | Reportar **siempre por hablante**, nunca por país | Agregados por país | Evita el confound país = persona = ideología; obliga a normalizar |
| **D-3** | 2023–2024 es la **única** ventana de comparación entre líderes | Series largas cruzadas | Comparaciones entre líderes válidas solo en esa ventana; el resto se lee intra-líder |
| **D-4** | M1 es la **fuente única de verdad** (dataset, IDF, matriz TF-IDF) | Corpus por módulo | Coherencia entre módulos; los artefactos de RI son dependencia obligatoria |
| **D-5** | **NMF** como modelo de tópicos principal; LDA como contraste | LDA como principal | Determinismo y mejor coherencia; se evita discutir el adjetivo "generativo" |
| **D-6** | Eliminar **puesta en escena** del vocabulario de MT | Usar todo el vocabulario | Evita tópicos basura de protocolo; puede borrar señal discursiva real |
| **D-7** | Toda cifra del reporte **cita su artefacto** | Cifras en prosa | Reporte verificable; nada es "de memoria" |
| **D-8** | En M7, los n-gramas de carácter se reportan como **diagnóstico de fuga**, no como resultado | Reportarlos como F1 del sistema | Se evita vender una tautología (identificar hablante) como hallazgo |
| **D-9** | Las referencias de ROUGE son **abstractivas** y solo comparan métodos | Referencias extractivas | ROUGE se interpreta como comparación relativa, no calidad absoluta |
| **D-10** | Un evento/promesa solo se cuenta si tiene **evidencia citable** (posición de carácter) | Contar agregados sin posición | Cada número del observatorio es navegable hasta la oración fuente |

---

## 7. Restricciones y limitaciones

Se declaran por escrito porque el reporte no se defiende ocultándolas, se defiende nombrándolas.

### 7.1 Del dataset

| # | Limitación | Efecto |
|---|---|---|
| L1 | **5 hablantes, 1 por país** → país = persona = ideología confundidos | Ninguna afirmación puede atribuirse a un país ni a una ideología; solo a un líder o a un estilo |
| L2 | **Desbalance severo** (AMLO 2,294 · Sánchez 1,136 · Piñera 884 · Milei 187 · Maduro 30) | Toda métrica agregada está dominada por AMLO; se reporta por hablante con normalización |
| L3 | **Periodos casi disjuntos** | La única ventana de comparación limpia entre líderes es 2023–2024 |
| L4 | **Calidad heterogénea** (AMLO desde CSV, el resto transcripciones) | Afecta segmentación, puntuación y por tanto estilometría y resumen |
| L5 | **Sin gold de temas ni de tipo de discurso** | M6 se evalúa con gold pequeño; M7 no puede clasificar tipo de discurso |
| L6 | **Maduro: 30 discursos en 8 años** | Su modelo se marca de baja confianza y no se mezcla en agregados sin nota |
| L7 | **Duplicados no eliminados globalmente** | Inflan ligeramente tópicos y series temporales; se mitiga dentro de M2 y como control en M7 |

### 7.2 De los métodos

| # | Limitación | Consecuencia práctica |
|---|---|---|
| M-1 | Sin generalización semántica | El buscador recupera por forma: sinónimos y paráfrasis se pierden salvo que estén en un léxico |
| M-2 | Sin negación ni ironía fuera de los DFA construidos | Un evento negado solo se filtra si la regla lo cubre |
| M-3 | Resumen **solo extractivo** | No hay abstracción; el resumen son oraciones del original |
| M-4 | Reglas frágiles por construcción | Gazetteers y léxicos escritos a mano no cubren entidades o términos nuevos |
| M-5 | Coreferencia parcial | Se resuelven apuestos y títulos, no la elisión del sujeto (*pro-drop*) |
| M-6 | Relaciones verbales con precisión baja | El grafo describe coocurrencia con fiabilidad, no semántica verbal |
| M-7 | Tópicos no interpretados automáticamente | Las etiquetas de tópico son lectura humana, y así se declara |
| M-8 | NMF no es probabilístico | "Participación" es descriptiva, no una posterior |

### 7.3 De la evaluación

| # | Limitación | Efecto |
|---|---|---|
| E-1 | Gold de M2 muestreado (~125 oraciones + estratos) | IC amplios; el F1 tiene incertidumbre real |
| E-2 | Gold de M3 pequeño (80 menciones, 53 relaciones) | La precisión de relaciones no distingue bien entre tipos |
| E-3 | 5 referencias de ROUGE, abstractivas | ROUGE es comparación entre métodos, no calidad absoluta |
| E-4 | Gold de temas de 47 discursos | La pureza es una señal, no una verdad |
| E-5 | Sin evaluación de extremo a extremo | No se mide el sistema completo como producto, solo por módulo |

---

## 8. Riesgos y mitigaciones

| Riesgo | Prob. | Impacto | Mitigación aplicada |
|---|---|---|---|
| Métricas infladas por el desbalance | Alta | Alto | Reporte por hablante + normalización + macro-F1 en vez de accuracy |
| Identificar hablante presentado como hallazgo | Alta | Alto | D-8: bloque controlado como resultado, n-gramas de carácter como diagnóstico de fuga |
| Gold demasiado pequeño para sostener conclusiones | Media | Medio | IC reportados; las cifras se presentan como relativas entre configuraciones |
| Tópicos contaminados por protocolo | Alta | Medio | D-6: poda de puesta en escena + coherencia + etiquetado humano |
| Comparaciones temporales inválidas | Media | Alto | D-3: solo 2023–2024 cruza líderes; el resto intra-líder con advertencia |
| Conclusiones leídas como juicio político | Media | Alto | §3.3 no objetivos + L1 + lenguaje de "texto, no de personalidad" |

---

## 9. Criterios de aceptación

El alcance se considera cumplido cuando:

1. Los seis paquetes se ejecutan desde cero y producen sus artefactos de forma determinista.
2. Cada módulo tiene documento con fórmulas, parámetros y evaluación reproducible.
3. La cobertura del temario es completa (§4.2) y verificable unidad por unidad.
4. Toda cifra del reporte es localizable en un artefacto (D-7).
5. Toda cifra agregada existe también desagregada por hablante (D-2).
6. Las limitaciones de §7 aparecen en el reporte con el mismo nivel de detalle que los resultados.
7. La restricción de §2.2 se verifica: ningún componente usa IA generativa, deep learning,
   transformers ni embeddings neuronales.
