# Tablas — estilometría y clasificación de hablante (módulo 7-lite)

Generado por `est_scripts/build_models.py`.

## 0. Corpus y validación

| documentos | descartados (<120 palabras) | grupos de casi-duplicados | mayor grupo | folds | umbral de duplicado (coseno) | accuracy de la clase mayoritaria |
|---|---|---|---|---|---|---|
| 4525 | 6 | 4470 | 5 | 5 | 0.800 | 0.507 |

La clase mayoritaria es AMLO (2,293 de 4,525): **accuracy no es una métrica útil aquí**; todo se reporta como macro-F1 y por clase.

## 1. Ablación: ¿qué aprende el clasificador?

| bloque | tipo | modelo | macro-F1 | ± σ folds | accuracy (OOF) | seg |
|---|---|---|---|---|---|---|
| estilo | forma | SVM | 0.839 | 0.034 | 0.967 | 1.900 |
| estilo | forma | NB | 0.733 | 0.016 | 0.911 | 1.400 |
| estilo | forma | LR | 0.850 | 0.019 | 0.964 | 1.900 |
| funcion | forma | SVM | 0.892 | 0.025 | 0.964 | 9.300 |
| funcion | forma | NB | 0.784 | 0.028 | 0.922 | 1.400 |
| funcion | forma | LR | 0.890 | 0.027 | 0.967 | 1.800 |
| estilo_controlado | forma (controlado) | SVM | 0.940 | 0.004 | 0.985 | 7.500 |
| estilo_controlado | forma (controlado) | LR | 0.935 | 0.014 | 0.986 | 5.900 |
| caracteres | diagnóstico (fuga de contenido) | SVM | 0.998 | 0.001 | 0.999 | 18.000 |
| caracteres_enmascarados | diagnóstico (fuga de contenido) | SVM | 0.997 | 0.002 | 0.998 | 22.700 |
| caracteres_enmascarados | diagnóstico (fuga de contenido) | NB | 0.740 | 0.016 | 0.971 | 2.100 |
| tema | contenido | SVM | 0.988 | 0.008 | 0.997 | 4.800 |
| tema | contenido | NB | 0.821 | 0.036 | 0.981 | 1.500 |
| fecha | periodo | SVM | 0.311 | 0.015 | 0.436 | 1.400 |
| fecha | periodo | NB | 0.331 | 0.008 | 0.536 | 1.400 |
| todo | cota superior | SVM | 0.990 | 0.011 | 0.998 | 7.000 |

`forma` = rasgos que no dependen del tema (riqueza, puntuación, deixis, palabras de función). `diagnóstico` = n-gramas de carácter, que **sí** filtran contenido (nombres, lugares, consignas). `contenido` = TF-IDF de tema del módulo 6. `periodo` = año/mes/trimestre. `cota superior` = forma + contenido + periodo.

**Mejor bloque controlado**: `estilo_controlado` con SVM lineal (macro-F1 = 0.940 ± 0.004).

## 2. Mejor modelo controlado: por clase y matriz de confusión

| clase | precisión | recall | F1 | docs |
|---|---|---|---|---|
| Andrés | 0.996 | 0.998 | 0.997 | 2293 |
| Javier | 0.930 | 0.925 | 0.927 | 186 |
| Nicolás | 0.806 | 0.833 | 0.820 | 30 |
| Pedro | 0.977 | 0.976 | 0.977 | 1133 |
| Sebastián | 0.984 | 0.980 | 0.982 | 883 |

| real \ pred | Andrés | Javier | Nicolás | Pedro | Sebastián |
|---|---|---|---|---|---|
| Andrés | 2288 | 1 | 0 | 3 | 1 |
| Javier | 2 | 172 | 0 | 11 | 1 |
| Nicolás | 2 | 2 | 25 | 0 | 1 |
| Pedro | 4 | 9 | 3 | 1106 | 11 |
| Sebastián | 2 | 1 | 3 | 12 | 865 |

## 3. Control de ventana 2023-2024 (3 líderes con ≥ 30 discursos)

| bloque | tipo | macro-F1 | ± σ folds | clases | docs |
|---|---|---|---|---|---|
| estilo_controlado | forma (controlado) | 0.992 | 0.005 | 3 | 917 |
| caracteres_enmascarados | diagnóstico (fuga de contenido) | 1.000 | 0.000 | 3 | 917 |
| tema | contenido | 0.994 | 0.007 | 3 | 917 |
| fecha | periodo | 0.348 | 0.026 | 3 | 917 |

En la ventana limpia el periodo deja de identificar; si el F1 de `tema` sigue alto, el clasificador usa de qué habla cada líder, no solo cuándo gobernó.

## 4. Curva de aprendizaje (bloque controlado)

| fracción de train | macro-F1 | ± σ |
|---|---|---|
| 10% | 0.805 | 0.031 |
| 25% | 0.884 | 0.020 |
| 50% | 0.929 | 0.028 |
| 75% | 0.925 | 0.021 |
| 100% | 0.946 | 0.013 |

## 5. Perfil estilométrico por líder (medianas)

| rasgo | Andrés | Javier | Nicolás | Pedro | Sebastián |
|---|---|---|---|---|---|
| palabras | 8429.000 | 3417.000 | 1807.500 | 1920.000 | 1096.000 |
| palabras/oración | 23.515 | 24.873 | 31.645 | 25.714 | 32.375 |
| σ palabras/oración | 18.570 | 16.199 | 23.339 | 17.368 | 19.256 |
| oraciones ≤8 palabras | 0.234 | 0.125 | 0.106 | 0.147 | 0.078 |
| TTR | 0.226 | 0.318 | 0.383 | 0.332 | 0.368 |
| MTLD | 66.858 | 73.824 | 58.847 | 71.756 | 65.966 |
| Yule's K | 113.781 | 123.251 | 143.816 | 133.179 | 135.286 |
| háрах | 0.599 | 0.664 | 0.669 | 0.634 | 0.645 |
| letras/palabra | 4.770 | 4.758 | 5.077 | 4.933 | 4.861 |
| palabras ≥7 letras | 0.274 | 0.277 | 0.326 | 0.305 | 0.298 |
| comas/1000 | 96.166 | 74.689 | 89.031 | 82.587 | 84.608 |
| ? /1000 | 4.250 | 1.029 | 0.000 | 0.712 | 0.000 |
| ! /1000 | 0.071 | 0.297 | 0.919 | 0.000 | 0.000 |
| mayúsculas no iniciales | 0.055 | 0.030 | 0.077 | 0.056 | 0.058 |
| yo/1000 | 5.383 | 3.692 | 2.446 | 5.284 | 2.865 |
| nosotros/1000 | 6.080 | 10.218 | 14.493 | 9.396 | 13.793 |
| (yo−nosotros)/(yo+nosotros) | -0.070 | -0.429 | -0.757 | -0.275 | -0.655 |
| muletillas/1000 | 10.273 | 3.818 | 2.326 | 4.292 | 1.462 |
| vocativos/1000 | 0.696 | 0.330 | 2.035 | 1.560 | 2.257 |
| atenuadores/1000 | 2.130 | 2.120 | 0.377 | 4.392 | 0.864 |
| Fernández-Huerta | 63.429 | 61.476 | 44.714 | 57.371 | 52.965 |
| "o sea"/1000 | 0.551 | 0.825 | 0.000 | 0.000 | 0.000 |

## 6. Rasgos más distintivos de cada líder (z de la media respecto al corpus)

| hablante | más alto | más bajo |
|---|---|---|
| Andrés | muletillas/1000 (+0.7σ), oraciones ≤8 palabras (+0.6σ), palabras (+0.6σ), ? /1000 (+0.5σ) | TTR (-0.5σ), Yule's K (-0.5σ), palabras ≥7 letras (-0.5σ), nosotros/1000 (-0.4σ) |
| Javier | "o sea"/1000 (+0.9σ), MTLD (+0.7σ), háрах (+0.7σ), nosotros/1000 (+0.3σ) | mayúsculas no iniciales (-1.0σ), comas/1000 (-0.9σ), oraciones ≤8 palabras (-0.5σ), vocativos/1000 (-0.4σ) |
| Nicolás | palabras/oración (+1.6σ), σ palabras/oración (+1.6σ), letras/palabra (+1.5σ), palabras ≥7 letras (+1.4σ) | Fernández-Huerta (-2.2σ), (yo−nosotros)/(yo+nosotros) (-1.1σ), muletillas/1000 (-0.9σ), ? /1000 (-0.7σ) |
| Pedro | atenuadores/1000 (+0.9σ), letras/palabra (+0.6σ), palabras ≥7 letras (+0.6σ), Yule's K (+0.5σ) | "o sea"/1000 (-0.5σ), palabras (-0.5σ), comas/1000 (-0.5σ), muletillas/1000 (-0.4σ) |
| Sebastián | palabras/oración (+1.0σ), nosotros/1000 (+0.9σ), TTR (+0.8σ), Yule's K (+0.7σ) | muletillas/1000 (-1.1σ), oraciones ≤8 palabras (-1.0σ), (yo−nosotros)/(yo+nosotros) (-0.8σ), Fernández-Huerta (-0.8σ) |

Son rasgos de **texto**, no de personalidad: describen cómo se escribe/habla en el corpus (transcripciones), no cómo es la persona.
