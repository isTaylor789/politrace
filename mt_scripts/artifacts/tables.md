# Tablas — Topic modeling y evolución temporal (módulo 6)

Generado por `mt_scripts/build_topics.py`.

## 0. Base de RI reutilizada

| documentos | vocabulario RI | diferencia máx. con w_global | reconstrucción exacta | vocabulario MT | términos en co-ocurrencia |
|---|---|---|---|---|---|
| 4531 | 30083 | 0.000 | sí | 8000 | 189 |

Términos del vocabulario de MT: df ≥ 10, df ≤ 50 % de los documentos, sin dígitos, sin la lista de puesta en escena (`config.BOILERPLATE_TERMS`: cargos, gentilicios de autorreferencia, nombres de los hablantes, dícticos), tope 8000 por masa df·idf. Ventana de co-ocurrencia: 10 tokens.

## 1. Barrido familia × k (coherencia)

| familia | k | c_v | UMass | diversidad | exclusividad | tiempo (s) |
|---|---|---|---|---|---|---|
| LSA (SVD truncada) | 6 | 0.140 | -1.303 | 0.600 | 0.322 | 1.900 |
| LSA (SVD truncada) | 8 | 0.141 | -1.435 | 0.575 | 0.266 | 2.100 |
| LSA (SVD truncada) | 10 | 0.128 | -1.596 | 0.540 | 0.225 | 1.800 |
| LSA (SVD truncada) | 12 | 0.147 | -1.634 | 0.500 | 0.194 | 2.100 |
| LSA (SVD truncada) | 15 | 0.137 | -1.694 | 0.473 | 0.172 | 1.900 |
| NMF | 6 | 0.198 | -0.962 | 0.917 | 0.708 | 3.000 |
| NMF | 8 | 0.331 | -1.018 | 0.925 | 0.688 | 3.200 |
| NMF | 10 | 0.443 | -1.058 | 0.940 | 0.682 | 4.600 |
| NMF | 12 | 0.444 | -1.053 | 0.925 | 0.653 | 4.700 |
| NMF | 15 | 0.492 | -1.149 | 0.920 | 0.634 | 17.800 |
| LDA | 6 | 0.142 | -0.828 | 0.883 | 0.532 | 19.400 |
| LDA | 8 | 0.158 | -0.814 | 0.800 | 0.426 | 17.300 |
| LDA | 10 | 0.271 | -0.811 | 0.860 | 0.423 | 19.600 |
| LDA | 12 | 0.250 | -0.837 | 0.817 | 0.400 | 19.700 |
| LDA | 15 | 0.284 | -0.930 | 0.760 | 0.384 | 23.600 |

Modelo elegido: **NMF con k = 10** (c_v = 0.443, UMass = -1.058, diversidad = 0.940). Regla 1-SE: c_v máxima 0.492 en k = 15, SE = 0.048, banda = 0.050, umbral = 0.442; se toma el menor k que supera el umbral.

## 2. Tópicos del modelo elegido

| tópico | nombre | términos | anclas | Andrés Manuel López Obrador | Javier Milei | Nicolás Maduro | Pedro Sánchez | Sebastián Piñera Echenique |
|---|---|---|---|---|---|---|---|---|
| 0 | seguridad y justicia | federal, guardia, investigacion, ciento, delitos, fiscalia, operacion, corrupcion, homicidios, marina | 2 | 0.227 | 0.037 | 0.100 | 0.026 | 0.031 |
| 1 | política exterior | europea, union, europa, ucrania, putin, consejo, guerra, euros, cumbre, logicamente | 7 | 0.017 | 0.038 | 0.204 | 0.398 | 0.037 |
| 2 | pandemia y crisis sanitaria | vacunas, dosis, pandemia, covid, pfizer, adultos, poblacion, contagios, sanitaria, virus | 3 | 0.062 | 0.001 | 0.069 | 0.031 | 0.072 |
| 3 | salud, educación y protección social | jovenes, becas, niños, escuelas, adultos, maestros, pension, corrupcion, dinero, creditos | 2 | 0.238 | 0.007 | 0.038 | 0.015 | 0.052 |
| 4 | seguridad y justicia | carabineros, compatriotas, niños, region, proteger, mujeres, recuperacion, calidad, esfuerzo, colaboracion | 2 | 0.040 | 0.114 | 0.242 | 0.133 | 0.590 |
| 5 | infraestructura y obras | tren, aeropuerto, maya, obra, tramo, sureste, cancun, arqueologicas, roo, quintana | 4 | 0.131 | 0.014 | 0.018 | 0.021 | 0.065 |
| 6 | economía, inflación y fiscal | inflacion, pbi, monetaria, libertad, crecimiento, fiscal, tasa, precio, deficit, banco | 10 | 0.063 | 0.761 | 0.186 | 0.068 | 0.072 |
| 7 | salud, educación y protección social | hospitales, medicos, imss, medicamentos, medicina, atencion, enfermeras, doctor, equipo, unidades | 3 | 0.105 | 0.004 | 0.037 | 0.012 | 0.037 |
| 8 | democracia, instituciones y electoral | cataluña, parlamentarios, autonomas, catalana, euros, popular, vivienda, independencia, ministro, grupo | 3 | 0.027 | 0.020 | 0.097 | 0.287 | 0.033 |
| 9 | Centavos · precio · litro | centavos, precio, litro, gas, margen, pemex, gasolina, franquicia, consumidores, ciento | 0 | 0.090 | 0.005 | 0.008 | 0.008 | 0.010 |

La columna *anclas* es el número de anclas del gazetteer que dispararon el nombre manual; cuando es 0 el nombre es automático (los tres términos de mayor peso).

## 3. Evolución temporal (vía A: un modelo, θ por periodo)

Participación media de cada tópico por hablante-año (cada fila de θ suma 1; *no* son conteos, así que el desbalance del corpus no domina la tabla).

| tópico | Andrés 2018 | Andrés 2019 | Andrés 2020 | Andrés 2021 | Andrés 2022 | Andrés 2023 | Andrés 2024 | Javier 2023 | Javier 2024 | Javier 2025 | Javier 2026 | Nicolás 2015 | Nicolás 2018 | Nicolás 2019 | Nicolás 2020 | Nicolás 2021 | Nicolás 2022 | Nicolás 2023 | Pedro 2018 | Pedro 2019 | Pedro 2020 | Pedro 2021 | Pedro 2022 | Pedro 2023 | Pedro 2024 | Pedro 2025 | Pedro 2026 | Sebastián 2018 | Sebastián 2019 | Sebastián 2020 | Sebastián 2021 | Sebastián 2022 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.298 | 0.202 | 0.187 | 0.195 | 0.272 | 0.274 | 0.267 | 0.013 | 0.035 | 0.040 | 0.039 | 0.262 | 0.163 | 0.118 | 0.078 | 0.095 | 0.101 | 0.049 | 0.038 | 0.032 | 0.012 | 0.019 | 0.019 | 0.018 | 0.034 | 0.027 | 0.033 | 0.032 | 0.034 | 0.023 | 0.034 | 0.032 |
| 1 | 0.018 | 0.020 | 0.014 | 0.012 | 0.026 | 0.015 | 0.014 | 0.002 | 0.028 | 0.034 | 0.083 | 0.231 | 0.108 | 0.161 | 0.167 | 0.256 | 0.159 | 0.829 | 0.312 | 0.266 | 0.277 | 0.337 | 0.543 | 0.575 | 0.375 | 0.412 | 0.427 | 0.056 | 0.046 | 0.017 | 0.018 | 0.020 |
| 2 | 0.002 | 0.005 | 0.067 | 0.228 | 0.053 | 0.020 | 0.007 | 0.002 | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 | 0.003 | 0.103 | 0.043 | 0.158 | 0.000 | 0.000 | 0.002 | 0.075 | 0.171 | 0.024 | 0.010 | 0.010 | 0.010 | 0.002 | 0.006 | 0.006 | 0.110 | 0.228 | 0.135 |
| 3 | 0.264 | 0.369 | 0.232 | 0.194 | 0.164 | 0.181 | 0.195 | 0.003 | 0.008 | 0.006 | 0.005 | 0.000 | 0.048 | 0.076 | 0.042 | 0.016 | 0.031 | 0.000 | 0.021 | 0.019 | 0.004 | 0.010 | 0.011 | 0.014 | 0.017 | 0.021 | 0.020 | 0.068 | 0.049 | 0.038 | 0.040 | 0.058 |
| 4 | 0.061 | 0.046 | 0.058 | 0.034 | 0.026 | 0.032 | 0.032 | 0.061 | 0.110 | 0.112 | 0.140 | 0.221 | 0.207 | 0.294 | 0.261 | 0.245 | 0.189 | 0.034 | 0.124 | 0.145 | 0.138 | 0.153 | 0.122 | 0.094 | 0.152 | 0.138 | 0.142 | 0.601 | 0.612 | 0.626 | 0.524 | 0.522 |
| 5 | 0.111 | 0.060 | 0.119 | 0.109 | 0.144 | 0.227 | 0.201 | 0.000 | 0.014 | 0.016 | 0.010 | 0.032 | 0.047 | 0.033 | 0.006 | 0.024 | 0.010 | 0.039 | 0.017 | 0.022 | 0.012 | 0.017 | 0.026 | 0.020 | 0.028 | 0.022 | 0.020 | 0.071 | 0.080 | 0.042 | 0.044 | 0.102 |
| 6 | 0.074 | 0.076 | 0.057 | 0.045 | 0.069 | 0.054 | 0.077 | 0.878 | 0.777 | 0.759 | 0.698 | 0.136 | 0.204 | 0.248 | 0.155 | 0.215 | 0.234 | 0.049 | 0.067 | 0.071 | 0.055 | 0.043 | 0.053 | 0.058 | 0.080 | 0.097 | 0.094 | 0.102 | 0.095 | 0.037 | 0.033 | 0.035 |
| 7 | 0.071 | 0.123 | 0.145 | 0.065 | 0.098 | 0.096 | 0.080 | 0.010 | 0.003 | 0.003 | 0.005 | 0.000 | 0.019 | 0.000 | 0.070 | 0.010 | 0.037 | 0.000 | 0.005 | 0.011 | 0.033 | 0.012 | 0.009 | 0.009 | 0.011 | 0.013 | 0.009 | 0.027 | 0.035 | 0.053 | 0.032 | 0.068 |
| 8 | 0.046 | 0.025 | 0.030 | 0.021 | 0.023 | 0.025 | 0.046 | 0.031 | 0.017 | 0.023 | 0.017 | 0.118 | 0.202 | 0.067 | 0.102 | 0.093 | 0.075 | 0.000 | 0.414 | 0.430 | 0.390 | 0.231 | 0.168 | 0.195 | 0.291 | 0.256 | 0.241 | 0.029 | 0.027 | 0.042 | 0.041 | 0.019 |
| 9 | 0.055 | 0.073 | 0.092 | 0.097 | 0.125 | 0.077 | 0.080 | 0.000 | 0.006 | 0.005 | 0.003 | 0.000 | 0.000 | 0.000 | 0.017 | 0.002 | 0.006 | 0.000 | 0.002 | 0.002 | 0.004 | 0.008 | 0.024 | 0.008 | 0.003 | 0.005 | 0.011 | 0.008 | 0.015 | 0.012 | 0.006 | 0.009 |

### 3.1 Ventana limpia 2023-2024

| tópico | Andrés 2023 | Andrés 2024 | Javier 2023 | Javier 2024 | Nicolás 2023 | Pedro 2023 | Pedro 2024 |
|---|---|---|---|---|---|---|---|
| 0 | 0.274 | 0.267 | 0.013 | 0.035 | 0.049 | 0.018 | 0.034 |
| 1 | 0.015 | 0.014 | 0.002 | 0.028 | 0.829 | 0.575 | 0.375 |
| 2 | 0.020 | 0.007 | 0.002 | 0.001 | 0.000 | 0.010 | 0.010 |
| 3 | 0.181 | 0.195 | 0.003 | 0.008 | 0.000 | 0.014 | 0.017 |
| 4 | 0.032 | 0.032 | 0.061 | 0.110 | 0.034 | 0.094 | 0.152 |
| 5 | 0.227 | 0.201 | 0.000 | 0.014 | 0.039 | 0.020 | 0.028 |
| 6 | 0.054 | 0.077 | 0.878 | 0.777 | 0.049 | 0.058 | 0.080 |
| 7 | 0.096 | 0.080 | 0.010 | 0.003 | 0.000 | 0.009 | 0.011 |
| 8 | 0.025 | 0.046 | 0.031 | 0.017 | 0.000 | 0.195 | 0.291 |
| 9 | 0.077 | 0.080 | 0.000 | 0.006 | 0.000 | 0.008 | 0.003 |

Única ventana donde 4 de los 5 líderes están activos (plan base, restricción 3).

## 4. Tópicos por hablante (modelo independiente, k = 8)

| hablante | discursos | términos | tópico principal | participación | términos del tópico | confianza |
|---|---|---|---|---|---|---|
| Andrés Manuel López Obrador | 2294 | 4000 | Becas · pension · creditos | 0.203 | becas, pension, creditos, banco, sucursales, productores, sembrando, mensuales | normal |
| Javier Milei | 187 | 4000 | economía, inflación y fiscal | 0.198 | deficit, monetaria, superavit, gasto, ajuste, reforma, eliminar, emision | normal |
| Nicolás Maduro | 30 | 2745 | Unilateral · coercitivas · bloqueo | 0.263 | unilateral, coercitivas, bloqueo, pandemia, criminal, ilegal, hegemonicos, agresion | baja |
| Pedro Sánchez | 1136 | 3303 | política exterior | 0.191 | relacion, cumbre, multilateral, cooperacion, global, climatico, bilateral, ambos | normal |
| Sebastián Piñera Echenique | 884 | 2406 | Energia · tecnologia · comercio | 0.217 | energia, tecnologia, comercio, desafios, desarrollo, materia, innovacion, cumbre | normal |

## 5. Dinámica por ventanas (vía B)

| hablante | año | discursos | k | tópico dominante | participación |
|---|---|---|---|---|---|
| Andrés Manuel López Obrador | 2019 | 552 | 8 | guardia, investigacion, policia, delitos, denuncia, fiscalia | 0.228 |
| Andrés Manuel López Obrador | 2020 | 439 | 8 | investigacion, agua, medicamentos, denuncia, fiscalia, consulta | 0.219 |
| Andrés Manuel López Obrador | 2021 | 365 | 8 | investigacion, fiscalia, denuncia, delitos, candidato, violencia | 0.177 |
| Andrés Manuel López Obrador | 2022 | 361 | 8 | electoral, video, mentiras, falso, ine, elecciones | 0.197 |
| Andrés Manuel López Obrador | 2023 | 308 | 8 | judicial, cuevas, mentiras, falso, ine, fox | 0.184 |
| Andrés Manuel López Obrador | 2024 | 248 | 8 | morelos, virreyes, felicidad, independencia, amor, becas | 0.226 |
| Javier Milei | 2024 | 87 | 8 | monetaria, deficit, ajuste, banco, central, dolares | 0.180 |
| Javier Milei | 2025 | 69 | 8 | superavit, cambiario, provincia, inversion, eliminar, emision | 0.228 |
| Pedro Sánchez | 2018 | 121 | 8 | iberoamericana, sociedades, nacion, bolivia, desafios, amigos | 0.220 |
| Pedro Sánchez | 2019 | 137 | 8 | turismo, sector, empleos, innovacion, turistico, mujeres | 0.257 |
| Pedro Sánchez | 2020 | 105 | 8 | digital, digitalizacion, transformacion, ecologica, cohesion, resiliencia | 0.213 |
| Pedro Sánchez | 2021 | 110 | 8 | euros, digital, transicion, transformacion, resiliencia, sector | 0.193 |
| Pedro Sánchez | 2022 | 172 | 8 | putin, ucraniano, otan, refugiados, invasion, solidaridad | 0.174 |
| Pedro Sánchez | 2023 | 129 | 8 | vivienda, usted, parlamentarios, ley, legislatura, autonomas | 0.187 |
| Pedro Sánchez | 2024 | 142 | 8 | euros, empresas, economia, sector, competitividad, india | 0.226 |
| Pedro Sánchez | 2025 | 167 | 8 | union, comercial, empresas, economia, inversion, sector | 0.180 |
| Pedro Sánchez | 2026 | 53 | 8 | euros, formacion, fondo, inversion, competitividad, vivienda | 0.206 |
| Sebastián Piñera Echenique | 2018 | 269 | 8 | region, magallanes, maravilloso, vivienda, valparaiso, alcalde | 0.192 |
| Sebastián Piñera Echenique | 2019 | 243 | 8 | comercio, china, cumbre, apec, calentamiento, materia | 0.174 |
| Sebastián Piñera Echenique | 2020 | 163 | 8 | region, plan, araucania, magallanes, recuperacion, empleos | 0.240 |
| Sebastián Piñera Echenique | 2021 | 159 | 8 | hospitales, deporte, salud, parque, emocion, vivienda | 0.164 |
| Sebastián Piñera Echenique | 2022 | 50 | 8 | metros, modernizacion, empleos, tren, aeropuerto, tunel | 0.158 |

| hablante | transición | similitud media | emparejados | nuevos | perdidos |
|---|---|---|---|---|---|
| Andrés Manuel López Obrador | 2019→2020 | 0.505 | 4 | 4 | 4 |
| Andrés Manuel López Obrador | 2020→2021 | 0.454 | 7 | 1 | 1 |
| Andrés Manuel López Obrador | 2021→2022 | 0.406 | 7 | 1 | 1 |
| Andrés Manuel López Obrador | 2022→2023 | 0.480 | 6 | 2 | 2 |
| Andrés Manuel López Obrador | 2023→2024 | 0.456 | 5 | 3 | 3 |
| Javier Milei | 2024→2025 | 0.340 | 6 | 2 | 2 |
| Pedro Sánchez | 2018→2019 | 0.387 | 3 | 5 | 5 |
| Pedro Sánchez | 2019→2020 | 0.287 | 3 | 5 | 5 |
| Pedro Sánchez | 2020→2021 | 0.440 | 1 | 7 | 7 |
| Pedro Sánchez | 2021→2022 | 0.345 | 4 | 4 | 4 |
| Pedro Sánchez | 2022→2023 | 0.410 | 4 | 4 | 4 |
| Pedro Sánchez | 2023→2024 | 0.480 | 3 | 5 | 5 |
| Pedro Sánchez | 2024→2025 | 0.436 | 5 | 3 | 3 |
| Pedro Sánchez | 2025→2026 | 0.345 | 4 | 4 | 4 |
| Sebastián Piñera Echenique | 2018→2019 | 0.388 | 5 | 3 | 3 |
| Sebastián Piñera Echenique | 2019→2020 | 0.400 | 3 | 5 | 5 |
| Sebastián Piñera Echenique | 2020→2021 | 0.383 | 6 | 2 | 2 |
| Sebastián Piñera Echenique | 2021→2022 | 0.320 | 2 | 6 | 6 |

Deriva = similitud de coseno entre el vector de términos del mismo tópico en dos años consecutivos, tras alineación húngara. `sim ≥ 0.25` cuenta como tópico emparejado; lo demás son tópicos nuevos/perdidos.

## 6. Evaluación

### 6.1 Anclas del gold anual del módulo 5

| caso | discursos | recall | anclas cubiertas | anclas | tópicos dominantes |
|---|---|---|---|---|---|
| Pedro Sánchez 2023 | 129 | 0.500 | 4 | 8 | T1, T8, T4 |
| Pedro Sánchez 2024 | 142 | 0.375 | 3 | 8 | T1, T8, T4 |
| Andrés Manuel López Obrador 2023 | 308 | 0.500 | 4 | 8 | T0, T5, T3 |
| Javier Milei 2024 | 87 | 0.500 | 4 | 8 | T6, T4, T0 |

Un ancla se cubre si sus stems están entre los 30 términos más pesados de los 3 tópicos dominantes del año.

### 6.2 Pureza contra el gold manual de áreas temáticas

Pureza = 0.553 (pureza@2 = 0.660, n = 47).

| área temática (manual) | discursos | tópico modal | nombre del tópico | coincidencia |
|---|---|---|---|---|
| economía, inflación y fiscal | 5 | T3 | salud, educación y protección social | 0.400 |
| política exterior | 5 | T1 | política exterior | 0.600 |
| infraestructura y obras | 5 | T5 | infraestructura y obras | 0.600 |
| democracia, instituciones y electoral | 5 | T0 | seguridad y justicia | 0.400 |
| pandemia y crisis sanitaria | 5 | T4 | seguridad y justicia | 0.400 |
| energía, clima y medio ambiente | 4 | T5 | infraestructura y obras | 0.250 |
| empleo y trabajo | 3 | T3 | salud, educación y protección social | 0.333 |
| salud, educación y protección social | 3 | T3 | salud, educación y protección social | 0.667 |
| seguridad y justicia | 3 | T0 | seguridad y justicia | 0.667 |
| vivienda y territorio | 3 | T8 | democracia, instituciones y electoral | 1.000 |
| migración y frontera | 3 | T4 | seguridad y justicia | 0.667 |
| ideología, libertad y valores | 3 | T6 | economía, inflación y fiscal | 1.000 |

### 6.3 Sensibilidad a k (misma familia)

| k | c_v | diversidad | pureza | pureza@2 | anclas | elegido |
|---|---|---|---|---|---|---|
| 6 | 0.198 | 0.917 | 0.638 | 0.766 | 0.344 |  |
| 8 | 0.331 | 0.925 | 0.511 | 0.681 | 0.406 |  |
| 10 | 0.443 | 0.940 | 0.553 | 0.660 | 0.469 | ← |
| 12 | 0.444 | 0.925 | 0.532 | 0.638 | 0.469 |  |
| 15 | 0.492 | 0.920 | 0.468 | 0.553 | 0.531 |  |

### 6.4 Estabilidad (remuestreo de documentos)

| corridas | submuestra | Jaccard medio | Jaccard σ | c_v σ |
|---|---|---|---|---|
| 5 | 0.800 | 0.858 | 0.175 | 0.010 |

## 7. Pares con mayor NPMI en el corpus

| término A | término B | NPMI | pares |
|---|---|---|---|
| quintan | roo | 0.697 | 3726 |
| jueg | panamerican | 0.565 | 135 |
| europe | union | 0.555 | 5095 |
| jueg | olimp | 0.546 | 127 |
| medall | olimp | 0.543 | 57 |
| may | tren | 0.539 | 5948 |
| adult | pension | 0.517 | 2918 |
| astrazenec | pfiz | 0.514 | 127 |
| medall | panamerican | 0.507 | 42 |
| catalan | cataluñ | 0.501 | 293 |
| grup | parlamentari | 0.495 | 1384 |
| centav | marg | 0.494 | 3448 |
| beisbol | futbol | 0.489 | 83 |
| digital | transicion | 0.483 | 523 |
| deficit | pbi | 0.481 | 189 |
| muj | vide | 0.475 | 1495 |
| putin | ucrani | 0.474 | 520 |
| dosis | vacun | 0.468 | 2306 |
| arqueolog | zon | 0.468 | 1528 |
| beisbol | deport | 0.467 | 233 |

NPMI sobre pares de tokens a distancia < 10 (ver `mt_scripts/coherence.py`).
