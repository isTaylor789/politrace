### Ablaciones (gold manual)

| Configuración | TP | FP | FN | P | R | F1 | IC95 P | IC95 R |
|---|---|---|---|---|---|---|---|---|
| A · autómata | 30 | 15 | 8 | 0.667 | 0.789 | 0.723 | [0.52, 0.79] | [0.64, 0.89] |
| B · + negación | 29 | 13 | 9 | 0.690 | 0.763 | 0.725 | [0.54, 0.81] | [0.61, 0.87] |
| C · + slots regex | 29 | 13 | 9 | 0.690 | 0.763 | 0.725 | [0.54, 0.81] | [0.61, 0.87] |
| D · + CFG/CYK (completo) | 25 | 4 | 13 | 0.862 | 0.658 | 0.746 | [0.69, 0.94] | [0.50, 0.79] |

### Niveles de evidencia (sistema completo)

| Nivel | Umbral | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| cualquier evidencia | 0.00 | 25 | 4 | 13 | 0.862 | 0.658 | 0.746 |
| acción de contenido | 0.65 | 18 | 3 | 20 | 0.857 | 0.474 | 0.610 |
| + CYK | 0.71 | 17 | 2 | 21 | 0.895 | 0.447 | 0.596 |
| + lugar explícito | 0.81 | 3 | 1 | 35 | 0.750 | 0.079 | 0.143 |
| + fecha / monto / cantidad | 0.86 | 1 | 0 | 37 | 1.000 | 0.026 | 0.051 |

### Por estrato de muestreo (sistema completo)

| Estrato | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| aleatorio | 4 | 0 | 6 | 1.000 | 0.400 | 0.571 |
| aceptado | 21 | 4 | 5 | 0.840 | 0.808 | 0.824 |
| rechazado | 0 | 0 | 2 | 0.000 | 0.000 | 0.000 |

### Por hablante (D · + CFG/CYK (completo))

| Hablante | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| Andrés Manuel López Obrador | 8 | 0 | 2 | 1.000 | 0.800 | 0.889 |
| Javier Milei | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Nicolás Maduro | 2 | 4 | 4 | 0.333 | 0.333 | 0.333 |
| Pedro Sánchez | 8 | 0 | 3 | 1.000 | 0.727 | 0.842 |
| Sebastián Piñera Echenique | 4 | 0 | 4 | 1.000 | 0.500 | 0.667 |

Prevalencia en el estrato aleatorio: 0.080 (10/125, IC95 [0.044, 0.141]).

Kappa entre configuraciones: A · autómata vs B · + negación = 0.956, A · autómata vs D · + CFG/CYK (completo) = 0.739

### Volumen del corpus

| Métrica | Valor |
|---|---|
| documentos | 4531 |
| eventos | 11052 |
| segundos | 145.5 |
| autómata | {'estados': 727, 'patrones': 619, 'aristas': 726, 'alfabeto': 656, 'profundidad_max': 11} |
