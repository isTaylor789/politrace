### Menciones (gold manual, emparejamiento estricto)

| Tipo | TP | FP | FN | P | R | F1 | IC95 P |
|---|---|---|---|---|---|---|---|
| ORG | 15 | 4 | 6 | 0.789 | 0.714 | 0.750 | [0.57, 0.92] |
| PER | 12 | 2 | 3 | 0.857 | 0.800 | 0.828 | [0.60, 0.96] |
| LOC | 31 | 0 | 7 | 1.000 | 0.816 | 0.899 | [0.89, 1.00] |
| **micro** | 58 | 6 | 16 | 0.906 | 0.784 | 0.841 | [0.81, 0.96] |

### Ablación: coreferencia y criterio de emparejamiento

| Configuración | P | R | F1 |
|---|---|---|---|
| gazetteer + patrones + coref (estricto) | 0.906 | 0.784 | 0.841 |
| sin coreferencia (estricto) | 0.905 | 0.770 | 0.832 |
| con coreferencia (laxo) | 0.906 | 0.784 | 0.841 |

### Relaciones (muestra etiquetada)

| Tipo | n | aciertos | precisión |
|---|---|---|---|
| coocurre | 30 | 21 | 0.700 |
| verbales (todos los tipos) | 21 | 6 | 0.286 |
| cargo (inferido/explícito) | 2 | 2 | 1.000 |
| **total** | 53 | 29 | 0.547 |
