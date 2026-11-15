# Discursos presidenciales en español (dataset unificado)

## Contenido

| country_iso | speaker_name | Registros | Desde | Hasta |
|-------------|--------------|-----------|-------|-------|
| ARG | Javier Milei | 187 | 2023-12-10 | 2026-04-29 |
| CHL | Sebastián Piñera Echenique | 884 | 2018-03-11 | 2022-03-05 |
| ESP | Pedro Sánchez | 1,136 | 2018-06-11 | 2026-04-29 |
| MEX | Andrés Manuel López Obrador | 2,294 | 2018-12-02 | 2024-09-30 |
| VEN | Nicolás Maduro | 30 | 2015-07-06 | 2023-01-24 |


## Columnas

| Columna | Descripción |
|---------|-------------|
| `speech_id` | Identificador único del discurso. Se conserva el de los parquets originales. Para AMLO se generó como `MEX-AMLO-AAAAMMDD-<hash12 de la URL>-es`. |
| `speaker_name` | Nombre del presidente. |
| `speech_date` | Fecha en formato ISO 8601 (`YYYY-MM-DD`). |
| `speech_title` | Título o encabezado del discurso. |
| `speech_text` | Texto completo del discurso. |
| `country_iso` | Código ISO 3166-1 alpha-3: `ARG`, `CHL`, `ESP`, `MEX`, `VEN`. |

## Reproducir

```bash
source ../env/bin/activate
python scripts/build_unified_dataset.py
```

Para generar además una copia CSV (archivo grande):

```bash
python scripts/build_unified_dataset.py --csv
```

## Cargar

```python
import pandas as pd

df = pd.read_parquet("data/unified/presidentes_latam_es.parquet")
print(df.shape)
```
