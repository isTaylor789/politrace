# Tablas del módulo RAT

Generado por `rat_scripts/build_summaries.py` · N = 4531 discursos · vocabulario = 30083 términos.

## 1. Parámetros de calibración

| parámetro | valor |
|---|---|
| MIN_SENT_WORDS | 7 |
| MAX_SENT_WORDS | 75 |
| SUMMARY_RATIO | 0.150 |
| MIN_SENTENCES | 3 |
| MAX_SENTENCES | 12 |
| WORD_BUDGET | 180 |
| MMR_LAMBDA | 0.700 |
| DUP_JACCARD | 0.800 |
| PAGERANK_DAMPING | 0.850 |
| LEXRANK_THRESHOLD | 0.100 |
| LSA_K | 100 |
| WEIGHTS | {'centroid': 0.28, 'textrank': 0.24, 'discourse': 0.26, 'lsa': 0.1, 'luhn': 0.12} |
| DISCOURSE_WEIGHTS | {'position': 0.5, 'cue': 0.35, 'title': 0.15} |
| idf | global del índice RI (min_df=2) |


## 2. Referencias escritas a mano (gold)

| doc_id | hablante | fecha | palabras ref. | referencia |
|---:|---|---|---:|---|
| 1065 | Andrés Manuel López Obrador | 2021-02-26 | 77 | En la inauguración del Hospital General de Querétaro, López Obrador afirmó que se garantiza el derecho del pueblo a la salud, con atención médica de calidad y medicamentos gratuitos. Señaló que el hospital atenderá a la población sin seguridad social y funcionará como hospital escuela de especialidades, para reducir el déficit de médicos. Informó que con este suman 120 centros de salud y hospitales terminados en dos años, con inversión compartida entre la federación y el estado. |
| 2381 | Javier Milei | 2024-12-17 | 64 | En la ceremonia de egreso de oficiales de las tres fuerzas, Milei exigió la liberación inmediata de Nahuel Gallo, detenido en Venezuela. Sostuvo que su gobierno está reconciliando a las Fuerzas Armadas con la política y que las jerarquizará, aunque les exigirá más compromiso. Recordó la batalla de Caseros y pidió a los nuevos oficiales servicio a la patria y fidelidad a la Constitución. |
| 2493 | Nicolás Maduro | 2020-09-18 | 67 | Ante la ONU, Maduro afirmó que Venezuela mantiene su compromiso con la Agenda 2030 pese a las sanciones y bloqueos de Estados Unidos. Destacó las Misiones sociales: los CLAP cubren más de 6 millones de familias, la Misión Vivienda entregó más de 3 millones de viviendas y casi 5 millones de personas reciben pensión. Aseguró que el sistema público atendió al 99% de los contagiados de COVID-19. |
| 2877 | Pedro Sánchez | 2021-01-26 | 63 | Sánchez anunció un reajuste mínimo del Gobierno tras la dimisión de Salvador Illa para ser candidato en las elecciones catalanas. Carolina Darias será la nueva ministra de Sanidad y Miquel Iceta el de Política Territorial y Función Pública. Agradeció el trabajo de Illa y pidió a los nuevos ministros mantener la prioridad en la emergencia sanitaria y en la recuperación económica y social. |
| 4432 | Sebastián Piñera Echenique | 2021-10-07 | 62 | Piñera presentó un balance de la vacunación: 14,2 millones de personas tienen dos dosis y 3,6 millones el refuerzo. Un estudio del Ministerio de Salud en 11 millones de personas mostró alta efectividad del refuerzo para evitar hospitalizaciones: 88% con Sinovac, 87% con Pfizer y 96% con AstraZeneca. Llamó a vacunarse y a mantener los protocolos sanitarios frente a la variante Delta. |


## 3. ROUGE por método (media de los 5 discursos)

| método | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | compresión | cobertura | redundancia | pal/oración |
|---|---|---|---|---|---|---|---|
| lead | 0.189 | 0.049 | 0.129 | 0.151 | 0.565 | 0.042 | 27.823 |
| tfidf | 0.171 | 0.049 | 0.130 | 0.090 | 0.478 | 0.031 | 17.850 |
| centroid | 0.236 | 0.086 | 0.177 | 0.194 | 0.713 | 0.229 | 35.943 |
| luhn | 0.232 | 0.059 | 0.148 | 0.202 | 0.723 | 0.119 | 40.957 |
| textrank | 0.236 | 0.086 | 0.177 | 0.194 | 0.713 | 0.229 | 35.943 |
| lexrank | 0.266 | 0.096 | 0.188 | 0.176 | 0.745 | 0.153 | 33.863 |
| lsa | 0.176 | 0.041 | 0.119 | 0.130 | 0.556 | 0.018 | 24.753 |
| hybrid | 0.285 | 0.110 | 0.202 | 0.189 | 0.756 | 0.156 | 35.803 |
| random | 0.181 | 0.042 | 0.118 | 0.000 | 0.000 | 0.000 | 0.000 |


## 4. Híbrido vs línea base, por discurso

| doc_id | hablante | lead R1 | híbrido R1 | híbrido R2 | híbrido RL | cobertura |
|---|---|---|---|---|---|---|
| 1065 | Andrés Manuel López Obrador | 0.157 | 0.389 | 0.189 | 0.296 | 0.795 |
| 2381 | Javier Milei | 0.222 | 0.226 | 0.077 | 0.132 | 0.709 |
| 2493 | Nicolás Maduro | 0.063 | 0.194 | 0.045 | 0.134 | 0.693 |
| 2877 | Pedro Sánchez | 0.235 | 0.318 | 0.154 | 0.258 | 0.826 |
| 4432 | Sebastián Piñera Echenique | 0.267 | 0.298 | 0.087 | 0.191 | 0.757 |


## 5. Ablación del híbrido

| variante | ROUGE-1 | ROUGE-2 | ROUGE-L | cobertura |
|---|---|---|---|---|
| completo | 0.285 | 0.110 | 0.202 | 0.756 |
| sin centroid | 0.296 | 0.094 | 0.184 | 0.743 |
| sin textrank | 0.296 | 0.094 | 0.184 | 0.743 |
| sin discourse | 0.247 | 0.093 | 0.175 | 0.764 |
| sin lsa | 0.277 | 0.110 | 0.202 | 0.749 |
| sin luhn | 0.270 | 0.108 | 0.205 | 0.745 |


## 6. Resúmenes de ejemplo (método híbrido)

### doc_id 1065 — Andrés Manuel López Obrador · 2021-02-26

*26.02.21 Versión estenográfica. Inauguración del Hospital General de Querétaro*

Unidades: 27 · seleccionadas: 5 · compresión: 18.5% · cobertura: 0.79 · redundancia: 0.17

> Me da mucho gusto estar de nuevo en esta histórica ciudad de Querétaro y me da más gusto el acompañar al ciudadano gobernador de Querétaro, Francisco Domínguez, en la inauguración de este hospital general. El anterior hospital se construyó, de acuerdo a lo que nos ha informado el gobernador, en 1963. Por eso es muy importante el que se esté inaugurando el día de hoy este hospital, que garantiza lo que establece el artículo 4º de la Constitución: el derecho del pueblo a la salud, atención médica de calidad y medicamentos gratuitos. Tenemos que trabajar juntos por la educación, por la salud, por el bienestar de nuestro pueblo y eso es lo que se está demostrando en este acto aquí en Querétaro. Por eso es muy importante decir que este es un hospital escuela, que es un hospital de especialidades, que es un hospital que va a poder atender a pacientes y que va a resolver la demanda de atención médica en Querétaro.

### doc_id 2381 — Javier Milei · 2024-12-17

*Discurso del Presidente Javier Milei en la entrega de sables y medallas en el Colegio Militar*

Unidades: 35 · seleccionadas: 6 · compresión: 18.3% · cobertura: 0.71 · redundancia: 0.06

> Voy a dirigirme ahora particularmente a los graduados. Han finalizado la formación básica para el servicio a la patria que ustedes eligieron, y por eso tienen mis más sinceras felicitaciones. Esto tiene un significado particular para ustedes, porque esta nueva camada de egresados será testigo de algo inédito en las últimas décadas, incluso para sus camaradas de mayor experiencia y rango: contarán con un gobierno que los acompañará y hará todo lo que esté a su alcance para jerarquizar las tres fuerzas, honrando el legado que nos dejaron Roca y San Martín. Por eso, queremos decirles, estimados oficiales que hoy egresan: la Nación cuenta con ustedes. Para finalizar, quiero hacer una especial mención a los hombres y mujeres de nuestras Fuerzas Armadas que contribuyeron a la formación de nuestros nuevos oficiales. Que Dios los bendiga a cada uno de ustedes y a todos los argentinos, y que las fuerzas del cielo nos acompañen.

### doc_id 2493 — Nicolás Maduro · 2020-09-18

*Discurso del presidente Nicolás Maduro en el Evento de Alto Nivel del Secretario General de la ONU: “El Momento de los ODS 2020”.*

Unidades: 14 · seleccionadas: 3 · compresión: 21.5% · cobertura: 0.69 · redundancia: 0.18

> Me encuentro aquí en Caracas, Venezuela, de pie, como Presidente Constitucional de la República Bolivariana de Venezuela, junto a nuestro pueblo, un pueblo heroico que resiste y vence las más diversas agresiones del imperio más poderoso que haya conocido la humanidad. La Revolución Bolivariana ha avanzado en el transcurso de dos décadas en la consolidación de un sistema de protección y seguridad social a través de las llamadas Misiones y Grande Misiones sociales, parte esencial de la estructura del ingreso familiar para la vida y protección, que a pesar de los desafíos que impone actualmente la pandemia del COVID-19 y de los bloqueos económicos y financieros se mantiene y se consolida. Misiones que como programa social acompañan al pueblo en su alimentación como la misión de los Comités Locales de Abastecimiento y Producción, los CLAP; la Misión Hogares de la Patria, que protege a la familia venezolana, a la madre trabajadora; la Misión Vivienda Venezuela; las misiones educativas; las misiones de salud; todas ellas que significan importantes transferencias e inversiones económicas y sociales para la calidad de vida del pueblo y como forma de atención directa.

### doc_id 2877 — Pedro Sánchez · 2021-01-26

*Intervención del presidente del Gobierno para comunicar nuevos nombramientos*

Unidades: 37 · seleccionadas: 6 · compresión: 22.0% · cobertura: 0.83 · redundancia: 0.09

> Comparezco para informarles de los nuevos nombramientos que se van a producir en el seno del Gobierno de España. Como todos ustedes saben, la dimisión del ministro Salvador Illa para concurrir como candidato a las elecciones catalanas del próximo 14 de febrero obliga a hacer modificaciones en el Gobierno de España. Fue nombrado ministro para poner en marcha la ampliación de servicios para una sanidad pública de calidad y para impulsar otras muchas medidas que figuraban en el programa de Gobierno. Y en segundo lugar, que Miquel Iceta Llorens será el nuevo ministro de Política Territorial y Función Pública. Los nuevos ministros, Carolina Darias y Miquel Iceta, conocen muy bien los ámbitos donde ahora desempeñarán sus tareas. Carolina Darias ha compartido, como saben, con Salvador Illa el día a día de la gestión de la pandemia, en primera línea del equipo frente al virus, presente tanto en el Consejo Interterritorial del Sistema Nacional de Salud como también en las Comisiones internas que desde el gobierno de España hemos venido celebrando durante estos largos meses de pandemia.

### doc_id 4432 — Sebastián Piñera Echenique · 2021-10-07

*Presidente Piñera encabeza entrega de resultados de estudio sobre efectividad de dosis de refuerzo*

Unidades: 24 · seleccionadas: 4 · compresión: 14.4% · cobertura: 0.76 · redundancia: 0.29

> Hoy queremos compartir con usted un balance del proceso de vacunación y un análisis de la efectividad de la vacuna de refuerzo. Al día de ayer, juntos hemos logrado vacunar con dos dosis a 14,2 millones de personas, es decir, 3 de cada 4 compatriotas. Chile fue pionero al iniciar en agosto un proceso masivo de vacunación de refuerzo que ya nos ha permitido proteger con este refuerzo a más 3,6 millones de personas. También fuimos pioneros cuando en septiembre iniciamos la vacunación de niños mayores de 6 años en sus establecimientos educacionales y, en pocas semanas, hemos logrado vacunar con la primera dosis a 4 de cada 10 niños entre 6 y 11 años.


## 7. Resumen multi-documento (Pedro Sánchez, 2023)

Discursos usados (5): 3225, 3157, 3278, 3271, 3193. Criterio: los más cercanos al centroide del conjunto.

Seleccionadas: 5 de 15 oraciones fusionadas · compresión sobre el total: 0.75% · cobertura: 0.80

> Creo que podamos dar ese impulso transformador que necesita Europa antes de que lleguemos a las elecciones europeas del año 2024 y vamos a poner todo nuestro empeño porque en efecto sigamos haciendo avanzar al conjunto de la Unión Europea. He dicho antes que vamos a celebrar en la Presidencia española de la Unión Europea una cumbre con esa Vecindad Sur a nivel europeo para implementar y desarrollar esa agenda positiva que debemos tener con nuestros vecinos del sur. Hoy mismo acabamos de conocer que se ha logrado un acuerdo importante entre el Parlamento Europeo, la Comisión Europea y el Consejo Europeo para lograr ese acuerdo de migración y asilo tan importante para el conjunto de la Unión Europea. El primero de ellos para demostrar que la Unión Europea bajo esta Presidencia ha avanzado, a pesar de lo complicado que resulta muchas veces poner de acuerdo a 27 Estados miembros; la Unión creo que a lo largo de estos seis meses ha dado pasos significativos en distintas materias a las cuales me voy a referir posteriormente. Y, además, reforzar eso que va a ser objeto de la presidencia española de la Unión Europea, como es la autonomía estratégica que, como saben, vamos a plantear como debate central en la cumbre que tengamos en Granada en el mes de octubre de este año.

Tópicos comunes (stem, nº de discursos, frecuencia total):

| término | discursos | frecuencia |
|---|---|---|
| europe | 5 | 218 |
| españ | 5 | 158 |
| cre | 5 | 154 |
| pais | 5 | 148 |
| president | 5 | 146 |
| europ | 5 | 132 |
| union | 5 | 122 |
| hac | 5 | 114 |
| import | 5 | 92 |
| deb | 5 | 87 |
| polit | 5 | 83 |
| español | 5 | 78 |

