# Tablas del resumen multi-documento por año (módulo 5)

Generado por `rat_scripts/build_multidoc.py`.

## 0. Años con material suficiente (≥ 5 discursos)
| hablante | año | discursos |
|---|---|---|
| Andrés Manuel López Obrador | 2018 | 21 |
| Andrés Manuel López Obrador | 2019 | 552 |
| Andrés Manuel López Obrador | 2020 | 439 |
| Andrés Manuel López Obrador | 2021 | 365 |
| Andrés Manuel López Obrador | 2022 | 361 |
| Andrés Manuel López Obrador | 2023 | 308 |
| Andrés Manuel López Obrador | 2024 | 248 |
| Javier Milei | 2024 | 87 |
| Javier Milei | 2025 | 69 |
| Javier Milei | 2026 | 27 |
| Nicolás Maduro | 2020 | 13 |
| Nicolás Maduro | 2021 | 6 |
| Pedro Sánchez | 2018 | 121 |
| Pedro Sánchez | 2019 | 137 |
| Pedro Sánchez | 2020 | 105 |
| Pedro Sánchez | 2021 | 110 |
| Pedro Sánchez | 2022 | 172 |
| Pedro Sánchez | 2023 | 129 |
| Pedro Sánchez | 2024 | 142 |
| Pedro Sánchez | 2025 | 167 |
| Pedro Sánchez | 2026 | 53 |
| Sebastián Piñera Echenique | 2018 | 269 |
| Sebastián Piñera Echenique | 2019 | 243 |
| Sebastián Piñera Echenique | 2020 | 163 |
| Sebastián Piñera Echenique | 2021 | 159 |
| Sebastián Piñera Echenique | 2022 | 50 |

## 1. Casos de referencia

| hablante | año | tópicos clave |
|---|---|---|
| Pedro Sánchez | 2023 | presidencia de la UE, Ucrania / Zelenski, Gaza / Palestina, Marruecos, reindustrialización, vivienda, CGPJ / poder judicial, Cataluña / investidura |
| Pedro Sánchez | 2024 | reconocimiento de Palestina, Gaza / Israel, DANA / Comunidad Valenciana, Ucrania, India / China, financiación autonómica / Cataluña, empleo / economía, vivienda |
| Andrés Manuel López Obrador | 2023 | Tren Maya, IMSS-Bienestar / federalización de la salud, programas sociales / Sembrando Vida, Acapulco / huracán Otis, Istmo / corredor interoceánico, seguridad / homicidios, Poder Judicial / fideicomisos, litio |
| Javier Milei | 2024 | ajuste fiscal, inflación / política monetaria, desregulación / DNU, recorte del gasto, Pacto de Mayo, Israel / Holocausto, Mercosur / aranceles, superávit / déficit |


## 2. Variantes del pipeline (media de los casos)

| variante | ROUGE-1 | ROUGE-2 | ROUGE-L | tópicos | sim. centroide | cobertura temas | compresión | redundancia | palabras |
|---|---|---|---|---|---|---|---|---|---|
| base (global) | 0.104 | 0.031 | 0.058 | 0.562 | 0.204 | 0.959 | 0.001 | 0.067 | 389.500 |
| idf blend | 0.105 | 0.030 | 0.060 | 0.562 | 0.205 | 0.959 | 0.001 | 0.056 | 392.250 |
| idf local | 0.100 | 0.030 | 0.059 | 0.562 | 0.208 | 0.959 | 0.001 | 0.049 | 391.500 |
| sin agrupamiento | 0.095 | 0.021 | 0.073 | 0.344 | 0.147 | 1.000 | 0.001 | 0.144 | 344.750 |
| sin cuotas | 0.094 | 0.023 | 0.061 | 0.406 | 0.230 | 0.647 | 0.001 | 0.089 | 393.250 |
| cuotas min-1 | 0.113 | 0.033 | 0.065 | 0.625 | 0.203 | 0.991 | 0.001 | 0.063 | 389.500 |
| aleatorio | 0.081 | 0.015 | 0.046 | 0.510 | 0.236 | 0.908 | 0.001 | 0.038 | 395.750 |


## 3. Detalle por caso (variante base)

| caso | grupos | docs | oraciones | palabras | ROUGE-1 | ROUGE-L | tópicos | temas |
|---|---|---|---|---|---|---|---|---|
| Pedro Sánchez 2023 | 6 | 9 | 11 | 381 | 0.106 | 0.077 | 0.500 | 0.977 |
| Pedro Sánchez 2024 | 6 | 9 | 11 | 377 | 0.087 | 0.038 | 0.500 | 0.972 |
| Andrés Manuel López Obrador 2023 | 8 | 10 | 12 | 400 | 0.115 | 0.062 | 0.625 | 0.922 |
| Javier Milei 2024 | 5 | 7 | 10 | 400 | 0.108 | 0.054 | 0.625 | 0.966 |


## 4. Tópicos clave cubiertos (variante base)

| caso | tópico | cubierto |
|---|---|---|
| Pedro Sánchez 2023 | presidencia de la UE | sí |
| Pedro Sánchez 2023 | Ucrania / Zelenski | sí |
| Pedro Sánchez 2023 | Gaza / Palestina | sí |
| Pedro Sánchez 2023 | Marruecos | no |
| Pedro Sánchez 2023 | reindustrialización | no |
| Pedro Sánchez 2023 | vivienda | no |
| Pedro Sánchez 2023 | CGPJ / poder judicial | sí |
| Pedro Sánchez 2023 | Cataluña / investidura | no |
| Pedro Sánchez 2024 | reconocimiento de Palestina | sí |
| Pedro Sánchez 2024 | Gaza / Israel | no |
| Pedro Sánchez 2024 | DANA / Comunidad Valenciana | sí |
| Pedro Sánchez 2024 | Ucrania | no |
| Pedro Sánchez 2024 | India / China | sí |
| Pedro Sánchez 2024 | financiación autonómica / Cataluña | sí |
| Pedro Sánchez 2024 | empleo / economía | no |
| Pedro Sánchez 2024 | vivienda | no |
| Andrés Manuel López Obrador 2023 | Tren Maya | sí |
| Andrés Manuel López Obrador 2023 | IMSS-Bienestar / federalización de la salud | sí |
| Andrés Manuel López Obrador 2023 | programas sociales / Sembrando Vida | sí |
| Andrés Manuel López Obrador 2023 | Acapulco / huracán Otis | no |
| Andrés Manuel López Obrador 2023 | Istmo / corredor interoceánico | no |
| Andrés Manuel López Obrador 2023 | seguridad / homicidios | sí |
| Andrés Manuel López Obrador 2023 | Poder Judicial / fideicomisos | sí |
| Andrés Manuel López Obrador 2023 | litio | no |
| Javier Milei 2024 | ajuste fiscal | sí |
| Javier Milei 2024 | inflación / política monetaria | sí |
| Javier Milei 2024 | desregulación / DNU | no |
| Javier Milei 2024 | recorte del gasto | sí |
| Javier Milei 2024 | Pacto de Mayo | sí |
| Javier Milei 2024 | Israel / Holocausto | sí |
| Javier Milei 2024 | Mercosur / aranceles | no |
| Javier Milei 2024 | superávit / déficit | no |


## 5. Resúmenes (variante base)

### Pedro Sánchez · 2023

Grupos: 6 · oraciones: 11 · palabras: 381 · compresión: 0.120% · sim. centroide: 0.25 · temas cubiertos: 98%

> Yo creo que en estos últimos años hemos demostrado que España no tiene que elegir entre crear empleo o mejorar las condiciones de los trabajadores y trabajadoras del mundo. Por tanto, creo que es imperativo para todas las fuerzas políticas el renovar el Consejo General del Poder Judicial en el año 2024. Y en ese sentido, si el Partido Popular considera que es necesario que interceda, que medie la Comisión Europea en esta cuestión, nosotros encantados. Creo que podamos dar ese impulso transformador que necesita Europa antes de que lleguemos a las elecciones europeas del año 2024 y vamos a poner todo nuestro empeño porque en efecto sigamos haciendo avanzar al conjunto de la Unión Europea. Pero en mi opinión, lo más importante, y a lo que tenemos que dar respuesta en los próximos años es qué tipo de Unión Europea queremos. Por eso, lo primero que le he transmitido a la presidenta de la Comisión y al Colegio de Comisarios es que el Gobierno y el pueblo de España van a cumplir con su deber, con nuestro deber. Y en ese contexto, España, junto con otros países de la Unión Europea, si tenemos que hacer algo, es respaldar la continuidad mientras las condiciones lógicamente de seguridad lo hagan posible. He dicho antes que vamos a celebrar en la Presidencia española de la Unión Europea una cumbre con esa Vecindad Sur a nivel europeo para implementar y desarrollar esa agenda positiva que debemos tener con nuestros vecinos del sur. España, les puedo garantizar señorías que no va a ser mero testigo de esos procesos, vamos a estar entre los países que lo lideren. Por tanto, señorías, me reafirmo en el compromiso que les manifesté a los españoles y a las españolas desde el inicio de la guerra: vamos a comprometer todos los recursos necesarios para proteger a las familias, a las industrias, a las empresas, a los hogares, a la gente, en definitiva, de nuestro país del impacto económico de la guerra de Putin en Ucrania, tal y como hicimos durante la pandemia. Como todos ustedes saben, hoy es el primer día de la Presidencia española del Consejo de la Unión Europea. En primer lugar, debemos proteger a la población civil y garantizar el acceso de la ayuda humanitaria a Gaza.

| grupo | docs | etiqueta | oraciones |
|---:|---:|---|---:|
| 0 | 49 | reindustrializ · innov · apuest · profesional | 2 |
| 1 | 43 | rotatori · celac · brusel · zelenski | 5 |
| 2 | 14 | señori · interprofesional · parlamentari · revaloriz | 2 |
| 3 | 12 | zelenski · ucranian · durader · rus | 1 |
| 4 | 8 | ham · palestin · gaz · rehen | 1 |
| 5 | 3 | aziz · ithmar · marruec · xii | 0 |

### Pedro Sánchez · 2024

Grupos: 6 · oraciones: 11 · palabras: 377 · compresión: 0.116% · sim. centroide: 0.30 · temas cubiertos: 97%

> Yo creo que va a ser una gran oportunidad para seguir caminando en esa hoja de ruta compartida entre España y China. Queremos que España, que también India, trabajen conjuntamente para abordar los grandes retos globales, reforzar también nuestra relación entre la India y la Unión Europea que, como saben, el próximo año, en el año 2025 va a cumplir un aniversario muy especial en sus relaciones estratégicas. Con esa intención, con la de compartir ese crecimiento económico inclusivo y socialmente justo, hoy vamos a firmar, junto con el presidente Boric, la renovación de la alianza estratégica entre Chile y España para reforzar nuestra relación bilateral en ese horizonte compartido por ambos Gobiernos. El nuevo Acuerdo Marco Avanzado Chile y Unión Europea, firmado en diciembre del año 2023 bajo la Presidencia española del Consejo de la Unión Europea y aprobado la semana pasada en el Parlamento Europeo, representa un hito en las relaciones comerciales entre nuestros dos países. Gracias a las políticas públicas también, que ha impulsado el Gobierno de España desde 2018. Por eso yo le digo que manifiesto mi confianza, no solo de que de que el año 2025 va a ser mejor que el año 2024 en ese horizonte que tenemos por delante de que la España de 2027 sea mejor que la España de 2023 y por supuesto, de largo. Y finalmente, desear que salga adelante este preacuerdo, porque será, como he dicho antes, bueno para para Cataluña y para España. Apostar por la convivencia en Cataluña, señoría, significa apostar por el autogobierno y es lo que vamos a hacer desde el Gobierno de España y espero que a partir del 12 de mayo con un president de la Generalitat socialista. A ver, yo creo que tenemos que partir de la base de que la pasada legislatura este Gobierno, o estas mayorías parlamentarias a las cuales usted antes ha hecho referencia, han aprobado más de 200 reformas, 200 reformas legislativas. España se sumará así a los más de 140 países que ya reconocen en el mundo a Palestina como estado. Con esto lo que quiero decir es que el Gobierno de España, como el conjunto de la sociedad española, está y estará con los valencianos y valencianas todo el tiempo que sea necesario.

| grupo | docs | etiqueta | oraciones |
|---:|---:|---|---:|
| 0 | 66 | apuest · innov · global · excelent | 4 |
| 1 | 29 | gaz · palestin · ucrani · coherent | 3 |
| 2 | 19 | señori · feijo · parlamentari · interprofesional | 2 |
| 3 | 14 | palestin · israeli · gaz · rehen | 1 |
| 4 | 10 | valencian · comunitat · valenci · ume | 1 |
| 5 | 4 | olimpi · olimp · medall · salis | 0 |

### Andrés Manuel López Obrador · 2023

Grupos: 8 · oraciones: 12 · palabras: 400 · compresión: 0.013% · sim. centroide: 0.10 · temas cubiertos: 92%

> Vamos a ir avanzando, pero donde ya de esos 15 estados, donde empezamos, primero, ya tenemos resuelto casi al 100 por ciento el abasto, ahora te lo voy a probar, de acuerdo al informe que presentamos aquí incluso el mismo martes. Hay que decir que en el Poder Judicial trabajan 55 mil personas y solamente mil 700 reciben alimentos por más de 300 millones de pesos al año. La campaña de medios aquí no termina, porque ahora se fueron contra la cancelación de 13 fideicomisos del Poder Judicial. Por eso, qué voy a estar yo pensando que vamos a transformar a México con el apoyo de los potentados, que vamos a transformar a México con el apoyo de los políticos corruptos, que vamos a transformar a México con los comentaristas de radio, famosos que se han vuelto inmensamente ricos, qué voy a estar pensando yo que vamos a transformar a México con el apoyo de Krauze, de Aguilar Camín. Vamos a seguir también con el programa de Sembrando Vida, se va a seguir ayudando hasta el final y yo creo que va a continuar, aun cuando ya no esté yo en la Presidencia, porque ya son 440 mil sembradores en todo el país y se están cultivando más de un millón d hectáreas, se están sembrando alrededor de mil millones de árboles frutales y maderables en todo el país. E hicimos el compromiso, y lo voy a cumplir, que se va a inaugurar antes de que yo termine, inauguramos todo el tren ‘Insurgente’, de Toluca a la Ciudad de México. Quedamos que vamos a hacer este informe por tramo, el lunes pasado fue el tramo 1, ahora corresponde al tramo 2, que va de Escárcega, Campeche, hasta Calkiní, los límites con Yucatán. Luego, vamos a informar sobre el Tren Maya. El día de hoy les vamos a dar el informe de seguridad conjunto del periodo del 11 al 24 del presente mes. En cuanto al registro de personas detenidas por los delitos del fuero común y federal, tenemos 17 mil 84 detenidos y, de ellos, presentados ante el Ministerio Público, 16 mil 764. En ese sentido, señor presidente, si me lo permite, vamos a llamar a cada gobernador y gobernadora a que pueda hacer la firma de este Acuerdo de Federalización del Sistema de Salud para el Bienestar. Y sí ayudamos y vamos a seguir ayudando al pueblo de México.

| grupo | docs | etiqueta | oraciones |
|---:|---:|---|---:|
| 0 | 107 | loret · cir · desinform · pri | 4 |
| 1 | 42 | discapacit · preescol · pamel · enton | 2 |
| 2 | 42 | cocher · ceram · carer · mobil | 2 |
| 3 | 37 | cartuch · metanfetamin · kilogr · cocain | 2 |
| 4 | 28 | zoe · aburt · robled · federaliz | 1 |
| 5 | 28 | enton · himn · corazon · tulum | 1 |
| 6 | 13 | otis · coyuc · evelyn · chilpancing | 0 |
| 7 | 11 | revolucionari · santacili · benit · vill | 0 |

### Javier Milei · 2024

Grupos: 5 · oraciones: 10 · palabras: 400 · compresión: 0.110% · sim. centroide: 0.17 · temas cubiertos: 97%

> De ser el país más rico del mundo, cuando abrazamos las ideas de la libertad a ser un país donde 6 de cada 10 argentinos son pobres, mientras la mayoría de los políticos – como muchos de ustedes – son ricos. Luego de más de cien años de insistir con un modelo empobrecedor y habiendo olvidado casi por completo las ideas que hicieron grande, a nuestro país, los últimos 20 años han sido, particularmente, un desastre económico; una orgía de gasto público; emisión descontrolada, que tuvo como resultado la peor herencia que ningún gobierno en la historia argentina haya recibido jamás. Para eso estamos aquí hoy, para establecer entre nosotros y junto a la sociedad un pacto de caballeros en torno a estos pilares, y para decir ante todos los argentinos de bien que reclamaron un cambio de rumbo con su voto que una Argentina distinta es imposible haciendo lo mismo de siempre. A pesar de haber sido una demanda histórica de la dirigencia política argentina y, por qué no también de la sociedad, la posibilidad de sentarnos en una mesa y establecer pautas comunes transversales a todas las fuerzas políticas siempre pareció una quimera en las últimas décadas. Eso fue lo que nosotros llamamos “la era de oro de Argentina”. Lo digo porque - lamentablemente - en los países líderes del mundo libre se vuelven cada vez más populares las ideas que hundieron a la Argentina en la miseria. Es decir, por qué existe el dinero, porque si no sabemos por qué existe el dinero, cómo vamos a entender la hipótesis monetaria de la inflación, si no entendemos de lo que estamos hablando. Argentina hizo un ajuste fiscal en 6 meses de 15 puntos del PBI, algo que no se ha visto nunca en la historia de la humanidad, y nosotros lo hemos hecho. Y ahí ustedes van a encontrar cinco artículos: hay uno donde se plantea el problema desde la perspectiva de la historia del pensamiento económico y es donde yo trato de rastrear el problema dentro de la teoría económica, que después lo voy a explicar, aparecen todos estos problemas por la teoría neoclásica. Además de ser importante para mí, por tratarse de mi primera participación como presidente, el Día de la Memoria de las Víctimas del Holocausto tiene este año un significado de especial relevancia para la comunidad judía argentina, la comunidad judía global y para todo el mundo libre.

| grupo | docs | etiqueta | oraciones |
|---:|---:|---|---:|
| 0 | 50 | miseri · empobrecedor · menem · default | 6 |
| 1 | 22 | anualiz · puts · motosierr · bausili | 2 |
| 2 | 7 | hockey · neoclas · smith · adam | 1 |
| 3 | 5 | ham · judi · antisemit · holocaust | 1 |
| 4 | 3 | cors · rigidez · imposicion · arancelari | 0 |

