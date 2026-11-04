# Proyecto 3: clasificación de sentimientos en tweets en español

Proyecto de procesamiento de lenguaje natural. La idea es leer un tweet en español y decir si expresa un sentimiento negativo, neutro o positivo.

Todo el desarrollo está en un notebook: carga de datos, exploración de las tres clases, preparación del texto y fine-tuning de BETO con tres tasas de aprendizaje. Se elige la mejor configuración por F1 macro en validación y después se evalúa en prueba.


## Dataset

Fuente: [Tweet Sentiment Multilingual, de Cardiff NLP](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual). Usamos únicamente el subconjunto `spanish`. Es un problema concreto en español, de tamaño manejable, con particiones ya definidas.

| Partición | Tweets | Uso |
|---|---:|---|
| Entrenamiento | 1.839 | Ajustar los pesos |
| Validación | 324 | Elegir la configuración |
| Prueba | 870 | Evaluar la configuración final |
| Total | 3.033 | |

Cada registro tiene `text`, el texto de entrada, y `label`, la etiqueta. Leemos los JSONL públicos directamente y fijamos su revisión. Los archivos tienen un registro más por partición que los conteos publicados en la ficha; usamos su contenido completo.

### Variable objetivo

| Etiqueta | Sentimiento | Ejemplo inventado |
|---|---|---|
| 0 | Negativo | «Pésimo servicio, nadie me responde». |
| 1 | Neutro | «La reunión empieza a las tres». |
| 2 | Positivo | «Me encantó, lo recomiendo». |

Es un problema **supervisado y multiclase**, con una sola etiqueta por tweet. Su modalidad es **sequence classification**, porque clasificamos el texto completo.

Las clases están balanceadas: 613 ejemplos por clase en entrenamiento, 108 en validación y 290 en prueba. Los controles no encontraron textos vacíos, duplicados exactos internos ni textos idénticos entre particiones.

## Metodología

1. **Explorar.** Contamos clases, mostramos ejemplos y analizamos longitudes para entender la tarea y revisar la calidad.
2. **Preparar el texto.** El tokenizador de BETO convierte los textos en identificadores numéricos. Usamos hasta 128 tokens, padding dinámico y máscara de atención. Conservamos negaciones, acentos, mayúsculas y emojis porque pueden aportar información.
3. **Establecer una referencia.** Una regla predice siempre la clase más frecuente. Como hay empate, la biblioteca lo resuelve de forma determinista. Es un punto de comparación.
4. **Hacer fine-tuning.** Partimos de [BETO cased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased), un BERT preentrenado en español, añadimos una capa de clasificación y ajustamos todos sus pesos con los tweets etiquetados.
5. **Comparar configuraciones.** Entrenamos tres veces desde el mismo modelo base, cambiando la tasa de aprendizaje y reiniciando la semilla.
6. **Elegir y evaluar.** Seleccionamos por F1 macro de validación; en empate, por accuracy y menor tasa. Evaluamos el ganador en prueba con métricas, matriz de confusión y ejemplos de errores.

### Hiperparámetros

| Hiperparámetro | Valor |
|---|---|
| Learning rate | 2e-5, 3e-5, 5e-5 |
| Épocas por experimento | 3 |
| Batch | 8 |
| Acumulación de gradientes | 2; batch efectivo de 16 en una GPU |
| Longitud máxima | 128 tokens |
| Weight decay | 0,01 |
| Warmup | 10 % de los pasos |
| Semilla | 42 |

La tasa de aprendizaje controla el tamaño de las actualizaciones de los pesos. Una época es un recorrido completo por los datos de entrenamiento. Comparamos tres configuraciones de BETO, no tres arquitecturas diferentes.

## Resultados

La exploración muestra clases balanceadas y textos de entrenamiento de entre 19 y 138 caracteres. El notebook incluye ejemplos y figuras con salidas reales.

El notebook publicado contiene la ejecución completa en Google Colab con GPU Tesla T4. Las tres configuraciones terminaron sus tres épocas y se compararon usando validación:

| Learning rate | F1 macro de validación | Accuracy de validación |
|---|---:|---:|
| **2e-5** | **0,6952** | **0,6975** |
| 3e-5 | 0,6843 | 0,6883 |
| 5e-5 | 0,6732 | 0,6790 |

Ganó **BETO con learning rate 2e-5**. La evaluación final sobre los 870 tweets de prueba produjo:

| Modelo | Accuracy de prueba | F1 macro de prueba |
|---|---:|---:|
| Referencia constante | 0,3333 | 0,1667 |
| **BETO seleccionado** | **0,6885** | **0,6886** |

El modelo acertó **599 tweets de 870 (68,85 %)** y cometió 271 errores. El sentimiento neutro tuvo el menor F1 (0,5966); positivo obtuvo 0,7443 y negativo 0,7248. La confusión individual más frecuente fue clasificar como negativos 65 tweets neutros.

BETO superó la referencia constante en esta ejecución. La selección del modelo utilizó exclusivamente validación; prueba se reservó para evaluar la configuración elegida. Accuracy es la proporción de aciertos y F1 macro da el mismo peso a las tres clases al promediar su F1.

Las salidas guardadas incluyen las tablas de comparación, el reporte por clase, la matriz de confusión y ejemplos de errores y predicciones. Al ejecutarlo también se generan `comparacion_experimentos.csv` y `metricas_prueba.csv`.

## Limitaciones

El conjunto es pequeño y comparamos con una sola semilla. El sarcasmo y el contexto ausente pueden dificultar la tarea; se revisan errores concretos antes de atribuirles una causa. El desempeño aquí no garantiza el mismo resultado en otras plataformas o regiones.

## Estructura del repositorio

- `Proyecto_3_Tweets_BETO.ipynb`: descripción, exploración, entrenamiento y evaluación.
- `README.md`: descripción y reproducción.
- `requirements.txt`: dependencias.
- `.gitignore`: excluye pesos, checkpoints y archivos temporales de Git.
## Reproducción

El notebook se ejecuta con Python 3 y una GPU NVIDIA; es compatible con Google Colab y GPU T4.
La primera celda instala las dependencias. Las celdas se ejecutan en orden para descargar los datos,
realizar la exploración, entrenar las tres configuraciones y evaluar la seleccionada.

El entrenamiento guarda las métricas, figuras, configuración y el modelo ganador en `resultados_tweets/`.
Los pesos y checkpoints se excluyen del control de versiones mediante `.gitignore`.


## Referencias

- Barbieri, F., Espinosa-Anke, L. y Camacho-Collados, J. (2022). [XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond](https://aclanthology.org/2022.lrec-1.27/).
- [Ficha y condiciones del dataset](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual): atribución CC BY 3.0 y condiciones de Twitter indicadas por los autores. No redistribuimos la base completa.
- [BETO: Spanish BERT](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased).
- [Documentación de Trainer 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).
