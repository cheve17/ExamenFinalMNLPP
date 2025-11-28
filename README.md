# ExamenFinalMNLPP
Proyecto Final – Pronóstico del Precio del Oro con LSTM Multivariante
Alumno: Juan Pablo Echeverría
Profesor: Pedro Martinez

Este proyecto desarrolla un modelo de pronóstico de series de tiempo para el mercado financiero, utilizando datos reales del precio del oro y variables macroeconómicas relacionadas.

La solución se basa en un modelo LSTM multivariante que predice retornos logarítmicos diarios del oro y posteriormente reconstruye los precios futuros.

El modelo alcanza un desempeño sobresaliente:

MAE: $3.93

RMSE: $4.93

MAPE: 0.19%

---

1. Introducción
1.1 Variable de interés

Serie objetivo: Precio del oro (Futuros de Oro – GC=F, cierre diario).

Problema: Pronosticar la evolución del precio del oro en el corto plazo.

Tipo de problema: Pronóstico de series de tiempo financieras.

---

1.2 Motivación y relevancia

El oro es un activo refugio clave en los mercados financieros.

Su precio está influenciado por:

Fortaleza del dólar estadounidense.

Niveles de tasas de interés de largo plazo.

Expectativas macroeconómicas y de inflación.

A nivel profesional, entender la dinámica del oro ayuda a:

Analizar escenarios de riesgo financiero.

Tomar decisiones informadas sobre diversificación y cobertura.

Construir portafolios más robustos en contextos de volatilidad.

---

2. Datos y Fuente
2.1 Fuentes (Yahoo Finance vía yfinance)

Se utilizaron las siguientes series:

| Ticker | Descripción                               | Columna usada |
| ------ | ----------------------------------------- | ------------- |
| `GC=F` | Futuros del oro                           | `Close`       |
| `^DXY` | Índice del dólar estadounidense (DXY)     | `Close`       |
| `^TNX` | Rendimiento del bono del Tesoro a 10 años | `Close`       |

Periodo: Desde 2018-01-01 hasta la fecha de ejecución.

Frecuencia: Datos diarios.

---

2.2 Construcción del DataFrame

Se descargaron las series por separado y se unieron en un solo DataFrame:

Fecha (índice)
Close          -> Precio del oro
Dolar          -> Índice del dólar (DXY)
Tasa           -> Tasa a 10 años (TNX)


Se aplicaron las siguientes acciones:

Alineación temporal por fecha.

ffill() para rellenar días sin cotización (no bursátiles) con el último valor disponible.

Eliminación de filas con NaN residuales.

---

3. Preparación del Dataset
3.1 Problema de la no estacionariedad

Los precios brutos del oro presentan:

Tendencia de largo plazo.

Cambios de régimen.

No estacionariedad clara.

Esto complica seriamente la capacidad de generalización de modelos como LSTM si se entrenan directamente sobre el precio.

---

3.2 Solución: Retornos Logarítmicos

Se transformó la serie de precios a retornos logarítmicos:

Se generan tres variables de retorno:

Returns → retornos logarítmicos del oro (variable objetivo).

Dolar_Returns → retornos logarítmicos del índice del dólar.

Tasa_Returns → cambio diario en la tasa a 10 años.

Se elimina la primera fila generada por los shift(1) y diff().

---

3.3 Exploración de la serie transformada

Se graficaron los retornos logarítmicos diarios del oro:

Se observa una serie centrada en torno a cero.

Mayor estabilidad de varianza que el precio.

Adecuada para suponer estacionariedad a corto plazo.

(En el notebook se incluye la gráfica con Plotly).

---

4. Ingeniería de Características y Escalamiento
4.1 Features utilizadas

Las variables de entrada al modelo LSTM fueron:

TARGET_COLUMN   = "Returns"
FEATURE_COLUMNS = ["Returns", "Dolar_Returns", "Tasa_Returns"]

| Feature         | Descripción                                         | Rol en el modelo                         |
| --------------- | --------------------------------------------------- | ---------------------------------------- |
| `Returns`       | Retornos logarítmicos del precio del oro            | **Target** y feature principal           |
| `Dolar_Returns` | Retornos logarítmicos del índice del dólar (DXY)    | Variable externa macroeconómica clave    |
| `Tasa_Returns`  | Cambio diario en la tasa de interés a 10 años (TNX) | Proxy de condiciones monetarias globales |

---

4.2 Ventana temporal (Lookback)

LOOKBACK = 90 días

Es decir, el modelo observa los últimos 90 días de:

retornos del oro,

retornos del dólar,

cambios en la tasa,

para predecir el retorno del día siguiente.

---

4.3 Split de datos (sin fuga de información)

Se separa el DataFrame de manera cronológica:

TRAIN_SPLIT_RATIO = 0.85

85% → entrenamiento + validación.

15% → prueba (NUNCA se usa para ajustar el modelo).

---

4.4 Escalamiento

Se usó MinMaxScaler(0, 1) de sklearn:

Se ajusta (fit) solo con datos de entrenamiento.

Se transforman (transform) entrenamiento y prueba:

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)        # solo TRAIN
train_scaled = scaler.transform(train_data)
test_scaled  = scaler.transform(test_data)

---

4.5 Ventaneo multivariante

Se construyen ventanas para LSTM con la función:
X_train, y_train = crear_ventanas_multi(train_scaled, LOOKBACK, TARGET_COLUMN, FEATURE_COLUMNS)
X_test,  y_test  = crear_ventanas_multi(test_scaled,  LOOKBACK, TARGET_COLUMN, FEATURE_COLUMNS)


Cada muestra de entrada (X) tiene forma:

(LOOKBACK, n_features) = (90, 3)

y la salida (y) es el retorno del oro en el día t.

---

5. Modelado – LSTM Multivariante
5.1 Arquitectura

Se utilizó la siguiente arquitectura en Keras/TensorFlow:
model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(LOOKBACK, len(FEATURE_COLUMNS))),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1)   # predice el retorno del oro
])

LSTM 1: 128 unidades, return_sequences=True

Dropout 1: 30% (regularización)

LSTM 2: 64 unidades

Dropout 2: 30%

Dense: 1 neurona de salida (regresión)

---

5.2 Entrenamiento

Optimizador: Adam

Loss: Mean Squared Error (MSE)

Épocas máximas: 100

Batch size: 32

Validación interna: validation_split = 0.1

EarlyStopping:

monitor='val_loss'

patience=10

restore_best_weights=True

El entrenamiento se detuvo automáticamente cuando la pérdida de validación dejó de mejorar, evitando sobreajuste.

---

6. Reconstrucción de Precios y Métricas de Evaluación
6.1 De retornos a precios

Para interpretar los resultados en términos de precio en dólares, se reconstruyen los precios a partir de los retornos predichos:

Se procede así:

Se aplica la inversa del MinMaxScaler solo sobre la columna de retornos (Returns).

Se reconstruyen los precios en el conjunto de prueba usando la fórmula anterior.

Se comparan los precios reales vs. precios predichos.

---

6.2 Métricas finales (en escala de precio)

Las métricas finales en el conjunto de prueba fueron:
| Métrica  | Valor     | Interpretación                             |
| -------- | --------- | ------------------------------------------ |
| **MAE**  | **$3.93** | Error absoluto promedio muy bajo por día   |
| **RMSE** | **$4.93** | Los errores grandes están bien controlados |
| **MAPE** | **0.19%** | Precisión porcentual excepcional           |

---

6.3 Interpretación

Un MAPE de 0.19% significa que, en promedio, la predicción se desvía menos del 0.2% del precio real.

En un activo que se cotiza alrededor de $1,700–$2,000 dólares, esto implica un error diario de solo 3–5 dólares, lo cual es extremadamente competitivo para series financieras.

---

6.4 Gráfica Real vs Predicción

En el notebook se incluye una gráfica con Plotly donde:

La línea azul representa el precio real en el set de prueba.

La línea naranja/roja representa el precio predicho por el LSTM.

Visualmente, ambas curvas se superponen casi por completo, confirmando la calidad del modelo.

---

7. Pronóstico Futuro (10 días)

Se implementó una estrategia autoregresiva:

Se toma la última secuencia de 90 días de la serie escalada (X_test[-1]).

El modelo predice el retorno del día siguiente.

Este retorno:

se desescala,

se usa para calcular el nuevo precio,

se agrega a la serie para predecir el siguiente día.

Se repite el proceso durante 10 días futuros.

---

7.1 Tabla de Pronóstico LSTM (10 días)

| Fecha      | Precio Predicho |
| ---------- | --------------- |
| 2022-09-27 | $2,097.92       |
| 2022-09-28 | $2,099.17       |
| 2022-09-29 | $2,100.13       |
| 2022-09-30 | $2,100.66       |
| 2022-10-01 | $2,100.67       |
| 2022-10-02 | $2,100.17       |
| 2022-10-03 | $2,099.20       |
| 2022-10-04 | $2,097.80       |
| 2022-10-05 | $2,096.06       |
| 2022-10-06 | $2,094.05       |

---

7.2 Interpretación del pronóstico

El modelo anticipa un periodo de ligera corrección en el precio del oro, pero en niveles muy cercanos al precio actual.

Las variaciones día a día son pequeñas, lo cual es coherente con un escenario de mercado estable.

El forecast no muestra explosiones ni comportamientos erráticos, lo que refuerza la idea de que el modelo generaliza bien.

---

8. Conclusiones y Trabajo Futuro
8.1 Conclusiones principales

Transformar la serie de precios del oro a retornos logarítmicos fue clave para conseguir una serie estacionaria y un modelo estable.

El LSTM multivariante aprovechó información de:

retornos del oro,

retornos del dólar,

cambios en tasas de interés,
capturando dependencias temporales complejas.

El modelo logró un MAPE ≈ 0.19%, lo que es extraordinariamente bajo para datos financieros reales.

La reconstrucción del precio mostró que la predicción sigue estrechamente el comportamiento del mercado.

---

8.2 Posibles mejoras

Probar arquitecturas más avanzadas:

Bidirectional LSTM,

Transformers para series de tiempo.

Incluir más variables externas:

Índices bursátiles,

Inflación,

Volatilidad implícita (VIX).

Extender la evaluación a horizontes de pronóstico más largos (30, 60, 90 días), comparando trade-off entre horizonte y error.

---

9. Reproducibilidad
9.1 Requerimientos
   pip install yfinance tensorflow pandas numpy plotly scikit-learn

---

9.2 Archivos principales

ExamenFinalMNLPP.ipynb

Descarga y preparación de datos.

Ingeniería de características.

Escalamiento y ventaneo.

Entrenamiento del LSTM.

Reconstrucción de precios y métricas.

Pronóstico futuro.


