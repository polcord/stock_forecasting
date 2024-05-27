from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as smape_loss
from get_stock_data import get_info
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd
from sktime.utils.plotting import plot_series

fecha_inicio = '2021-01-01'
frecuencia = 'D'

df = get_info('GOLD', fecha_inicio)
df.index = df.index.to_period(frecuencia)
df_serie = df.loc[:, 'Close']

# Cargar los datos

# Dividir los datos en conjuntos de entrenamiento y prueba
y_train, y_test = temporal_train_test_split(df_serie, test_size=0.10)


# Crear y ajustar el modelo de pronóstico
# forecaster = NaiveForecaster(strategy="last")
# forecaster = AutoARIMA(
#     suppress_warnings=True
# ) 

# forecaster.fit(y_train)


# # Crear un horizonte de pronóstico 
fh = ForecastingHorizon(
    pd.PeriodIndex(pd.bdate_range(start=str(y_test.index[0]), 
                                  periods=len(y_test), 
                                  freq=frecuencia)), is_relative=False
)


# # Realizar el pronóstico
# y_pred = forecaster.predict(fh=fh)
# y_pred = y_pred.rename('Prediction')


# # Calcular el error de pronóstico
# error = smape_loss(y_test, y_pred)
# print(f'Error de pronóstico: {error}')

# # plotting for illustration
# plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

from sktime.datasets import load_airline
from sktime.forecasting.trend import ProphetPiecewiseLinearTrendForecaster
# y =load_airline().to_timestamp(freq='M')
# y_train, y_test = temporal_train_test_split(y)
# fh = ForecastingHorizon(y.index, is_relative=False)
forecaster =  ProphetPiecewiseLinearTrendForecaster() 
forecaster.fit(y_train) 
y_pred = forecaster.predict(fh) 
print(y_pred)

# import matplotlib.pyplot as plt

# # Crear un DataFrame para las predicciones
# df_total = df.join(y_pred, how='left')
# df_total.index = df_total.index.to_series().astype(str)

# print(df_total)

# # Graficar las series
# plt.figure(figsize=(10,6))
# plt.plot(df_total.index, df_total['Close'], label='Original')
# plt.plot(df_total.index, df_total['Prediction'], label='Predicciones')
# plt.title('Precio de Accion: Original vs Predicciones')
# plt.xlabel('Fecha')
# plt.ylabel('Precio')
# plt.legend()
# plt.show()