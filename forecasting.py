from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as smape_loss
from get_stock_data import get_info
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd


fecha_inicio = '2021-01-01'

df = get_info('GOLD', fecha_inicio)
df.index = df.index.to_period('B')
df_serie = df.loc[:, 'Close']

print(df_serie)

# Cargar los datos

# Dividir los datos en conjuntos de entrenamiento y prueba
y_train, y_test = temporal_train_test_split(df_serie, test_size=0.10)
print(y_test)




# Crear y ajustar el modelo de pronóstico
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)


# Crear un horizonte de pronóstico de 10 días
fh = ForecastingHorizon(
    pd.PeriodIndex(pd.bdate_range(start=str(y_test.index[0]), periods=len(y_test), freq="B")), is_relative=False
)
fh


# Realizar el pronóstico
y_pred = forecaster.predict(fh=fh)
print(y_pred)

print(len(y_test), len(y_pred))

# Calcular el error de pronóstico
error = smape_loss(y_test, y_pred)
print(f'Error de pronóstico: {error}')

# Encontrar los índices que son diferentes
# different_indices = y_test.index.difference(y_pred.index)
# print(different_indices)



import matplotlib.pyplot as plt

# Crear un DataFrame para las predicciones
y_pred = y_pred.rename('Prediction')
# print(y_pred)
# Concatenar las series original y de predicciones
# df_total = pd.concat([df, y_pred], axis=1)
# print(df_total)

df_total = df.join(y_pred, how='left')
df_total.index = df_total.index.to_series().astype(str)

print(df_total)

# Graficar las series
plt.figure(figsize=(10,6))
plt.plot(df_total.index, df_total['Close'], label='Original')
plt.plot(df_total.index, df_total['Prediction'], label='Predicciones')
plt.title('Precio de Accion: Original vs Predicciones')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.show()