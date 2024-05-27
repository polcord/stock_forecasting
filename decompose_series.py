from get_stock_data import get_info
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_seasonal_decomposition(ts_data):
    """
    Realiza la descomposición de una serie temporal y visualiza sus componentes.

    Args:
        ts_data (pd.Series): Serie temporal a descomponer.

    Returns:
        None
    """
    decomposition = seasonal_decompose(ts_data, model='additive', period=7)

    plt.figure(figsize=(10, 6))
    plt.subplot(411)
    plt.plot(ts_data, label='Original')
    plt.legend()
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Tendencia')
    plt.legend()
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Estacionalidad')
    plt.legend()
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuos')
    plt.legend()
    plt.tight_layout()
    plt.show()


fecha_inicio = '2024-01-01'
frecuencia = 'B'

df = get_info('GOLD', fecha_inicio)
df_serie = df.loc[:, 'Close']

# Llama a la función para descomponer y visualizar los componentes
plot_seasonal_decomposition(df_serie)


