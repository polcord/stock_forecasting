import yfinance as yf
from datetime import datetime
import pandas as pd

def get_info(symbol, start_date):
    
    # Obtén la fecha de hoy
    hoy = datetime.today().strftime('%Y-%m-%d')

    # Descarga los datos históricos de la acción una fecha de inicio delimitada hasta hoy
    data = yf.download(symbol, start=start_date, end=hoy)

    return data