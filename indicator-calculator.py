import csv
import numpy as np
import pandas as pd

def x_average(price_series, length):
    """
    Implementación de la función XAverage (EMA - Exponential Moving Average)
    
    Args:
        price_series: Lista o array con los precios
        length: Longitud para el cálculo del promedio móvil
    
    Returns:
        Array con los valores del promedio móvil exponencial
    """
    if length < 0:
        raise ValueError("La longitud del promedio móvil debe ser mayor o igual a 0")
    
    smoothing_factor = 2 / (length + 1)
    x_average_values = np.zeros(len(price_series))
    
    # Inicializar el primer valor
    x_average_values[0] = price_series[0]
    
    # Calcular los demás valores
    for i in range(1, len(price_series)):
        x_average_values[i] = x_average_values[i-1] + smoothing_factor * (price_series[i] - x_average_values[i-1])
    
    return x_average_values

def tema(price_series, length):
    """
    Calcula el Triple Exponential Moving Average (TEMA)
    
    Args:
        price_series: Lista o array con los precios
        length: Longitud para el cálculo
    
    Returns:
        Array con los valores TEMA
    """
    # Calcular XMA (primer EMA)
    xma = x_average(price_series, length)
    
    # Calcular XMA2 (EMA del primer EMA)
    xma2 = x_average(xma, length)
    
    # Calcular XMA3 (EMA del segundo EMA)
    xma3 = x_average(xma2, length)
    
    # Calcular TEMA
    tema_values = 3 * xma - 3 * xma2 + xma3
    
    return tema_values

def calculate_macd(close_prices, fast_length=12, slow_length=26, macd_length=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) indicator.
    
    Parameters:
    -----------
    close_prices : array-like
        Closing prices of the financial instrument
    fast_length : int, optional (default=12)
        Number of periods for the fast moving average
    slow_length : int, optional (default=26)
        Number of periods for the slow moving average
    macd_length : int, optional (default=9)
        Number of periods for the MACD signal line
    
    Returns:
    --------
    dict containing MACD components and signals
    """
    # Convert input to numpy array for compatibility
    close_prices = np.array(close_prices)
    
    # Calculate exponential moving averages
    fast_ema = pd.Series(close_prices).ewm(span=fast_length, adjust=False).mean().values
    slow_ema = pd.Series(close_prices).ewm(span=slow_length, adjust=False).mean().values
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate MACD signal line
    macd_signal = pd.Series(macd_line).ewm(span=macd_length, adjust=False).mean().values
    
    # Calculate MACD histogram
    macd_histogram = macd_line - macd_signal
    
    # Detect crossovers
    crossover_up = np.where((macd_histogram > 0) & (np.roll(macd_histogram, 1) <= 0))[0]
    crossover_down = np.where((macd_histogram < 0) & (np.roll(macd_histogram, 1) >= 0))[0]
    
    return {
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_histogram': macd_histogram,
        'crossover_up': crossover_up,
        'crossover_down': crossover_down
    }

class MACDIndicator:
    def __init__(self, 
                 fast_length=12, 
                 slow_length=26, 
                 macd_length=9, 
                 alert_if_cross_up=True, 
                 alert_if_cross_down=True):
        """
        MACD Indicator with customizable parameters and alert settings.
        
        Parameters:
        -----------
        fast_length : int, optional (default=12)
            Length of fast moving average
        slow_length : int, optional (default=26)
            Length of slow moving average
        macd_length : int, optional (default=9)
            Length of MACD signal line
        alert_if_cross_up : bool, optional (default=True)
            Enable alerts for crossovers above zero
        alert_if_cross_down : bool, optional (default=True)
            Enable alerts for crossovers below zero
        """
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.macd_length = macd_length
        self.alert_if_cross_up = alert_if_cross_up
        self.alert_if_cross_down = alert_if_cross_down
    
    def calculate(self, close_prices):
        """
        Calculate MACD indicator for given close prices.
        
        Parameters:
        -----------
        close_prices : array-like
            Closing prices of the financial instrument
        
        Returns:
        --------
        dict containing MACD calculation results
        """
        macd_results = calculate_macd(
            close_prices, 
            self.fast_length, 
            self.slow_length, 
            self.macd_length
        )
        
        # Optional alert logic can be added here
        if self.alert_if_cross_up:
            for idx in macd_results['crossover_up']:
                print(f"Alert: MACD crossed over zero at index {idx}")
        
        if self.alert_if_cross_down:
            for idx in macd_results['crossover_down']:
                print(f"Alert: MACD crossed under zero at index {idx}")
        
        return macd_results

def generate_csv(high, low, up, down, close=None, length=14, macd_fast=12, macd_slow=26, macd_signal=9, output_file="indicators_results.csv"):
    """
    Genera un archivo CSV con los resultados de los cálculos
    
    Args:
        high: Lista o array con los precios altos
        low: Lista o array con los precios bajos
        up: Lista o array con los valores up
        down: Lista o array con los valores down
        close: Lista o array con los precios de cierre (para MACD)
        length: Longitud para el cálculo del TEMA (por defecto 14)
        macd_fast: Longitud para el EMA rápido del MACD (por defecto 12)
        macd_slow: Longitud para el EMA lento del MACD (por defecto 26)
        macd_signal: Longitud para la señal MACD (por defecto 9)
        output_file: Nombre del archivo de salida
    """
    # Asegurarse de que todos los arrays tengan la misma longitud
    if not (len(high) == len(low) == len(up) == len(down)):
        raise ValueError("Todos los arrays de entrada deben tener la misma longitud")
    
    # Calcular TEMA para cada serie
    tema_high = tema(high, length)
    tema_low = tema(low, length)
    tema_up = tema(up, length)
    tema_down = tema(down, length)
    
    # Calcular MACD si se proporcionan precios de cierre
    macd_results = None
    if close is not None:
        if len(close) != len(high):
            raise ValueError("El array de precios de cierre debe tener la misma longitud que los otros arrays")
        macd_results = calculate_macd(close, macd_fast, macd_slow, macd_signal)
    
    # Crear el archivo CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escribir encabezados
        headers = ['Índice', 'TEMA_High', 'TEMA_Low', 'TEMA_Up', 'TEMA_Down']
        if macd_results is not None:
            headers.extend(['MACD_Line', 'MACD_Signal', 'MACD_Histogram'])
        writer.writerow(headers)
        
        # Escribir datos
        for i in range(len(high)):
            row = [i, tema_high[i], tema_low[i], tema_up[i], tema_down[i]]
            if macd_results is not None:
                row.extend([macd_results['macd_line'][i], 
                           macd_results['macd_signal'][i], 
                           macd_results['macd_histogram'][i]])
            writer.writerow(row)
    
    print(f"Archivo CSV generado: {output_file}")

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo con datos de prueba
    # Estos arrays deberían ser reemplazados con tus datos reales
    sample_size = 100
    high = np.random.uniform(100, 110, sample_size)  # Ejemplo de precios altos
    low = np.random.uniform(90, 100, sample_size)    # Ejemplo de precios bajos
    up = np.random.uniform(0, 10, sample_size)       # Ejemplo de valores up
    down = np.random.uniform(0, 10, sample_size)     # Ejemplo de valores down
    close = np.random.uniform(95, 105, sample_size)  # Ejemplo de precios de cierre
    
    # Generar el archivo CSV con los resultados
    generate_csv(high, low, up, down, close=close, length=14)
    
    # Ejemplo de uso de MACD
    print("\nEjemplo de cálculo de MACD:")
    macd_indicator = MACDIndicator()
    results = macd_indicator.calculate(close)
    
    print(f"Tamaño del array MACD_Line: {len(results['macd_line'])}")
    print(f"Primeros 5 valores de MACD_Line: {results['macd_line'][:5]}")
    print(f"Crossover Up Indices: {results['crossover_up']}")
    print(f"Crossover Down Indices: {results['crossover_down']}")