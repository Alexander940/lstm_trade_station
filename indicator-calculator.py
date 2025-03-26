import csv
import numpy as np

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

def generate_csv(high, low, up, down, length=14, output_file="tema_results.csv"):
    """
    Genera un archivo CSV con los resultados de los cálculos
    
    Args:
        high: Lista o array con los precios altos
        low: Lista o array con los precios bajos
        up: Lista o array con los valores up
        down: Lista o array con los valores down
        length: Longitud para el cálculo del TEMA (por defecto 14)
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
    
    # Crear el archivo CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Escribir encabezados
        writer.writerow(['Índice', 'TEMA_High', 'TEMA_Low', 'TEMA_Up', 'TEMA_Down'])
        
        # Escribir datos
        for i in range(len(high)):
            writer.writerow([i, tema_high[i], tema_low[i], tema_up[i], tema_down[i]])
    
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
    
    # Generar el archivo CSV con los resultados
    generate_csv(high, low, up, down, length=14)