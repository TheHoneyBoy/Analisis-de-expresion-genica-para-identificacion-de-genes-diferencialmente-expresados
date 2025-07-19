import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def load_microarray_data(file_path):
    """Carga datos de microarray desde un archivo series_matrix.txt"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    data = pd.read_csv(file_path, delimiter='\t', comment='!')
    return data.set_index('ID_REF')

def clean_data(data):
    """Limpia los datos: convierte a numérico y elimina columnas no numéricas"""
    numeric_data = data.apply(pd.to_numeric, errors='coerce')
    return numeric_data.dropna(axis=1, how='all')

def normalize_data(data, method='none'):
    """Aplica normalización según el método especificado"""
    cleaned_data = clean_data(data)
    
    if method == 'none':
        return cleaned_data
    
    elif method == 'log2':
        # Aplicamos log2 directamente a los datos originales
        log_data = np.log2(cleaned_data)
        return log_data
    
    elif method == 'zscore':
        scaler = StandardScaler()
        zscore_data = pd.DataFrame(
            scaler.fit_transform(cleaned_data),
            columns=cleaned_data.columns,
            index=cleaned_data.index
        )
        return zscore_data
    
    elif method == 'robust':
        robust_scaler = RobustScaler()
        robust_data = pd.DataFrame(
            robust_scaler.fit_transform(cleaned_data),
            columns=cleaned_data.columns,
            index=cleaned_data.index
        )
        return robust_data
    
    return cleaned_data