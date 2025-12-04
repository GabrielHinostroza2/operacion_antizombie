"""
SCRIPT 07: LIMPIEZA DE DATOS
CRISP-DM Fase 3: Data Preparation

Limpieza y manejo de datos faltantes y outliers

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from config import *
from utils import *

def handle_missing_values(df, dataset_name='Dataset'):
    """Manejar valores faltantes"""
    print_subsection(f"Manejando valores faltantes: {dataset_name}")

    missing_summary = get_missing_data_summary(df)

    if len(missing_summary) == 0:
        print("   No hay valores faltantes")
        return df

    # Imputacion numericas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        imputer_num = SimpleImputer(strategy=IMPUTATION_STRATEGY_NUM)
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
        print(f"   Imputadas {len(numeric_cols)} columnas numericas con {IMPUTATION_STRATEGY_NUM}")

    # Imputacion categoricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Desconocido', inplace=True)

    print(f"   Limpieza completada")
    return df

def handle_outliers(df, dataset_name='Dataset'):
    """Manejar outliers usando winsorization"""
    print_subsection(f"Manejando outliers: {dataset_name}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if 'ID' not in col.upper()]

    outliers_df = detect_outliers_iqr(df, numeric_cols)

    # Contar outliers por columna
    outlier_counts = outliers_df.sum()
    cols_with_outliers = outlier_counts[outlier_counts > 0]

    if len(cols_with_outliers) == 0:
        print("   No se detectaron outliers significativos")
        return df

    print(f"   Columnas con outliers: {len(cols_with_outliers)}")

    # Aplicar winsorization (cap en percentiles 1 y 99)
    for col in cols_with_outliers.index:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p1, upper=p99)
        print(f"      {col}: {outlier_counts[col]} outliers tratados")

    return df

def validate_data_types(df, dataset_name='Dataset'):
    """Validar y corregir tipos de datos"""
    print_subsection(f"Validando tipos de datos: {dataset_name}")

    # Convertir fechas
    date_cols = [col for col in df.columns if 'fecha' in col.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"   Convertido a datetime: {col}")
        except:
            pass

    # Asegurar tipos numericos
    for col in df.columns:
        if 'nivel' in col.lower() or 'tasa' in col.lower() or 'porcentual' in col.lower():
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"   Convertido a numerico: {col}")
                except:
                    pass

    return df

def remove_duplicates(df, dataset_name='Dataset'):
    """Eliminar duplicados"""
    print_subsection(f"Verificando duplicados: {dataset_name}")

    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)

    duplicates = initial_count - final_count
    if duplicates > 0:
        print(f"   Eliminados {duplicates} registros duplicados")
    else:
        print(f"   No se encontraron duplicados")

    return df

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 07: LIMPIEZA DE DATOS",
             "Preparacion y limpieza de datos")

    setup_directories()

    # Cargar datos
    print_section("CARGANDO DATOS")
    try:
        pacientes = load_dataframe('pacientes_clean.csv')
        evolucion = load_dataframe('evolucion_clean.csv')
        tratamientos = load_dataframe('tratamientos_clean.csv')
        red_contagios = load_dataframe('red_contagios_clean.csv')
    except FileNotFoundError:
        print("ERROR: Ejecute primero 01_data_loading.py")
        return

    # Limpiar cada dataset
    print_section("LIMPIEZA: PACIENTES")
    pacientes = validate_data_types(pacientes, 'Pacientes')
    pacientes = remove_duplicates(pacientes, 'Pacientes')
    pacientes = handle_missing_values(pacientes, 'Pacientes')
    pacientes = handle_outliers(pacientes, 'Pacientes')

    print_section("LIMPIEZA: EVOLUCION")
    evolucion = validate_data_types(evolucion, 'Evolucion')
    evolucion = remove_duplicates(evolucion, 'Evolucion')
    evolucion = handle_missing_values(evolucion, 'Evolucion')

    print_section("LIMPIEZA: TRATAMIENTOS")
    tratamientos = validate_data_types(tratamientos, 'Tratamientos')
    tratamientos = remove_duplicates(tratamientos, 'Tratamientos')
    tratamientos = handle_missing_values(tratamientos, 'Tratamientos')

    print_section("LIMPIEZA: RED CONTAGIOS")
    red_contagios = validate_data_types(red_contagios, 'RedContagios')
    red_contagios = remove_duplicates(red_contagios, 'RedContagios')
    red_contagios = handle_missing_values(red_contagios, 'RedContagios')

    # Guardar datos limpios
    print_section("GUARDANDO DATOS LIMPIOS")
    save_dataframe(pacientes, 'pacientes_cleaned.csv')
    save_dataframe(evolucion, 'evolucion_cleaned.csv')
    save_dataframe(tratamientos, 'tratamientos_cleaned.csv')
    save_dataframe(red_contagios, 'red_contagios_cleaned.csv')

    # Guardar log de limpieza
    log_lines = [
        "="*80,
        "LOG DE LIMPIEZA DE DATOS",
        f"Fecha: {pd.Timestamp.now()}",
        "="*80,
        "",
        f"Pacientes: {len(pacientes)} registros",
        f"Evolucion: {len(evolucion)} registros",
        f"Tratamientos: {len(tratamientos)} registros",
        f"Red Contagios: {len(red_contagios)} registros",
        "",
        "Operaciones realizadas:",
        "- Validacion de tipos de datos",
        "- Eliminacion de duplicados",
        "- Imputacion de valores faltantes",
        "- Tratamiento de outliers (winsorization)",
        "="*80
    ]

    log_path = DATA_PROCESSED / 'cleaning_log.txt'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log_step("SCRIPT 07 COMPLETADO",
             "Datos limpios guardados en datos_procesados/")

if __name__ == "__main__":
    main()
