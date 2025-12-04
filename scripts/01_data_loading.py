"""
SCRIPT 01: CARGA Y VALIDACION DE DATOS
CRISP-DM Fase 2: Data Understanding

Este script:
1. Carga los 4 datasets desde archivos Excel
2. Valida estructura y tipos de datos
3. Genera estadisticas iniciales
4. Exporta a CSV para carga rapida en scripts posteriores
5. Crea reporte de calidad de datos inicial

Autor: Proyecto Operacion Anti-Zombie
Fecha: 2025
"""

import sys
from pathlib import Path

# Agregar el directorio scripts al path para imports
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime

# Importar configuracion y utilidades
from config import *
from utils import *

# ==============================================================================
# FUNCIONES DE CARGA
# ==============================================================================

def load_pacientes():
    """
    Cargar dataset de pacientes del brote zombie

    Returns:
        DataFrame de pacientes
    """
    print_subsection("Cargando dataset: PACIENTES")

    filepath = get_data_path('pacientes')
    print(f"Archivo: {filepath}")

    df = pd.read_excel(filepath)

    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df

def load_evolucion():
    """
    Cargar dataset de evolucion del brote

    Returns:
        DataFrame de evolucion
    """
    print_subsection("Cargando dataset: EVOLUCION BROTE")

    filepath = get_data_path('evolucion')
    print(f"Archivo: {filepath}")

    df = pd.read_excel(filepath)

    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df

def load_tratamientos():
    """
    Cargar dataset de tratamientos experimentales

    Returns:
        DataFrame de tratamientos
    """
    print_subsection("Cargando dataset: TRATAMIENTOS")

    filepath = get_data_path('tratamientos')
    print(f"Archivo: {filepath}")

    df = pd.read_excel(filepath)

    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df

def load_red_contagios():
    """
    Cargar dataset de red de contagios

    Returns:
        DataFrame de red de contagios
    """
    print_subsection("Cargando dataset: RED DE CONTAGIOS")

    filepath = get_data_path('red_contagios')
    print(f"Archivo: {filepath}")

    df = pd.read_excel(filepath)

    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df

# ==============================================================================
# FUNCIONES DE VALIDACION
# ==============================================================================

def validate_pacientes(df):
    """
    Validar estructura y datos del dataset de pacientes

    Args:
        df: DataFrame de pacientes

    Returns:
        Dict con resultados de validacion
    """
    print_subsection("Validando dataset: PACIENTES")

    validation = {
        'dataset': 'pacientes',
        'rows': len(df),
        'columns': len(df.columns),
        'issues': []
    }

    # Verificar columnas esperadas (las criticas)
    expected_cols = [
        'ID_Paciente', 'Edad', 'Sexo', 'Estado_Actual',
        'Nivel_Zombificacion', 'Tratamiento_Recibido'
    ]

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        validation['issues'].append(f"Columnas faltantes: {missing_cols}")
        print(f"   ADVERTENCIA: Columnas faltantes: {missing_cols}")

    # Verificar duplicados en ID
    if 'ID_Paciente' in df.columns:
        duplicates = df['ID_Paciente'].duplicated().sum()
        if duplicates > 0:
            validation['issues'].append(f"IDs duplicados: {duplicates}")
            print(f"   ADVERTENCIA: {duplicates} IDs duplicados")
        else:
            print(f"   OK: Sin IDs duplicados")

    # Verificar valores nulos en columnas criticas
    for col in expected_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_pct = (null_count / len(df)) * 100
                print(f"   INFO: {col}: {null_count} nulos ({null_pct:.1f}%)")

    # Verificar rangos de valores
    if 'Edad' in df.columns:
        if df['Edad'].min() < 0 or df['Edad'].max() > 120:
            validation['issues'].append("Edad fuera de rango valido")
            print(f"   ADVERTENCIA: Edad fuera de rango: {df['Edad'].min()} - {df['Edad'].max()}")

    if 'Nivel_Zombificacion' in df.columns:
        if df['Nivel_Zombificacion'].min() < 0 or df['Nivel_Zombificacion'].max() > 100:
            validation['issues'].append("Nivel_Zombificacion fuera de rango")
            print(f"   ADVERTENCIA: Nivel_Zombificacion fuera de rango: {df['Nivel_Zombificacion'].min()} - {df['Nivel_Zombificacion'].max()}")

    if len(validation['issues']) == 0:
        print("   VALIDACION EXITOSA")
    else:
        print(f"   VALIDACION COMPLETA con {len(validation['issues'])} advertencias")

    return validation

def validate_evolucion(df):
    """
    Validar dataset de evolucion

    Args:
        df: DataFrame de evolucion

    Returns:
        Dict con resultados de validacion
    """
    print_subsection("Validando dataset: EVOLUCION")

    validation = {
        'dataset': 'evolucion',
        'rows': len(df),
        'columns': len(df.columns),
        'issues': []
    }

    expected_cols = ['Fecha', 'Casos_Nuevos', 'Tasa_Contagio_R0']

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        validation['issues'].append(f"Columnas faltantes: {missing_cols}")

    # Verificar que no haya valores negativos en casos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith('Casos') or col.startswith('Tasa'):
            if df[col].min() < 0:
                validation['issues'].append(f"{col} tiene valores negativos")
                print(f"   ADVERTENCIA: {col} tiene valores negativos")

    if len(validation['issues']) == 0:
        print("   VALIDACION EXITOSA")

    return validation

def validate_tratamientos(df):
    """
    Validar dataset de tratamientos

    Args:
        df: DataFrame de tratamientos

    Returns:
        Dict con resultados de validacion
    """
    print_subsection("Validando dataset: TRATAMIENTOS")

    validation = {
        'dataset': 'tratamientos',
        'rows': len(df),
        'columns': len(df.columns),
        'issues': []
    }

    expected_cols = ['ID_Tratamiento', 'Nombre_Tratamiento', 'Tasa_Exito_Promedio']

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        validation['issues'].append(f"Columnas faltantes: {missing_cols}")

    # Verificar tasa de exito entre 0 y 100
    if 'Tasa_Exito_Promedio' in df.columns:
        if df['Tasa_Exito_Promedio'].min() < 0 or df['Tasa_Exito_Promedio'].max() > 100:
            validation['issues'].append("Tasa_Exito_Promedio fuera de rango [0-100]")

    if len(validation['issues']) == 0:
        print("   VALIDACION EXITOSA")

    return validation

def validate_red_contagios(df):
    """
    Validar dataset de red de contagios

    Args:
        df: DataFrame de red de contagios

    Returns:
        Dict con resultados de validacion
    """
    print_subsection("Validando dataset: RED CONTAGIOS")

    validation = {
        'dataset': 'red_contagios',
        'rows': len(df),
        'columns': len(df.columns),
        'issues': []
    }

    expected_cols = ['ID_Contagio', 'ID_Infectado', 'ID_Infectante']

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        validation['issues'].append(f"Columnas faltantes: {missing_cols}")

    # Verificar que no haya auto-contagios
    if 'ID_Infectado' in df.columns and 'ID_Infectante' in df.columns:
        auto_contagios = (df['ID_Infectado'] == df['ID_Infectante']).sum()
        if auto_contagios > 0:
            validation['issues'].append(f"Auto-contagios detectados: {auto_contagios}")
            print(f"   ADVERTENCIA: {auto_contagios} casos de auto-contagio")

    if len(validation['issues']) == 0:
        print("   VALIDACION EXITOSA")

    return validation

# ==============================================================================
# FUNCIONES DE EXPORTACION
# ==============================================================================

def export_to_csv(df, name):
    """
    Exportar DataFrame a CSV

    Args:
        df: DataFrame
        name: Nombre del dataset

    Returns:
        Path del archivo guardado
    """
    filename = f"{name}_clean.csv"
    filepath = save_dataframe(df, filename)
    return filepath

# ==============================================================================
# FUNCION DE REPORTE
# ==============================================================================

def generate_data_quality_report(validations, datasets):
    """
    Generar reporte de calidad de datos

    Args:
        validations: Lista de diccionarios de validacion
        datasets: Dict con los DataFrames

    Returns:
        Path del archivo de reporte
    """
    print_subsection("Generando reporte de calidad de datos")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("REPORTE DE CALIDAD DE DATOS")
    report_lines.append("Operacion Anti-Zombie - Pipeline CRISP-DM")
    report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    report_lines.append("")

    # Resumen general
    report_lines.append("RESUMEN GENERAL")
    report_lines.append("-"*80)
    total_rows = sum(len(df) for df in datasets.values())
    total_cols = sum(len(df.columns) for df in datasets.values())
    total_memory = sum(df.memory_usage(deep=True).sum() for df in datasets.values()) / 1024**2

    report_lines.append(f"Total de datasets: {len(datasets)}")
    report_lines.append(f"Total de filas: {total_rows:,}")
    report_lines.append(f"Total de columnas: {total_cols}")
    report_lines.append(f"Memoria total: {total_memory:.2f} MB")
    report_lines.append("")

    # Detalle por dataset
    for validation in validations:
        report_lines.append("")
        report_lines.append(f"DATASET: {validation['dataset'].upper()}")
        report_lines.append("-"*80)
        report_lines.append(f"Filas: {validation['rows']:,}")
        report_lines.append(f"Columnas: {validation['columns']}")

        if validation['issues']:
            report_lines.append(f"Problemas encontrados: {len(validation['issues'])}")
            for issue in validation['issues']:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.append("Sin problemas detectados")

        # Agregar info de valores faltantes
        dataset_name = validation['dataset']
        if dataset_name in datasets:
            df = datasets[dataset_name]
            missing = get_missing_data_summary(df)
            if len(missing) > 0:
                report_lines.append("")
                report_lines.append("Valores faltantes:")
                for _, row in missing.iterrows():
                    report_lines.append(
                        f"  - {row['Column']}: {int(row['Missing_Count'])} ({row['Missing_Percentage']:.1f}%)"
                    )

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("FIN DEL REPORTE")
    report_lines.append("="*80)

    # Guardar reporte
    report_text = "\n".join(report_lines)
    report_path = DATA_PROCESSED / "data_quality_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"   Reporte guardado: {report_path.relative_to(PROJECT_ROOT)}")

    # Tambien imprimir en consola
    print("\n" + report_text)

    return report_path

# ==============================================================================
# FUNCION PRINCIPAL
# ==============================================================================

def main():
    """
    Funcion principal del script
    """
    print(BANNER)
    log_step(
        "SCRIPT 01: CARGA Y VALIDACION DE DATOS",
        "CRISP-DM Fase 2: Data Understanding\n"
        "Cargando datasets desde Excel y exportando a CSV"
    )

    # Setup de directorios
    setup_directories()

    # ==============================================================================
    # PASO 1: CARGAR DATASETS
    # ==============================================================================

    print_section("PASO 1: CARGA DE DATASETS")

    pacientes = load_pacientes()
    evolucion = load_evolucion()
    tratamientos = load_tratamientos()
    red_contagios = load_red_contagios()

    datasets = {
        'pacientes': pacientes,
        'evolucion': evolucion,
        'tratamientos': tratamientos,
        'red_contagios': red_contagios
    }

    print(f"\nTODOS LOS DATASETS CARGADOS EXITOSAMENTE")
    print(f"Total de filas: {sum(len(df) for df in datasets.values()):,}")

    # ==============================================================================
    # PASO 2: VALIDAR DATASETS
    # ==============================================================================

    print_section("PASO 2: VALIDACION DE DATASETS")

    validations = []
    validations.append(validate_pacientes(pacientes))
    validations.append(validate_evolucion(evolucion))
    validations.append(validate_tratamientos(tratamientos))
    validations.append(validate_red_contagios(red_contagios))

    # ==============================================================================
    # PASO 3: EXPORTAR A CSV
    # ==============================================================================

    print_section("PASO 3: EXPORTACION A CSV")

    export_to_csv(pacientes, 'pacientes')
    export_to_csv(evolucion, 'evolucion')
    export_to_csv(tratamientos, 'tratamientos')
    export_to_csv(red_contagios, 'red_contagios')

    print("\nTODOS LOS DATASETS EXPORTADOS A CSV")

    # ==============================================================================
    # PASO 4: GENERAR REPORTE
    # ==============================================================================

    print_section("PASO 4: REPORTE DE CALIDAD DE DATOS")

    report_path = generate_data_quality_report(validations, datasets)

    # ==============================================================================
    # RESUMEN FINAL
    # ==============================================================================

    log_step(
        "SCRIPT 01 COMPLETADO EXITOSAMENTE",
        f"Datasets cargados: {len(datasets)}\n"
        f"Total de filas: {sum(len(df) for df in datasets.values()):,}\n"
        f"Archivos CSV generados: {len(datasets)}\n"
        f"Reporte de calidad: {report_path.name}"
    )

# ==============================================================================
# EJECUCION
# ==============================================================================

if __name__ == "__main__":
    main()
