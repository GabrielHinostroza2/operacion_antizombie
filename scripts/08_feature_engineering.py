"""
SCRIPT 08: INGENIERIA DE FEATURES
CRISP-DM Fase 3: Data Preparation

Creacion de nuevas features para mejorar modelos

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np

from config import *
from utils import *

def create_temporal_features(df):
    """Crear features temporales"""
    print_subsection("Creando features temporales")

    # Convertir fechas
    date_cols = [col for col in df.columns if 'fecha' in col.lower()]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Dias desde primer contacto hasta diagnostico
    if 'Fecha_Primer_Contacto' in df.columns and 'Fecha_Diagnostico_Confirmado' in df.columns:
        df['Days_Since_First_Contact'] = (
            df['Fecha_Diagnostico_Confirmado'] - df['Fecha_Primer_Contacto']
        ).dt.days
        print("   Creado: Days_Since_First_Contact")

    # Dias hasta sintomas
    if 'Fecha_Primer_Contacto' in df.columns and 'Fecha_Sintomas_Iniciales' in df.columns:
        df['Days_To_Symptoms'] = (
            df['Fecha_Sintomas_Iniciales'] - df['Fecha_Primer_Contacto']
        ).dt.days
        print("   Creado: Days_To_Symptoms")

    return df

def create_clinical_features(df):
    """Crear features clinicas compuestas"""
    print_subsection("Creando features clinicas")

    # Score de severidad clinica
    if all(col in df.columns for col in ['Nivel_Zombificacion', 'Nivel_Consciencia',
                                          'Agresividad', 'Capacidad_Cognitiva']):
        df['Clinical_Severity_Score'] = (
            0.4 * df['Nivel_Zombificacion'] +
            0.2 * (100 - df['Nivel_Consciencia']) +
            0.2 * df['Agresividad'] +
            0.2 * (100 - df['Capacidad_Cognitiva'])
        )
        print("   Creado: Clinical_Severity_Score")

    # Anormalidad de signos vitales
    if all(col in df.columns for col in ['Temperatura_Corporal', 'Frecuencia_Cardiaca']):
        df['Vital_Abnormality_Score'] = (
            abs(df['Temperatura_Corporal'] - 37) * 2 +
            abs(df['Frecuencia_Cardiaca'] - 70) / 10
        )
        print("   Creado: Vital_Abnormality_Score")

    return df

def create_treatment_features(df, tratamientos_df):
    """Crear features relacionadas con tratamiento"""
    print_subsection("Creando features de tratamiento")

    # Join con info de tratamientos
    if 'Tratamiento_Recibido' in df.columns and 'Nombre_Tratamiento' in tratamientos_df.columns:
        df = df.merge(
            tratamientos_df[['Nombre_Tratamiento', 'Tasa_Exito_Promedio', 'Costo_Produccion']],
            left_on='Tratamiento_Recibido',
            right_on='Nombre_Tratamiento',
            how='left'
        )
        print("   Agregadas: Tasa_Exito_Promedio, Costo_Produccion")

        # Efectividad del tratamiento
        if 'Mejoria_Porcentual' in df.columns and 'Tasa_Exito_Promedio' in df.columns:
            df['Treatment_Effectiveness'] = df['Mejoria_Porcentual'] / (df['Tasa_Exito_Promedio'] + 1)
            print("   Creado: Treatment_Effectiveness")

    return df

def create_categorical_bins(df):
    """Crear variables categoricas por binning"""
    print_subsection("Creando bins categoricos")

    # Grupos de edad
    if 'Edad' in df.columns:
        df['Age_Group'] = pd.cut(
            df['Edad'],
            bins=[0, 30, 50, 70, 100],
            labels=['Joven', 'Adulto', 'Mayor', 'Anciano']
        )
        print("   Creado: Age_Group")

    # Categoria de zombificacion
    if 'Nivel_Zombificacion' in df.columns:
        df['Zombification_Category'] = pd.cut(
            df['Nivel_Zombificacion'],
            bins=[0, 25, 50, 75, 100],
            labels=['Leve', 'Moderado', 'Severo', 'Critico']
        )
        print("   Creado: Zombification_Category")

    # Duracion de exposicion
    if 'Tiempo_Exposicion_Minutos' in df.columns:
        df['Exposure_Duration'] = pd.cut(
            df['Tiempo_Exposicion_Minutos'],
            bins=[0, 15, 30, 60, 999],
            labels=['Breve', 'Corta', 'Media', 'Prolongada']
        )
        print("   Creado: Exposure_Duration")

    return df

def create_risk_features(df):
    """Crear features de riesgo"""
    print_subsection("Creando features de riesgo")

    # Velocidad de zombificacion
    if 'Nivel_Zombificacion' in df.columns and 'Dias_Incubacion' in df.columns:
        df['Zombification_Velocity'] = df['Nivel_Zombificacion'] / (df['Dias_Incubacion'] + 1)
        print("   Creado: Zombification_Velocity")

    # Indice de contagio
    if 'Numero_Personas_Contagiadas' in df.columns and 'Tiempo_Exposicion_Minutos' in df.columns:
        df['Contagion_Index'] = df['Numero_Personas_Contagiadas'] / (df['Tiempo_Exposicion_Minutos'] + 1)
        print("   Creado: Contagion_Index")

    # Factor de riesgo compuesto
    risk_factors = []
    if 'Uso_EPP' in df.columns:
        risk_factors.append((df['Uso_EPP'] == 'No').astype(int) * 20)
    if 'Exposicion_Inicial' in df.columns:
        risk_factors.append((df['Exposicion_Inicial'] == 'Directa').astype(int) * 30)
    if 'Vacunacion_Previa' in df.columns:
        risk_factors.append((df['Vacunacion_Previa'] == 'No').astype(int) * 15)

    if risk_factors:
        df['Composite_Risk_Score'] = sum(risk_factors)
        print("   Creado: Composite_Risk_Score")

    return df

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 08: INGENIERIA DE FEATURES",
             "Creacion de nuevas features")

    setup_directories()

    # Cargar datos limpios
    print_section("CARGANDO DATOS LIMPIOS")
    try:
        pacientes = load_dataframe('pacientes_cleaned.csv')
        tratamientos = load_dataframe('tratamientos_cleaned.csv')
    except FileNotFoundError:
        print("ERROR: Ejecute primero 07_data_cleaning.py")
        return

    initial_cols = len(pacientes.columns)

    # Crear features
    print_section("CREANDO FEATURES")
    pacientes = create_temporal_features(pacientes)
    pacientes = create_clinical_features(pacientes)
    pacientes = create_treatment_features(pacientes, tratamientos)
    pacientes = create_categorical_bins(pacientes)
    pacientes = create_risk_features(pacientes)

    final_cols = len(pacientes.columns)
    new_features = final_cols - initial_cols

    # Guardar
    print_section("GUARDANDO FEATURES ENGINEERED")
    save_dataframe(pacientes, 'features_engineered.csv')

    # Guardar descripcion de features
    feature_desc = []
    feature_desc.append("="*80)
    feature_desc.append("FEATURES CREADAS")
    feature_desc.append("="*80)
    feature_desc.append(f"Total de nuevas features: {new_features}")
    feature_desc.append("")
    feature_desc.append("Categorias:")
    feature_desc.append("- Features temporales")
    feature_desc.append("- Features clinicas compuestas")
    feature_desc.append("- Features de tratamiento")
    feature_desc.append("- Bins categoricos")
    feature_desc.append("- Features de riesgo")
    feature_desc.append("="*80)

    desc_path = DATA_PROCESSED / 'feature_descriptions.txt'
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(feature_desc))

    log_step("SCRIPT 08 COMPLETADO",
             f"Creadas {new_features} nuevas features")

if __name__ == "__main__":
    main()
