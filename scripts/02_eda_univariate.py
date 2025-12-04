"""
SCRIPT 02: ANALISIS EXPLORATORIO UNIVARIADO
CRISP-DM Fase 2: Data Understanding

Este script realiza analisis univariado exhaustivo de todas las variables:
- Distribuciones de variables numericas (histogramas + KDE)
- Box plots para deteccion de outliers
- Estadisticas descriptivas completas
- Analisis de variables categoricas
- Analisis de datos faltantes
- Q-Q plots para normalidad

Genera ~20 visualizaciones

Autor: Proyecto Operacion Anti-Zombie
Fecha: 2025
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import *
from utils import *

# ==============================================================================
# FUNCIONES DE ANALISIS UNIVARIADO
# ==============================================================================

def analyze_numerical_distributions(df, dataset_name='Dataset'):
    """
    Analizar distribuciones de variables numericas

    Args:
        df: DataFrame
        dataset_name: Nombre del dataset

    Returns:
        Dict con estadisticas
    """
    print_subsection(f"Analizando distribuciones numericas: {dataset_name}")

    # Seleccionar columnas numericas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir ID si existe
    numeric_cols = [col for col in numeric_cols if 'ID' not in col.upper()]

    print(f"Variables numericas a analizar: {len(numeric_cols)}")

    stats_dict = {}

    for col in numeric_cols:
        # Crear figura con 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Histograma + KDE
        data_clean = df[col].dropna()

        axes[0].hist(data_clean, bins=30, alpha=0.7, edgecolor='black', density=True)
        if len(data_clean) > 1:
            data_clean.plot.kde(ax=axes[0], linewidth=2, color='red')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Densidad')
        axes[0].set_title(f'Distribucion de {col}')
        axes[0].grid(True, alpha=0.3)

        # 2. Box plot
        axes[1].boxplot(data_clean, vert=True)
        axes[1].set_ylabel(col)
        axes[1].set_title(f'Box Plot de {col}')
        axes[1].grid(True, alpha=0.3)

        # 3. Q-Q plot
        stats.probplot(data_clean, dist="norm", plot=axes[2])
        axes[2].set_title(f'Q-Q Plot de {col}')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'{dataset_name} - {col}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Guardar
        filename = f'distribucion_{col.lower()}.png'
        save_plot(fig, filename, subdir='eda/univariate')

        # Calcular estadisticas
        stats_dict[col] = {
            'count': len(data_clean),
            'mean': data_clean.mean(),
            'median': data_clean.median(),
            'std': data_clean.std(),
            'min': data_clean.min(),
            'max': data_clean.max(),
            'skewness': data_clean.skew(),
            'kurtosis': data_clean.kurtosis(),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100
        }

    print(f"   Generadas {len(numeric_cols)} visualizaciones de distribuciones")

    return stats_dict

def analyze_categorical_distributions(df, dataset_name='Dataset'):
    """
    Analizar distribuciones de variables categoricas

    Args:
        df: DataFrame
        dataset_name: Nombre del dataset
    """
    print_subsection(f"Analizando distribuciones categoricas: {dataset_name}")

    # Seleccionar columnas categoricas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Excluir columnas con demasiados valores unicos (ej: nombres)
    categorical_cols = [col for col in categorical_cols
                       if df[col].nunique() < 50 and 'nombre' not in col.lower()]

    print(f"Variables categoricas a analizar: {len(categorical_cols)}")

    for col in categorical_cols:
        # Contar valores
        value_counts = df[col].value_counts()

        # Crear figura con 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Grafico de barras
        value_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title(f'Distribucion de {col}')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)

        # 2. Grafico de pastel (si no hay demasiadas categorias)
        if len(value_counts) <= 10:
            axes[1].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                       startangle=90)
            axes[1].set_title(f'Proporcion de {col}')
        else:
            # Si hay muchas categorias, mostrar top 10
            top_10 = value_counts.head(10)
            axes[1].barh(range(len(top_10)), top_10.values)
            axes[1].set_yticks(range(len(top_10)))
            axes[1].set_yticklabels(top_10.index)
            axes[1].set_xlabel('Frecuencia')
            axes[1].set_title(f'Top 10 valores de {col}')
            axes[1].grid(True, alpha=0.3, axis='x')

        plt.suptitle(f'{dataset_name} - {col}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Guardar
        filename = f'distribucion_{col.lower()}.png'
        save_plot(fig, filename, subdir='eda/univariate')

    print(f"   Generadas {len(categorical_cols)} visualizaciones de categoricas")

def analyze_missing_data(df, dataset_name='Dataset'):
    """
    Analizar datos faltantes

    Args:
        df: DataFrame
        dataset_name: Nombre del dataset
    """
    print_subsection(f"Analizando datos faltantes: {dataset_name}")

    # Obtener resumen de missing data
    missing_summary = get_missing_data_summary(df)

    if len(missing_summary) == 0:
        print("   No hay datos faltantes en este dataset")
        return

    print(f"   Columnas con datos faltantes: {len(missing_summary)}")

    # Crear visualizacion
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Heatmap de valores faltantes
    # Tomar una muestra si el dataset es muy grande
    sample_size = min(500, len(df))
    df_sample = df.sample(n=sample_size, random_state=RANDOM_SEED)

    missing_matrix = df_sample.isnull().astype(int)
    cols_with_missing = missing_summary['Column'].tolist()

    if len(cols_with_missing) > 0:
        sns.heatmap(missing_matrix[cols_with_missing].T,
                   cmap='viridis', cbar=True, ax=axes[0],
                   yticklabels=cols_with_missing)
        axes[0].set_title(f'Patron de Valores Faltantes\n(muestra de {sample_size} registros)')
        axes[0].set_xlabel('Registros')
        axes[0].set_ylabel('Variables')

    # 2. Grafico de barras de porcentaje faltante
    axes[1].barh(range(len(missing_summary)),
                missing_summary['Missing_Percentage'].values,
                color='coral', edgecolor='black')
    axes[1].set_yticks(range(len(missing_summary)))
    axes[1].set_yticklabels(missing_summary['Column'].values)
    axes[1].set_xlabel('Porcentaje Faltante (%)')
    axes[1].set_title('Porcentaje de Valores Faltantes por Variable')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'{dataset_name} - Analisis de Datos Faltantes',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Guardar
    filename = f'missing_data_analysis.png'
    save_plot(fig, filename, subdir='eda/univariate')

    # Guardar tabla de resumen
    save_dataframe(missing_summary, f'missing_data_summary_{dataset_name.lower()}.csv',
                  subdir='eda/univariate')

def create_summary_statistics_table(df, dataset_name='Dataset'):
    """
    Crear tabla de estadisticas descriptivas completa

    Args:
        df: DataFrame
        dataset_name: Nombre del dataset
    """
    print_subsection(f"Generando tabla de estadisticas: {dataset_name}")

    # Estadisticas de variables numericas
    stats = get_basic_stats(df)

    # Guardar
    filename = f'estadisticas_descriptivas_{dataset_name.lower()}.csv'
    save_dataframe(stats, filename, subdir='eda/univariate')

    print(f"   Tabla guardada: {filename}")

    # Crear visualizacion de la tabla (top 15 variables)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Seleccionar columnas clave para mostrar
    display_cols = ['count', 'mean', 'std', 'min', 'max', 'missing_pct', 'skewness']
    stats_display = stats[display_cols].head(15)

    # Formatear valores
    stats_display = stats_display.round(2)

    # Crear tabla
    table = ax.table(cellText=stats_display.values,
                    colLabels=stats_display.columns,
                    rowLabels=stats_display.index,
                    cellLoc='right',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Estilo
    for i in range(1, len(display_cols) + 1):
        table[(0, i-1)].set_facecolor('#40466e')
        table[(0, i-1)].set_text_props(weight='bold', color='white')

    plt.title(f'{dataset_name} - Estadisticas Descriptivas (Top 15 variables)',
             fontsize=14, fontweight='bold', pad=20)

    # Guardar
    filename = f'tabla_estadisticas_{dataset_name.lower()}.png'
    save_plot(fig, filename, subdir='eda/univariate')

# ==============================================================================
# ANALISIS POR DATASET
# ==============================================================================

def analyze_pacientes(df):
    """
    Analisis univariado del dataset de pacientes

    Args:
        df: DataFrame de pacientes
    """
    print_section("ANALISIS UNIVARIADO: PACIENTES")

    # Distribuciones numericas
    stats = analyze_numerical_distributions(df, 'Pacientes')

    # Distribuciones categoricas
    analyze_categorical_distributions(df, 'Pacientes')

    # Datos faltantes
    analyze_missing_data(df, 'Pacientes')

    # Tabla de estadisticas
    create_summary_statistics_table(df, 'Pacientes')

    # Analisis especifico: Balance de clases del target
    if 'Estado_Actual' in df.columns:
        print_subsection("Analisis de Balance de Clases (Target)")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Conteo
        estado_counts = df['Estado_Actual'].value_counts()

        # Grafico de barras
        estado_counts.plot(kind='bar', ax=axes[0], colormap=COLOR_PALETTE_CATEGORICAL,
                          edgecolor='black')
        axes[0].set_xlabel('Estado Actual')
        axes[0].set_ylabel('Cantidad de Pacientes')
        axes[0].set_title('Distribucion de Estados (Target de Clasificacion)')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)

        # Agregar valores en las barras
        for i, v in enumerate(estado_counts.values):
            axes[0].text(i, v + max(estado_counts.values)*0.01, str(v),
                        ha='center', va='bottom', fontweight='bold')

        # Grafico de pastel
        colors = [ESTADO_COLORS.get(estado, '#999999') for estado in estado_counts.index]
        axes[1].pie(estado_counts, labels=estado_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors)
        axes[1].set_title('Proporcion de Estados')

        plt.suptitle('Balance de Clases - Estado Actual', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_plot(fig, 'balance_clases_estado_actual.png', subdir='eda/univariate')

        # Calcular balance
        total = len(df)
        print(f"\n   Total de pacientes: {total}")
        print(f"   Distribucion de clases:")
        for estado, count in estado_counts.items():
            pct = (count / total) * 100
            print(f"      {estado}: {count} ({pct:.1f}%)")

    return stats

def analyze_evolucion(df):
    """
    Analisis univariado del dataset de evolucion

    Args:
        df: DataFrame de evolucion
    """
    print_section("ANALISIS UNIVARIADO: EVOLUCION BROTE")

    # Distribuciones numericas
    stats = analyze_numerical_distributions(df, 'Evolucion')

    # Distribuciones categoricas (si las hay)
    analyze_categorical_distributions(df, 'Evolucion')

    # Datos faltantes
    analyze_missing_data(df, 'Evolucion')

    # Tabla de estadisticas
    create_summary_statistics_table(df, 'Evolucion')

    return stats

def analyze_tratamientos(df):
    """
    Analisis univariado del dataset de tratamientos

    Args:
        df: DataFrame de tratamientos
    """
    print_section("ANALISIS UNIVARIADO: TRATAMIENTOS")

    # Distribuciones numericas
    stats = analyze_numerical_distributions(df, 'Tratamientos')

    # Distribuciones categoricas
    analyze_categorical_distributions(df, 'Tratamientos')

    # Datos faltantes
    analyze_missing_data(df, 'Tratamientos')

    # Tabla de estadisticas
    create_summary_statistics_table(df, 'Tratamientos')

    return stats

def analyze_red_contagios(df):
    """
    Analisis univariado del dataset de red de contagios

    Args:
        df: DataFrame de red de contagios
    """
    print_section("ANALISIS UNIVARIADO: RED DE CONTAGIOS")

    # Distribuciones numericas
    stats = analyze_numerical_distributions(df, 'RedContagios')

    # Distribuciones categoricas
    analyze_categorical_distributions(df, 'RedContagios')

    # Datos faltantes
    analyze_missing_data(df, 'RedContagios')

    # Tabla de estadisticas
    create_summary_statistics_table(df, 'RedContagios')

    return stats

# ==============================================================================
# FUNCION PRINCIPAL
# ==============================================================================

def main():
    """
    Funcion principal del script
    """
    print(BANNER)
    log_step(
        "SCRIPT 02: ANALISIS EXPLORATORIO UNIVARIADO",
        "CRISP-DM Fase 2: Data Understanding\n"
        "Analizando distribuciones de todas las variables"
    )

    # Configurar estilo de plots
    setup_plot_style()

    # Asegurar que existan los directorios
    setup_directories()

    # ==============================================================================
    # PASO 1: CARGAR DATOS
    # ==============================================================================

    print_section("PASO 1: CARGA DE DATOS PROCESADOS")

    try:
        pacientes = load_dataframe('pacientes_clean.csv')
        evolucion = load_dataframe('evolucion_clean.csv')
        tratamientos = load_dataframe('tratamientos_clean.csv')
        red_contagios = load_dataframe('red_contagios_clean.csv')
    except FileNotFoundError as e:
        print(f"\nERROR: Archivos CSV no encontrados.")
        print(f"Por favor ejecute primero: python scripts/01_data_loading.py")
        return

    # ==============================================================================
    # PASO 2: ANALISIS UNIVARIADO POR DATASET
    # ==============================================================================

    all_stats = {}

    # Pacientes
    all_stats['pacientes'] = analyze_pacientes(pacientes)

    # Evolucion
    all_stats['evolucion'] = analyze_evolucion(evolucion)

    # Tratamientos
    all_stats['tratamientos'] = analyze_tratamientos(tratamientos)

    # Red de contagios
    all_stats['red_contagios'] = analyze_red_contagios(red_contagios)

    # ==============================================================================
    # PASO 3: RESUMEN FINAL
    # ==============================================================================

    print_section("RESUMEN DEL ANALISIS UNIVARIADO")

    total_visualizations = 0
    for dataset_name, stats in all_stats.items():
        num_vars = len(stats) if stats else 0
        print(f"{dataset_name.capitalize()}: {num_vars} variables numericas analizadas")
        total_visualizations += num_vars

    # Contar variables categoricas aproximadamente
    cat_count = (
        len(pacientes.select_dtypes(include=['object', 'category']).columns) +
        len(evolucion.select_dtypes(include=['object', 'category']).columns) +
        len(tratamientos.select_dtypes(include=['object', 'category']).columns) +
        len(red_contagios.select_dtypes(include=['object', 'category']).columns)
    )

    total_visualizations += min(cat_count, 20)  # Estimacion

    log_step(
        "SCRIPT 02 COMPLETADO EXITOSAMENTE",
        f"Visualizaciones generadas: ~{total_visualizations}\n"
        f"Ubicacion: resultados/eda/univariate/\n"
        f"Estadisticas guardadas en CSV"
    )

# ==============================================================================
# EJECUCION
# ==============================================================================

if __name__ == "__main__":
    main()
