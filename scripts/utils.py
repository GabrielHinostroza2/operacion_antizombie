"""
Funciones utilitarias para el proyecto Operacion Anti-Zombie

Este modulo contiene funciones reutilizables para:
- Gestion de directorios
- Guardado/carga de archivos
- Visualizaciones
- Logging
- Procesamiento de datos
"""

import os
import sys
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importar configuracion
from config import *

# Suprimir warnings molestos
warnings.filterwarnings('ignore')

# ==============================================================================
# FUNCIONES DE GESTION DE DIRECTORIOS
# ==============================================================================

def setup_directories():
    """
    Crear toda la estructura de directorios del proyecto

    Crea todas las carpetas necesarias para el pipeline CRISP-DM:
    - datos_procesados/
    - resultados/eda/ (univariate, bivariate, multivariate, temporal, network)
    - resultados/classification/ (models, metrics, visualizations, predictions)
    - resultados/regression/ (models, metrics, visualizations, predictions)
    - resultados/clustering/ (models, metrics, visualizations, segments)
    - resultados/reports/
    - modelos/
    """

    directories = [
        # Datos procesados
        DATA_PROCESSED,

        # EDA
        RESULTS_EDA_UNI,
        RESULTS_EDA_BI,
        RESULTS_EDA_MULTI,
        RESULTS_EDA_TEMPORAL,
        RESULTS_EDA_NETWORK,

        # Clasificacion
        RESULTS_CLASSIFICATION / 'models',
        RESULTS_CLASSIFICATION / 'metrics',
        RESULTS_CLASSIFICATION / 'visualizations',
        RESULTS_CLASSIFICATION / 'predictions',

        # Regresion
        RESULTS_REGRESSION / 'models',
        RESULTS_REGRESSION / 'metrics',
        RESULTS_REGRESSION / 'visualizations',
        RESULTS_REGRESSION / 'predictions',

        # Clustering
        RESULTS_CLUSTERING / 'models',
        RESULTS_CLUSTERING / 'metrics',
        RESULTS_CLUSTERING / 'visualizations',
        RESULTS_CLUSTERING / 'segments',

        # Reportes
        RESULTS_REPORTS,

        # Modelos
        MODELS
    ]

    created_count = 0
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_count += 1

    log_step(
        "SETUP DIRECTORIOS",
        f"Estructura de directorios creada: {created_count} nuevos directorios, {len(directories)} total"
    )

    return len(directories)

# ==============================================================================
# FUNCIONES DE GUARDADO Y CARGA
# ==============================================================================

def save_plot(fig, filename, subdir='', dpi=None, close=True):
    """
    Guardar figura matplotlib con configuracion consistente

    Args:
        fig: Figura de matplotlib
        filename: Nombre del archivo (con extension .png)
        subdir: Subdirectorio dentro de resultados/ (ej: 'eda/univariate')
        dpi: DPI para guardar (por defecto usa config.DPI)
        close: Si cerrar la figura despues de guardar

    Returns:
        Path del archivo guardado
    """
    if dpi is None:
        dpi = DPI

    if subdir:
        path = RESULTS / subdir / filename
    else:
        path = RESULTS / filename

    # Asegurar que el directorio exista
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')

    if close:
        plt.close(fig)

    print(f"   Guardado: {path.relative_to(PROJECT_ROOT)}")
    return path

def save_dataframe(df, filename, subdir='', index=False):
    """
    Guardar dataframe a CSV

    Args:
        df: DataFrame de pandas
        filename: Nombre del archivo (con extension .csv)
        subdir: Subdirectorio (si vacio, guarda en datos_procesados/)
        index: Si guardar el indice

    Returns:
        Path del archivo guardado
    """
    if subdir:
        path = RESULTS / subdir / filename
    else:
        path = DATA_PROCESSED / filename

    # Asegurar que el directorio exista
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar
    df.to_csv(path, index=index)

    print(f"   Guardado: {path.relative_to(PROJECT_ROOT)} ({len(df)} filas, {len(df.columns)} columnas)")
    return path

def load_dataframe(filename, subdir=''):
    """
    Cargar dataframe desde CSV

    Args:
        filename: Nombre del archivo
        subdir: Subdirectorio (si vacio, carga de datos_procesados/)

    Returns:
        DataFrame de pandas
    """
    if subdir:
        path = RESULTS / subdir / filename
    else:
        path = DATA_PROCESSED / filename

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    df = pd.read_csv(path)
    print(f"   Cargado: {path.relative_to(PROJECT_ROOT)} ({len(df)} filas, {len(df.columns)} columnas)")
    return df

def save_model(model, filename, subdir='models'):
    """
    Guardar modelo entrenado usando pickle

    Args:
        model: Modelo entrenado (sklearn, etc.)
        filename: Nombre del archivo (con extension .pkl)
        subdir: Subdirectorio dentro de resultados/

    Returns:
        Path del archivo guardado
    """
    if subdir == 'models':
        path = MODELS / filename
    else:
        path = RESULTS / subdir / filename

    # Asegurar que el directorio exista
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar
    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"   Modelo guardado: {path.relative_to(PROJECT_ROOT)}")
    return path

def load_model(filename, subdir='models'):
    """
    Cargar modelo entrenado desde pickle

    Args:
        filename: Nombre del archivo
        subdir: Subdirectorio

    Returns:
        Modelo cargado
    """
    if subdir == 'models':
        path = MODELS / filename
    else:
        path = RESULTS / subdir / filename

    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")

    with open(path, 'rb') as f:
        model = pickle.load(f)

    print(f"   Modelo cargado: {path.relative_to(PROJECT_ROOT)}")
    return model

def save_metrics(metrics_dict, filename, subdir=''):
    """
    Guardar diccionario de metricas como JSON

    Args:
        metrics_dict: Diccionario con metricas
        filename: Nombre del archivo (con extension .json)
        subdir: Subdirectorio dentro de resultados/

    Returns:
        Path del archivo guardado
    """
    if subdir:
        path = RESULTS / subdir / filename
    else:
        path = RESULTS / filename

    # Asegurar que el directorio exista
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"   Metricas guardadas: {path.relative_to(PROJECT_ROOT)}")
    return path

def load_metrics(filename, subdir=''):
    """
    Cargar diccionario de metricas desde JSON

    Args:
        filename: Nombre del archivo
        subdir: Subdirectorio

    Returns:
        Diccionario de metricas
    """
    if subdir:
        path = RESULTS / subdir / filename
    else:
        path = RESULTS / filename

    if not path.exists():
        raise FileNotFoundError(f"Archivo de metricas no encontrado: {path}")

    with open(path, 'r') as f:
        metrics = json.load(f)

    print(f"   Metricas cargadas: {path.relative_to(PROJECT_ROOT)}")
    return metrics

# ==============================================================================
# FUNCIONES DE LOGGING
# ==============================================================================

def log_step(step_name, description=""):
    """
    Log de un paso de ejecucion con timestamp

    Args:
        step_name: Nombre del paso
        description: Descripcion opcional
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 80

    print(f"\n{separator}")
    print(f"[{timestamp}] {step_name}")
    if description:
        print(f"{description}")
    print(f"{separator}\n")

def print_section(title):
    """
    Imprimir titulo de seccion

    Args:
        title: Titulo de la seccion
    """
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def print_subsection(title):
    """
    Imprimir titulo de subseccion

    Args:
        title: Titulo de la subseccion
    """
    print(f"\n{'-'*80}")
    print(f"  {title}")
    print(f"{'-'*80}\n")

# ==============================================================================
# FUNCIONES DE ANALISIS DE DATOS
# ==============================================================================

def detect_outliers_iqr(df, columns, threshold=1.5):
    """
    Detectar outliers usando metodo IQR

    Args:
        df: DataFrame
        columns: Lista de columnas a analizar
        threshold: Multiplicador de IQR (por defecto 1.5)

    Returns:
        DataFrame con valores booleanos (True = outlier)
    """
    outliers = pd.DataFrame(index=df.index)

    for col in columns:
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        else:
            outliers[col] = False

    return outliers

def get_missing_data_summary(df):
    """
    Generar resumen completo de datos faltantes

    Args:
        df: DataFrame

    Returns:
        DataFrame con resumen de missing data
    """
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values,
        'Data_Type': df.dtypes.values,
        'Non_Missing_Count': df.notnull().sum().values
    })

    missing = missing[missing['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    ).reset_index(drop=True)

    return missing

def get_basic_stats(df, columns=None):
    """
    Obtener estadisticas basicas de un DataFrame

    Args:
        df: DataFrame
        columns: Lista de columnas (si None, usa todas las numericas)

    Returns:
        DataFrame con estadisticas
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = df[columns].describe().T
    stats['missing'] = df[columns].isnull().sum()
    stats['missing_pct'] = (df[columns].isnull().sum() / len(df) * 100).round(2)
    stats['unique'] = df[columns].nunique()

    # Agregar asimetria y curtosis
    stats['skewness'] = df[columns].skew()
    stats['kurtosis'] = df[columns].kurtosis()

    return stats

def get_categorical_summary(df, columns=None):
    """
    Obtener resumen de variables categoricas

    Args:
        df: DataFrame
        columns: Lista de columnas categoricas (si None, detecta automaticamente)

    Returns:
        Dict con resumen por columna
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    summary = {}
    for col in columns:
        summary[col] = {
            'unique_values': df[col].nunique(),
            'most_common': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            'most_common_count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df) * 100).round(2),
            'value_counts': df[col].value_counts().to_dict()
        }

    return summary

# ==============================================================================
# FUNCIONES DE VISUALIZACION
# ==============================================================================

def setup_plot_style():
    """
    Configurar estilo consistente para todas las visualizaciones
    """
    # Estilo general
    plt.style.use(PLOT_STYLE)

    # Configuracion de seaborn
    sns.set_context(SEABORN_CONTEXT, font_scale=SEABORN_FONT_SCALE)
    sns.set_palette(COLOR_PALETTE)

    # Configuracion de matplotlib
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 100  # DPI para display
    plt.rcParams['savefig.dpi'] = DPI  # DPI para guardar

    print("Estilo de visualizacion configurado correctamente")

def create_distribution_plot(data, column, title=None, bins=30, kde=True, figsize=None):
    """
    Crear grafico de distribucion con histograma y KDE

    Args:
        data: DataFrame o Serie
        column: Nombre de la columna (si data es DataFrame)
        title: Titulo del grafico
        bins: Numero de bins para histograma
        kde: Si mostrar KDE
        figsize: Tamano de la figura

    Returns:
        Figura de matplotlib
    """
    if figsize is None:
        figsize = FIGURE_SIZE_SMALL

    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data, pd.DataFrame):
        series = data[column]
        if title is None:
            title = f'Distribucion de {column}'
    else:
        series = data
        if title is None:
            title = 'Distribucion'

    # Histograma + KDE
    series.hist(bins=bins, ax=ax, alpha=0.7, edgecolor='black')

    if kde and len(series.dropna()) > 1:
        series_clean = series.dropna()
        series_clean.plot.kde(ax=ax, secondary_y=True, linewidth=2, color='red')
        ax.right_ax.set_ylabel('Densidad KDE')

    ax.set_xlabel(column if isinstance(data, pd.DataFrame) else 'Valor')
    ax.set_ylabel('Frecuencia')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_boxplot(data, column, by=None, title=None, figsize=None):
    """
    Crear boxplot para analisis de outliers

    Args:
        data: DataFrame
        column: Columna a graficar
        by: Columna para agrupar (opcional)
        title: Titulo del grafico
        figsize: Tamano de la figura

    Returns:
        Figura de matplotlib
    """
    if figsize is None:
        figsize = FIGURE_SIZE_SMALL

    fig, ax = plt.subplots(figsize=figsize)

    if by is None:
        data.boxplot(column=column, ax=ax)
        if title is None:
            title = f'Boxplot de {column}'
    else:
        data.boxplot(column=column, by=by, ax=ax)
        if title is None:
            title = f'Boxplot de {column} por {by}'

    ax.set_title(title)
    ax.set_ylabel(column)
    plt.tight_layout()
    return fig

def create_correlation_heatmap(data, columns=None, title=None, figsize=None, annot=True):
    """
    Crear heatmap de correlaciones

    Args:
        data: DataFrame
        columns: Columnas a incluir (si None, usa todas las numericas)
        title: Titulo del grafico
        figsize: Tamano de la figura
        annot: Si anotar valores

    Returns:
        Figura de matplotlib
    """
    if figsize is None:
        figsize = FIGURE_SIZE

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    corr = data[columns].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})

    if title is None:
        title = 'Matriz de Correlacion'
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

# ==============================================================================
# FUNCIONES DE EVALUACION DE MODELOS
# ==============================================================================

def print_classification_metrics(y_true, y_pred, model_name="Modelo"):
    """
    Imprimir metricas de clasificacion

    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name: Nombre del modelo
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    print(f"\n{'='*60}")
    print(f"  METRICAS: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"{'='*60}\n")

def print_regression_metrics(y_true, y_pred, model_name="Modelo"):
    """
    Imprimir metricas de regresion

    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name: Nombre del modelo
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  METRICAS: {model_name}")
    print(f"{'='*60}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"{'='*60}\n")

# ==============================================================================
# FUNCIONES DE TIEMPO
# ==============================================================================

def format_time(seconds):
    """
    Formatear tiempo en segundos a formato legible

    Args:
        seconds: Tiempo en segundos

    Returns:
        String formateado (ej: "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

# ==============================================================================
# FUNCIONES MAIN (para testing)
# ==============================================================================

if __name__ == "__main__":
    print(BANNER)
    print("Modulo de utilidades cargado correctamente\n")

    # Test: setup directories
    log_step("TEST: Setup de directorios")
    num_dirs = setup_directories()
    print(f"Total de directorios creados/verificados: {num_dirs}\n")

    # Test: plot style
    log_step("TEST: Configuracion de estilo de plots")
    setup_plot_style()
