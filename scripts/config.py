"""
Archivo de configuracion central para el proyecto Operacion Anti-Zombie

Este archivo contiene todas las configuraciones, rutas, constantes y parametros
utilizados a lo largo del pipeline CRISP-DM
"""

import os
from pathlib import Path
import numpy as np

# ==============================================================================
# RUTAS DEL PROYECTO
# ==============================================================================

# Ruta raiz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Rutas de datos
DATA_RAW = PROJECT_ROOT / "data"
DATA_PROCESSED = PROJECT_ROOT / "datos_procesados"

# Rutas de resultados
RESULTS = PROJECT_ROOT / "resultados"
RESULTS_EDA = RESULTS / "eda"
RESULTS_EDA_UNI = RESULTS_EDA / "univariate"
RESULTS_EDA_BI = RESULTS_EDA / "bivariate"
RESULTS_EDA_MULTI = RESULTS_EDA / "multivariate"
RESULTS_EDA_TEMPORAL = RESULTS_EDA / "temporal"
RESULTS_EDA_NETWORK = RESULTS_EDA / "network"

RESULTS_CLASSIFICATION = RESULTS / "classification"
RESULTS_REGRESSION = RESULTS / "regression"
RESULTS_CLUSTERING = RESULTS / "clustering"
RESULTS_REPORTS = RESULTS / "reports"

# Ruta de modelos
MODELS = PROJECT_ROOT / "modelos"

# ==============================================================================
# SEMILLA ALEATORIA PARA REPRODUCIBILIDAD
# ==============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==============================================================================
# ARCHIVOS DE DATOS
# ==============================================================================

DATA_FILES = {
    'pacientes': 'pacientes_brote_zombie.xlsx',
    'evolucion': 'evolucion_brote.xlsx',
    'tratamientos': 'tratamientos_experimentales.xlsx',
    'red_contagios': 'red_contagios.xlsx'
}

# ==============================================================================
# VARIABLES OBJETIVO (TARGETS)
# ==============================================================================

TARGETS = {
    'classification': 'Estado_Actual',
    'regression_primary': 'Nivel_Zombificacion',
    'regression_secondary': 'Tasa_Contagio_R0'
}

# ==============================================================================
# CONJUNTOS DE FEATURES POR TIPO
# ==============================================================================

# Features clinicas
FEATURES_CLINICAL = [
    'Temperatura_Corporal',
    'Presion_Arterial',
    'Frecuencia_Cardiaca',
    'Nivel_Consciencia',
    'Agresividad',
    'Capacidad_Cognitiva'
]

# Features demograficas
FEATURES_DEMOGRAPHIC = [
    'Edad',
    'Sexo',
    'Tipo_Sangre'
]

# Features de exposicion
FEATURES_EXPOSURE = [
    'Dias_Incubacion',
    'Exposicion_Inicial',
    'Tiempo_Exposicion_Minutos',
    'Uso_EPP',
    'Distancia_Paciente_Cero'
]

# Features de tratamiento
FEATURES_TREATMENT = [
    'Tratamiento_Recibido',
    'Dosis_Tratamiento',
    'Dias_Desde_Tratamiento',
    'Mejoria_Porcentual',
    'Respuesta_Tratamiento'
]

# Features de localizacion
FEATURES_LOCATION = [
    'Departamento',
    'Edificio',
    'Piso',
    'Zona_Contagio'
]

# Features de contagio
FEATURES_CONTAGION = [
    'Numero_Personas_Contagiadas',
    'Nivel_Zombificacion',
    'Contagiado_Por'
]

# Features para clasificacion (mismo set usado en notebook)
FEATURES_CLASSIFICATION = [
    'Edad',
    'Dias_Incubacion',
    'Nivel_Zombificacion',
    'Numero_Personas_Contagiadas',
    'Temperatura_Corporal',
    'Frecuencia_Cardiaca',
    'Nivel_Consciencia',
    'Agresividad',
    'Capacidad_Cognitiva',
    'Dias_Desde_Tratamiento',
    'Mejoria_Porcentual'
]

# Features para clustering (mismo set)
FEATURES_CLUSTERING = FEATURES_CLASSIFICATION.copy()

# ==============================================================================
# PARAMETROS DE MODELOS
# ==============================================================================

# Parametros de hiperparametros para tuning
MODEL_PARAMS = {
    # Decision Tree Classifier
    'decision_tree_clf': {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [10, 20, 50],
        'min_samples_leaf': [5, 10, 20],
        'criterion': ['gini', 'entropy']
    },

    # Random Forest Classifier
    'random_forest_clf': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    },

    # XGBoost Classifier
    'xgboost_clf': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },

    # Logistic Regression
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [1000]
    },

    # Decision Tree Regressor
    'decision_tree_reg': {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [10, 20, 50],
        'min_samples_leaf': [5, 10, 20]
    },

    # Random Forest Regressor
    'random_forest_reg': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [10, 20]
    },

    # XGBoost Regressor
    'xgboost_reg': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    },

    # Ridge Regression
    'ridge': {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    },

    # Lasso Regression
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1, 10]
    },

    # Clustering
    'kmeans': {
        'n_clusters': range(2, 11),
        'n_init': 10,
        'max_iter': 300
    },

    'hierarchical': {
        'n_clusters': range(2, 11),
        'linkage': ['ward', 'complete', 'average']
    },

    'dbscan': {
        'eps': [0.3, 0.5, 0.7, 1.0],
        'min_samples': [5, 10, 15]
    }
}

# ==============================================================================
# PARAMETROS DE ENTRENAMIENTO
# ==============================================================================

# Train-test split
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.2

# Validacion cruzada
CV_FOLDS = 5

# Manejo de clases desbalanceadas
CLASS_WEIGHT = 'balanced'

# Early stopping (para XGBoost)
EARLY_STOPPING_ROUNDS = 10

# ==============================================================================
# CONFIGURACION DE VISUALIZACIONES
# ==============================================================================

# Estilo de plots
PLOT_STYLE = 'seaborn-v0_8-whitegrid'

# Tamano de figuras por defecto
FIGURE_SIZE = (12, 8)
FIGURE_SIZE_SMALL = (8, 6)
FIGURE_SIZE_LARGE = (16, 10)

# DPI para guardar imagenes
DPI = 300

# Paleta de colores
COLOR_PALETTE = 'viridis'
COLOR_PALETTE_CATEGORICAL = 'Set2'

# Configuracion de seaborn
SEABORN_CONTEXT = 'notebook'
SEABORN_FONT_SCALE = 1.2

# ==============================================================================
# PARAMETROS DE PROCESAMIENTO DE DATOS
# ==============================================================================

# Umbral de datos faltantes (eliminar features con mas de X% missing)
MISSING_THRESHOLD = 0.5  # 50%

# Umbral de correlacion (eliminar features altamente correlacionadas)
CORRELATION_THRESHOLD = 0.95

# Umbral de varianza (eliminar features de baja varianza)
VARIANCE_THRESHOLD = 0.01

# Estrategia de imputacion
IMPUTATION_STRATEGY_NUM = 'median'  # median, mean
IMPUTATION_STRATEGY_CAT = 'most_frequent'  # most_frequent, constant

# Metodo de escalado
SCALER_METHOD = 'standard'  # standard, robust, minmax

# ==============================================================================
# CONFIGURACION DE LOGGING
# ==============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ==============================================================================
# CONFIGURACION DE ANALISIS DE REDES
# ==============================================================================

# Algoritmo de deteccion de comunidades
COMMUNITY_ALGORITHM = 'louvain'  # louvain, girvan_newman, label_propagation

# Layout para visualizacion de grafos
GRAPH_LAYOUT = 'spring'  # spring, kamada_kawai, circular

# Tamano de nodos en grafos
NODE_SIZE_MIN = 100
NODE_SIZE_MAX = 1000

# ==============================================================================
# ETIQUETAS Y NOMBRES PARA VISUALIZACIONES
# ==============================================================================

# Nombres amigables para Estado_Actual
ESTADO_LABELS = {
    'Sano': 'Sano',
    'Infectado_Leve': 'Infectado Leve',
    'Infectado_Moderado': 'Infectado Moderado',
    'Infectado_Grave': 'Infectado Grave',
    'Zombificado': 'Zombificado',
    'Recuperado': 'Recuperado',
    'Fallecido': 'Fallecido'
}

# Colores para cada estado
ESTADO_COLORS = {
    'Sano': '#2ecc71',
    'Infectado_Leve': '#f39c12',
    'Infectado_Moderado': '#e67e22',
    'Infectado_Grave': '#e74c3c',
    'Zombificado': '#8e44ad',
    'Recuperado': '#3498db',
    'Fallecido': '#34495e'
}

# ==============================================================================
# MENSAJES Y TEXTO
# ==============================================================================

BANNER = """
===============================================================================
                    OPERACION ANTI-ZOMBIE
         Pipeline CRISP-DM para Analisis de Datos y Machine Learning
===============================================================================
"""

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def get_data_path(dataset_name):
    """
    Obtener ruta completa de un dataset

    Args:
        dataset_name: Nombre del dataset ('pacientes', 'evolucion', etc.)

    Returns:
        Path completo al archivo
    """
    if dataset_name not in DATA_FILES:
        raise ValueError(f"Dataset desconocido: {dataset_name}. Opciones: {list(DATA_FILES.keys())}")

    return DATA_RAW / DATA_FILES[dataset_name]

def get_processed_path(filename):
    """
    Obtener ruta completa para datos procesados

    Args:
        filename: Nombre del archivo

    Returns:
        Path completo en datos_procesados/
    """
    return DATA_PROCESSED / filename

def get_results_path(subdir, filename):
    """
    Obtener ruta completa para resultados

    Args:
        subdir: Subdirectorio (ej: 'eda/univariate', 'classification/models')
        filename: Nombre del archivo

    Returns:
        Path completo en resultados/
    """
    return RESULTS / subdir / filename

def get_model_path(filename):
    """
    Obtener ruta completa para modelos

    Args:
        filename: Nombre del archivo del modelo

    Returns:
        Path completo en modelos/
    """
    return MODELS / filename

# ==============================================================================
# VALIDACION DE CONFIGURACION
# ==============================================================================

def validate_config():
    """Validar que las rutas y archivos de datos existan"""

    # Verificar que data/ exista
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Directorio de datos no encontrado: {DATA_RAW}")

    # Verificar que los archivos de datos existan
    missing_files = []
    for name, filename in DATA_FILES.items():
        filepath = DATA_RAW / filename
        if not filepath.exists():
            missing_files.append(f"{name}: {filepath}")

    if missing_files:
        print("ADVERTENCIA: Archivos de datos faltantes:")
        for mf in missing_files:
            print(f"  - {mf}")
    else:
        print("Todos los archivos de datos encontrados correctamente")

    return len(missing_files) == 0

# ==============================================================================
# INICIALIZACION
# ==============================================================================

if __name__ == "__main__":
    print(BANNER)
    print(f"Directorio raiz del proyecto: {PROJECT_ROOT}")
    print(f"Directorio de datos: {DATA_RAW}")
    print(f"Directorio de resultados: {RESULTS}")
    print(f"Directorio de modelos: {MODELS}")
    print()
    validate_config()
