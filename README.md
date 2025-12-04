# Operacion Anti-Zombie: Analisis BI de Supervivencia Comercial

Pipeline completo de Machine Learning usando metodologia CRISP-DM para analizar un brote de zombificacion de clientes. Proyecto academico de Inteligencia de Negocio (Evaluacion 3).

## Descripcion del Proyecto

Este proyecto utiliza tecnicas de ciencia de datos y machine learning para analizar un escenario ficticio de "brote zombie" como metafora de clientes en riesgo de abandono o deterioro. Se aplica la metodologia CRISP-DM completa con analisis exploratorio exhaustivo y multiples modelos predictivos.

### Objetivos

- Identificar clientes/pacientes en riesgo de zombificacion (churn severo)
- Predecir niveles de gravedad y tasas de contagio
- Segmentar poblacion en perfiles de riesgo homogeneos
- Optimizar estrategias de intervencion y tratamiento
- Generar insights accionables para toma de decisiones

## Estructura del Proyecto

```
operacion_antizombie/
├── data/                           # Datos originales (Excel)
│   ├── pacientes_brote_zombie.xlsx (700 pacientes, 35 columnas)
│   ├── evolucion_brote.xlsx       (75 dias, 14 columnas)
│   ├── tratamientos_experimentales.xlsx (9 tratamientos, 12 columnas)
│   └── red_contagios.xlsx         (850 contagios, 8 columnas)
│
├── datos_procesados/               # Datos limpios y procesados
│   ├── *.csv                      # Versiones CSV de los datos
│   ├── features_engineered.csv    # Features creadas
│   └── X_train.csv, X_test.csv    # Train/test splits
│
├── scripts/                        # Scripts modulares del pipeline
│   ├── config.py                  # Configuracion central
│   ├── utils.py                   # Funciones utilitarias
│   ├── 01_data_loading.py         # Carga y validacion
│   ├── 02_eda_univariate.py       # Analisis univariado
│   ├── 03_eda_bivariate.py        # Analisis bivariado
│   ├── 04_eda_multivariate.py     # PCA e interacciones
│   ├── 05_eda_temporal.py         # Series temporales
│   ├── 06_eda_network.py          # Analisis de redes
│   ├── 07_data_cleaning.py        # Limpieza de datos
│   ├── 08_feature_engineering.py  # Ingenieria de features
│   ├── 09_data_preprocessing.py   # Preprocesamiento
│   ├── 10_model_classification.py # Modelos clasificacion
│   ├── 11_model_regression.py     # Modelos regresion
│   ├── 12_model_clustering.py     # Modelos clustering
│   ├── 13_model_network_analysis.py # Analisis de redes
│   ├── 14_model_evaluation.py     # Evaluacion modelos
│   ├── 15_model_comparison.py     # Comparacion modelos
│   └── 16_results_export.py       # Exportar resultados
│
├── resultados/                     # Todos los outputs
│   ├── eda/                       # 65+ visualizaciones EDA
│   │   ├── univariate/           # ~20 graficos
│   │   ├── bivariate/            # ~15 graficos
│   │   ├── multivariate/         # ~10 graficos
│   │   ├── temporal/             # ~10 graficos
│   │   └── network/              # ~10 graficos
│   ├── classification/            # Modelos y resultados
│   ├── regression/                # Modelos y resultados
│   ├── clustering/                # Modelos y resultados
│   └── reports/                   # Reportes finales
│
├── modelos/                        # Mejores modelos entrenados
├── notebook/                       # Notebook original (referencia)
├── main.py                         # Orquestador principal
├── requirements.txt                # Dependencias Python
├── README.md                       # Este archivo
└── .gitignore                      # Archivos a ignorar en Git
```

## Datasets

### 1. Pacientes Brote Zombie (700 registros, 35 columnas)
Registro principal de pacientes con informacion demografica, clinica, de exposicion y tratamiento.

**Columnas principales:**
- **Identificacion:** ID_Paciente, Nombre_Completo, Edad, Sexo, Tipo_Sangre
- **Estado:** Estado_Actual (7 clases: Sano, Infectado Leve/Moderado/Grave, Zombificado, Recuperado, Fallecido)
- **Clinicas:** Temperatura_Corporal, Presion_Arterial, Frecuencia_Cardiaca, Nivel_Consciencia, Agresividad, Capacidad_Cognitiva
- **Contagio:** Nivel_Zombificacion (0-100), Numero_Personas_Contagiadas, Distancia_Paciente_Cero
- **Tratamiento:** Tratamiento_Recibido, Mejoria_Porcentual, Respuesta_Tratamiento

### 2. Evolucion Brote (75 dias, 14 columnas)
Serie temporal de la evolucion del brote.

**Metricas:** Casos nuevos/acumulados/activos, Tasa_Contagio_R0, Tasas de mortalidad/recuperacion

### 3. Tratamientos Experimentales (9 tratamientos, 12 columnas)
Catalogo de tratamientos disponibles con eficacia y costos.

### 4. Red Contagios (850 conexiones, 8 columnas)
Grafo dirigido de transmision entre pacientes.

## Metodologia: CRISP-DM

El proyecto sigue estrictamente la metodologia CRISP-DM:

### Fase 1: Business Understanding
- Problema: Identificar y segmentar clientes en riesgo
- Objetivo: Modelos predictivos y estrategias de intervencion

### Fase 2: Data Understanding (EDA Exhaustivo)
- **Analisis Univariado:** Distribuciones, outliers, estadisticas descriptivas (20+ visualizaciones)
- **Analisis Bivariado:** Correlaciones, relaciones con target (15+ visualizaciones)
- **Analisis Multivariado:** PCA, feature importance, interacciones (10+ visualizaciones)
- **Analisis Temporal:** Tendencias, estacionalidad, puntos de cambio (10+ visualizaciones)
- **Analisis de Redes:** Centralidad, comunidades, super-spreaders (10+ visualizaciones)

**Total: 65+ visualizaciones**

### Fase 3: Data Preparation
- Limpieza de datos (valores faltantes, outliers)
- Ingenieria de features (temporales, clinicas, de red, geograficas)
- Codificacion y escalado
- Train-test split estratificado

### Fase 4: Modeling

#### Clasificacion (Predecir Estado_Actual)
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- Logistic Regression

#### Regresion (Predecir Nivel_Zombificacion y Tasa_Contagio_R0)
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Linear Regression (Ridge/Lasso)

#### Clustering (Segmentar pacientes)
- K-Means
- Hierarchical Clustering (Aglomerativo y Divisivo)
- DBSCAN
- Gaussian Mixture Models

#### Analisis de Redes
- Link Prediction
- Community Detection
- Influence Maximization

### Fase 5: Evaluation
- Metricas de clasificacion: Accuracy, F1, Precision, Recall, ROC-AUC
- Metricas de regresion: RMSE, MAE, R²
- Metricas de clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Comparacion exhaustiva de modelos

### Fase 6: Deployment
- Exportacion de mejores modelos
- Reportes ejecutivos y tecnicos
- Visualizaciones clave para presentacion

## Instalacion

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes Python)
- 2GB de espacio en disco
- 4GB RAM recomendado

### Pasos de Instalacion

1. **Clonar o descargar el repositorio**
   ```bash
   cd operacion_antizombie
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalacion**
   ```bash
   python scripts/config.py
   ```

## Uso

### Ejecucion Completa del Pipeline

```bash
# Ejecutar todo el pipeline CRISP-DM
python main.py

# Ejecutar solo fase de EDA
python main.py --phase eda

# Ejecutar solo preparacion de datos
python main.py --phase prep

# Ejecutar solo modelado
python main.py --phase model

# Ejecutar solo evaluacion
python main.py --phase eval
```

### Ejecucion de Scripts Individuales

```bash
# Carga de datos
python scripts/01_data_loading.py

# EDA Univariado
python scripts/02_eda_univariate.py

# Modelos de clasificacion
python scripts/10_model_classification.py

# ... etc
```

### Verificacion de Resultados

Despues de la ejecucion, revisar:
- `resultados/eda/`: Visualizaciones del analisis exploratorio
- `resultados/classification/`: Metricas y modelos de clasificacion
- `resultados/regression/`: Metricas y modelos de regresion
- `resultados/clustering/`: Segmentaciones y perfiles
- `resultados/reports/`: Reportes finales

## Reproducibilidad

- **Semilla aleatoria:** RANDOM_SEED = 42 (configurado en `scripts/config.py`)
- **Train-test split:** 70-30 estratificado
- **Validacion cruzada:** 5-fold
- **Versiones fijas:** requirements.txt especifica versiones exactas

## Tecnologias Utilizadas

- **Python 3.11**
- **Pandas & NumPy:** Manipulacion de datos
- **Matplotlib & Seaborn:** Visualizacion
- **Scikit-learn:** Machine learning
- **XGBoost:** Gradient boosting
- **NetworkX:** Analisis de redes
- **SciPy & Statsmodels:** Analisis estadistico

## Resultados Esperados

### EDA
- 65+ visualizaciones organizadas por tipo de analisis
- Reportes de calidad de datos
- Insights de patrones y correlaciones

### Modelos
- 5 clasificadores entrenados y evaluados
- 10 regresores (5 por cada target)
- 5 algoritmos de clustering
- Analisis de redes completo

### Metricas
- Accuracy > 0.85 en clasificacion (esperado)
- R² > 0.70 en regresion (esperado)
- Silhouette > 0.40 en clustering (esperado)

### Outputs
- Modelos serializados (.pkl)
- Predicciones en test set
- Perfiles de clusters
- Recomendaciones de intervencion

## Autores

**Proyecto Operacion Anti-Zombie**
- Evaluacion 3 - Inteligencia de Negocio
- Institucion: DUOC UC
- Año: 2025

## Contexto Academico

Este proyecto es parte de la Evaluacion 3 del ramo de Inteligencia de Negocio. Utiliza una narrativa creativa ("brote zombie") como marco para aplicar metodologias reales de ciencia de datos a problemas de negocio:

- **Business Intelligence:** Dashboards, KPIs, reportes ejecutivos
- **Machine Learning:** Modelos predictivos, clustering, analisis de redes
- **CRISP-DM:** Metodologia industrial estandar
- **Python:** Stack tecnologico profesional

El objetivo es demostrar competencias en:
- Analisis exploratorio exhaustivo
- Ingenieria de features
- Seleccion y optimizacion de modelos
- Evaluacion rigurosa
- Comunicacion de resultados

## Licencia

Este proyecto es de uso academico. Todos los datos son ficticios y generados para propositos educativos.

## Contacto

Para preguntas o comentarios sobre el proyecto, consultar con el equipo docente del curso.

---










**Operacion Anti-Zombie - Sobrevivir al Data Science**

*"En un mundo infectado de datos, solo los mejor entrenados sobreviven"*
