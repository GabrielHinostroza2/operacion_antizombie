"""
SCRIPT 03: ANALISIS EXPLORATORIO BIVARIADO
CRISP-DM Fase 2: Data Understanding

Este script realiza analisis bivariado:
- Matriz de correlaciones completa
- Relaciones entre features y targets
- Analisis de efectividad de tratamientos
- Tablas cruzadas de variables categoricas
- Scatter plots de relaciones clave

Genera ~15 visualizaciones

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
from scipy.stats import chi2_contingency

from config import *
from utils import *

# ==============================================================================
# ANALISIS DE CORRELACIONES
# ==============================================================================

def analyze_correlations(df, dataset_name='Dataset'):
    """Analizar correlaciones entre variables numericas"""
    print_subsection(f"Analizando correlaciones: {dataset_name}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if 'ID' not in col.upper()]

    if len(numeric_cols) < 2:
        print("   Insuficientes variables numericas para analisis de correlacion")
        return

    # Calcular matriz de correlacion
    corr_matrix = df[numeric_cols].corr()

    # 1. Heatmap completo
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f'{dataset_name} - Matriz de Correlacion Completa',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, f'correlacion_matriz_{dataset_name.lower()}.png',
              subdir='eda/bivariate')

    # 2. Top correlaciones
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)

    # Guardar top correlaciones
    save_dataframe(corr_df.head(20),
                   f'top_correlaciones_{dataset_name.lower()}.csv',
                   subdir='eda/bivariate')

    print(f"   Top 10 correlaciones positivas:")
    for _, row in corr_df.head(10).iterrows():
        print(f"      {row['var1']} <-> {row['var2']}: {row['correlation']:.3f}")

def analyze_target_relationships(df, target_col, dataset_name='Dataset'):
    """Analizar relaciones entre features y target"""
    print_subsection(f"Analizando relaciones con target: {target_col}")

    if target_col not in df.columns:
        print(f"   Target {target_col} no encontrado")
        return

    # Variables numericas vs target categorico
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols
                   if col != target_col and 'ID' not in col.upper()]

    # Crear grid de box plots
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, col in enumerate(numeric_cols[:15]):  # Limitar a 15
            if idx < len(axes):
                df.boxplot(column=col, by=target_col, ax=axes[idx])
                axes[idx].set_title(f'{col} por {target_col}')
                axes[idx].set_xlabel('')
                axes[idx].tick_params(axis='x', rotation=45)
                plt.sca(axes[idx])
                plt.xticks(rotation=45, ha='right')

        # Ocultar axes no usados
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'{dataset_name} - Features Numericas vs {target_col}',
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        save_plot(fig, f'features_vs_{target_col.lower()}.png',
                 subdir='eda/bivariate')

def analyze_treatment_effectiveness(pacientes, tratamientos):
    """Analizar efectividad de tratamientos"""
    print_subsection("Analizando efectividad de tratamientos")

    if 'Tratamiento_Recibido' not in pacientes.columns:
        return

    # 1. Mejoria por tratamiento
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Box plot de mejoria por tratamiento
    pacientes.boxplot(column='Mejoria_Porcentual', by='Tratamiento_Recibido',
                     ax=axes[0, 0])
    axes[0, 0].set_title('Mejoria Porcentual por Tratamiento')
    axes[0, 0].set_xlabel('Tratamiento')
    axes[0, 0].set_ylabel('Mejoria (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Nivel de zombificacion por tratamiento
    pacientes.boxplot(column='Nivel_Zombificacion', by='Tratamiento_Recibido',
                     ax=axes[0, 1])
    axes[0, 1].set_title('Nivel Zombificacion por Tratamiento')
    axes[0, 1].set_xlabel('Tratamiento')
    axes[0, 1].set_ylabel('Nivel Zombificacion')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Respuesta al tratamiento
    if 'Respuesta_Tratamiento' in pacientes.columns:
        ct = pd.crosstab(pacientes['Tratamiento_Recibido'],
                        pacientes['Respuesta_Tratamiento'])
        ct.plot(kind='bar', stacked=True, ax=axes[1, 0])
        axes[1, 0].set_title('Respuesta al Tratamiento')
        axes[1, 0].set_xlabel('Tratamiento')
        axes[1, 0].set_ylabel('Cantidad de Pacientes')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Respuesta')

    # Tasa de exito esperada vs real (si hay join con tratamientos)
    if 'Tasa_Exito_Promedio' in pacientes.columns:
        mejoria_promedio = pacientes.groupby('Tratamiento_Recibido')['Mejoria_Porcentual'].mean()

        axes[1, 1].scatter(range(len(mejoria_promedio)), mejoria_promedio.values,
                         label='Mejoria Real', s=100)
        axes[1, 1].set_xticks(range(len(mejoria_promedio)))
        axes[1, 1].set_xticklabels(mejoria_promedio.index, rotation=45)
        axes[1, 1].set_xlabel('Tratamiento')
        axes[1, 1].set_ylabel('Mejoria Promedio (%)')
        axes[1, 1].set_title('Efectividad Observada por Tratamiento')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

    plt.suptitle('Analisis de Efectividad de Tratamientos',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'efectividad_tratamientos.png', subdir='eda/bivariate')

def analyze_categorical_relationships(df, dataset_name='Dataset'):
    """Analizar relaciones entre variables categoricas"""
    print_subsection(f"Analizando relaciones categoricas: {dataset_name}")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [col for col in cat_cols if df[col].nunique() < 20]

    if len(cat_cols) < 2:
        return

    # Analizar pares importantes
    important_pairs = []

    if 'Estado_Actual' in cat_cols and 'Uso_EPP' in cat_cols:
        important_pairs.append(('Estado_Actual', 'Uso_EPP'))
    if 'Estado_Actual' in cat_cols and 'Tipo_Sangre' in cat_cols:
        important_pairs.append(('Estado_Actual', 'Tipo_Sangre'))

    for var1, var2 in important_pairs[:5]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Tabla cruzada
        ct = pd.crosstab(df[var1], df[var2])

        # Heatmap
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title(f'Tabla Cruzada: {var1} vs {var2}')

        # Barras agrupadas
        ct.plot(kind='bar', ax=axes[1])
        axes[1].set_title(f'Distribucion: {var1} por {var2}')
        axes[1].set_xlabel(var1)
        axes[1].set_ylabel('Cantidad')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title=var2)

        plt.tight_layout()
        save_plot(fig, f'crosstab_{var1}_{var2}.png'.lower(),
                 subdir='eda/bivariate')

def analyze_exposure_risk(df):
    """Analizar relacion entre exposicion y gravedad"""
    print_subsection("Analizando exposicion vs gravedad de infeccion")

    if 'Tiempo_Exposicion_Minutos' not in df.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Exposicion vs Nivel Zombificacion
    axes[0, 0].scatter(df['Tiempo_Exposicion_Minutos'],
                       df['Nivel_Zombificacion'], alpha=0.5)
    axes[0, 0].set_xlabel('Tiempo Exposicion (min)')
    axes[0, 0].set_ylabel('Nivel Zombificacion')
    axes[0, 0].set_title('Exposicion vs Nivel Zombificacion')
    axes[0, 0].grid(True, alpha=0.3)

    # Dias incubacion vs Nivel Zombificacion
    if 'Dias_Incubacion' in df.columns:
        axes[0, 1].scatter(df['Dias_Incubacion'],
                          df['Nivel_Zombificacion'], alpha=0.5, color='coral')
        axes[0, 1].set_xlabel('Dias Incubacion')
        axes[0, 1].set_ylabel('Nivel Zombificacion')
        axes[0, 1].set_title('Incubacion vs Nivel Zombificacion')
        axes[0, 1].grid(True, alpha=0.3)

    # Distancia a paciente cero vs Nivel
    if 'Distancia_Paciente_Cero' in df.columns:
        axes[1, 0].scatter(df['Distancia_Paciente_Cero'],
                          df['Nivel_Zombificacion'], alpha=0.5, color='green')
        axes[1, 0].set_xlabel('Distancia a Paciente Cero')
        axes[1, 0].set_ylabel('Nivel Zombificacion')
        axes[1, 0].set_title('Distancia vs Nivel Zombificacion')
        axes[1, 0].grid(True, alpha=0.3)

    # Edad vs Nivel
    if 'Edad' in df.columns:
        axes[1, 1].scatter(df['Edad'], df['Nivel_Zombificacion'],
                          alpha=0.5, color='purple')
        axes[1, 1].set_xlabel('Edad')
        axes[1, 1].set_ylabel('Nivel Zombificacion')
        axes[1, 1].set_title('Edad vs Nivel Zombificacion')
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Analisis de Factores de Riesgo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'factores_riesgo_exposicion.png', subdir='eda/bivariate')

# ==============================================================================
# FUNCION PRINCIPAL
# ==============================================================================

def main():
    """Funcion principal del script"""
    print(BANNER)
    log_step(
        "SCRIPT 03: ANALISIS EXPLORATORIO BIVARIADO",
        "CRISP-DM Fase 2: Data Understanding\n"
        "Analizando relaciones entre variables"
    )

    setup_plot_style()
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

    # Analisis de correlaciones
    print_section("ANALISIS DE CORRELACIONES")
    analyze_correlations(pacientes, 'Pacientes')
    analyze_correlations(evolucion, 'Evolucion')

    # Relaciones con target
    print_section("RELACIONES CON TARGET")
    analyze_target_relationships(pacientes, 'Estado_Actual', 'Pacientes')

    # Efectividad de tratamientos
    print_section("EFECTIVIDAD DE TRATAMIENTOS")
    analyze_treatment_effectiveness(pacientes, tratamientos)

    # Relaciones categoricas
    print_section("RELACIONES CATEGORICAS")
    analyze_categorical_relationships(pacientes, 'Pacientes')

    # Exposicion y riesgo
    print_section("ANALISIS DE EXPOSICION Y RIESGO")
    analyze_exposure_risk(pacientes)

    log_step(
        "SCRIPT 03 COMPLETADO EXITOSAMENTE",
        "Visualizaciones en: resultados/eda/bivariate/"
    )

if __name__ == "__main__":
    main()
