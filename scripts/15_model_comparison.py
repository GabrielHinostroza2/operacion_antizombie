"""
SCRIPT 15: COMPARACION DE MODELOS
CRISP-DM Fase 5: Evaluation

Comparativa de rendimiento entre modelos y seleccion del mejor

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from utils import *

def compare_classification_models():
    """Comparar modelos de clasificacion"""
    print_subsection("Comparando Modelos de Clasificacion")

    try:
        df = load_dataframe('classification_results.csv', 'classification/metrics')
    except FileNotFoundError:
        print("   No se encontraron resultados de clasificacion")
        return None

    # Visualizacion Comparativa
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=df, x='model', y='accuracy', ax=axes[0], palette='viridis')
    axes[0].set_title('Accuracy por Modelo')
    axes[0].set_ylim(0, 1.1)
    
    for i, v in enumerate(df['accuracy']):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center')

    sns.barplot(data=df, x='model', y='f1_score', ax=axes[1], palette='viridis')
    axes[1].set_title('F1 Score (Weighted) por Modelo')
    axes[1].set_ylim(0, 1.1)
    
    for i, v in enumerate(df['f1_score']):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')

    plt.tight_layout()
    save_plot(fig, 'model_comparison_classification.png', 'reports')

    # Seleccionar mejor modelo
    best_model = df.loc[df['f1_score'].idxmax()]
    print(f"   Mejor modelo (Clasificacion): {best_model['model']} (F1: {best_model['f1_score']:.4f})")
    
    return best_model

def compare_regression_models():
    """Comparar modelos de regresion"""
    print_subsection("Comparando Modelos de Regresion")

    try:
        df = load_dataframe('regression_results.csv', 'regression/metrics')
    except FileNotFoundError:
        print("   No se encontraron resultados de regresion")
        return None

    # Visualizacion Comparativa
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=df, x='model', y='rmse', ax=axes[0], palette='magma')
    axes[0].set_title('RMSE por Modelo (Menor es mejor)')
    
    for i, v in enumerate(df['rmse']):
        axes[0].text(i, v, f'{v:.2f}', ha='center', va='bottom')

    sns.barplot(data=df, x='model', y='r2', ax=axes[1], palette='viridis')
    axes[1].set_title('R2 Score por Modelo (Mayor es mejor)')
    axes[1].set_ylim(0, 1.1)
    
    for i, v in enumerate(df['r2']):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')

    plt.tight_layout()
    save_plot(fig, 'model_comparison_regression.png', 'reports')

    # Seleccionar mejor modelo
    best_model = df.loc[df['rmse'].idxmin()] # Menor RMSE es mejor
    print(f"   Mejor modelo (Regresion): {best_model['model']} (RMSE: {best_model['rmse']:.4f})")
    
    return best_model

def compare_clustering_models():
    """Comparar modelos de clustering"""
    print_subsection("Comparando Modelos de Clustering")

    try:
        df = load_dataframe('clustering_results.csv', 'clustering/metrics')
    except FileNotFoundError:
        print("   No se encontraron resultados de clustering")
        return None
    
    # Visualizacion
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=df, x='model', y='silhouette', ax=axes[0], palette='cool')
    axes[0].set_title('Silhouette Score (Mayor es mejor)')
    
    sns.barplot(data=df, x='model', y='davies_bouldin', ax=axes[1], palette='cool_r')
    axes[1].set_title('Davies-Bouldin Index (Menor es mejor)')
    
    plt.tight_layout()
    save_plot(fig, 'model_comparison_clustering.png', 'reports')
    
    best_model = df.loc[df['silhouette'].idxmax()]
    print(f"   Mejor modelo (Clustering): {best_model['model']}")
    
    return best_model

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 15: COMPARACION DE MODELOS",
             "Seleccion de mejores modelos por tarea")

    setup_directories()

    # Comparar
    best_clf = compare_classification_models()
    best_reg = compare_regression_models()
    best_cluster = compare_clustering_models()
    
    # Guardar resumen final
    summary = []
    if best_clf is not None:
        summary.append({
            'Task': 'Classification',
            'Best_Model': best_clf['model'],
            'Primary_Metric': 'F1-Score',
            'Metric_Value': best_clf['f1_score']
        })
    
    if best_reg is not None:
        summary.append({
            'Task': 'Regression',
            'Best_Model': best_reg['model'],
            'Primary_Metric': 'RMSE',
            'Metric_Value': best_reg['rmse']
        })
        
    if best_cluster is not None:
        summary.append({
            'Task': 'Clustering',
            'Best_Model': best_cluster['model'],
            'Primary_Metric': 'Silhouette',
            'Metric_Value': best_cluster['silhouette']
        })
        
    if summary:
        summary_df = pd.DataFrame(summary)
        save_dataframe(summary_df, 'best_models_summary.csv', 'reports')
        print("\nRESUMEN DE MEJORES MODELOS:")
        print(summary_df.to_string(index=False))

    log_step("SCRIPT 15 COMPLETADO",
             "Comparacion finalizada y mejores modelos seleccionados")

if __name__ == "__main__":
    main()
