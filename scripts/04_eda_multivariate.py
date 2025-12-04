"""
SCRIPT 04: ANALISIS MULTIVARIADO
CRISP-DM Fase 2: Data Understanding

Analisis PCA, feature importance, interacciones

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from config import *
from utils import *

def perform_pca_analysis(df, target_col='Estado_Actual'):
    """Analisis PCA"""
    print_subsection("Realizando PCA")

    # Seleccionar features numericas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if 'ID' not in col.upper()]

    # Eliminar target si esta
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    X = df[numeric_cols].fillna(df[numeric_cols].median())

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # 1. Varianza explicada
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_)
    axes[0].set_xlabel('Componente Principal')
    axes[0].set_ylabel('Varianza Explicada')
    axes[0].set_title('Scree Plot - Varianza por Componente')
    axes[0].grid(True, alpha=0.3)

    # Varianza acumulada
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, marker='o')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
    axes[1].set_xlabel('Numero de Componentes')
    axes[1].set_ylabel('Varianza Explicada Acumulada')
    axes[1].set_title('Varianza Acumulada')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_plot(fig, 'pca_varianza_explicada.png', subdir='eda/multivariate')

    # 2. Biplot (PC1 vs PC2)
    fig, ax = plt.subplots(figsize=(12, 8))

    if target_col in df.columns:
        # Colorear por target
        targets = df[target_col].fillna('Desconocido')
        for target in targets.unique():
            mask = targets == target
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      label=target, alpha=0.6, s=50)
        ax.legend()
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    ax.set_title('PCA Biplot - Primeros 2 Componentes')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, 'pca_biplot.png', subdir='eda/multivariate')

    print(f"   Varianza explicada por PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"   Varianza explicada por PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"   Componentes necesarios para 95% varianza: {np.argmax(cumsum_var >= 0.95) + 1}")

def analyze_feature_importance(df, target_col='Estado_Actual'):
    """Feature importance con Random Forest"""
    print_subsection("Analizando Feature Importance")

    if target_col not in df.columns:
        return

    # Preparar datos
    numeric_cols = [col for col in FEATURES_CLASSIFICATION if col in df.columns]
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y = df[target_col]

    # Eliminar filas con target nulo
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X, y)

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Visualizar
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df['importance'].values)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('Importancia')
    ax.set_title('Feature Importance (Random Forest)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    save_plot(fig, 'feature_importance_rf.png', subdir='eda/multivariate')

    # Guardar
    save_dataframe(importance_df, 'feature_importance.csv',
                   subdir='eda/multivariate')

    print(f"   Top 5 features mas importantes:")
    for _, row in importance_df.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

def analyze_pairplot(df):
    """Pairplot de features clave"""
    print_subsection("Generando Pairplot")

    # Seleccionar top 6 features
    top_features = ['Edad', 'Nivel_Zombificacion', 'Temperatura_Corporal',
                   'Nivel_Consciencia', 'Agresividad', 'Mejoria_Porcentual']
    top_features = [f for f in top_features if f in df.columns]

    if 'Estado_Actual' in df.columns:
        plot_df = df[top_features + ['Estado_Actual']].sample(
            min(500, len(df)), random_state=RANDOM_SEED
        )

        g = sns.pairplot(plot_df, hue='Estado_Actual', diag_kind='kde',
                        plot_kws={'alpha': 0.6}, corner=True)
        g.fig.suptitle('Pairplot de Features Principales', y=1.02)
        save_plot(g.fig, 'pairplot_features.png', subdir='eda/multivariate')

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 04: ANALISIS MULTIVARIADO",
             "PCA, Feature Importance, Interacciones")

    setup_plot_style()
    setup_directories()

    try:
        pacientes = load_dataframe('pacientes_clean.csv')
    except FileNotFoundError:
        print("ERROR: Ejecute primero 01_data_loading.py")
        return

    print_section("ANALISIS PCA")
    perform_pca_analysis(pacientes)

    print_section("FEATURE IMPORTANCE")
    analyze_feature_importance(pacientes)

    print_section("PAIRPLOT")
    analyze_pairplot(pacientes)

    log_step("SCRIPT 04 COMPLETADO",
             "Visualizaciones en: resultados/eda/multivariate/")

if __name__ == "__main__":
    main()
