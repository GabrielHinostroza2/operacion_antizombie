"""
SCRIPT 12: MODELOS DE CLUSTERING
CRISP-DM Fase 4: Modeling

Segmentar pacientes por perfiles de riesgo

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from config import *
from utils import *

def prepare_clustering_data():
    """Preparar datos para clustering"""
    try:
        df = load_dataframe('features_engineered.csv')
        feature_cols = [col for col in FEATURES_CLUSTERING if col in df.columns]
        X = df[feature_cols].fillna(df[feature_cols].median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, df, feature_cols
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None, None

def train_kmeans(X, n_clusters=4):
    """Entrenar K-Means"""
    print_subsection(f"K-Means (k={n_clusters})")

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    print(f"   Silhouette: {sil:.3f}")
    print(f"   Davies-Bouldin: {db:.3f}")
    print(f"   Calinski-Harabasz: {ch:.3f}")

    save_model(kmeans, f'kmeans_{n_clusters}.pkl', 'clustering/models')

    return labels, {'model': 'KMeans', 'n_clusters': n_clusters, 'silhouette': sil, 'davies_bouldin': db}

def train_hierarchical(X, n_clusters=4):
    """Entrenar Hierarchical Clustering"""
    print_subsection(f"Hierarchical Agglomerative (k={n_clusters})")

    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agglo.fit_predict(X)

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)

    print(f"   Silhouette: {sil:.3f}")
    print(f"   Davies-Bouldin: {db:.3f}")

    # Dendrograma
    fig, ax = plt.subplots(figsize=(12, 6))
    linked = linkage(X, method='ward')
    dendrogram(linked, truncate_mode='lastp', p=20, ax=ax)
    ax.set_title('Dendrograma - Clustering Jerarquico')
    save_plot(fig, 'dendrograma_hierarchical.png', 'clustering/visualizations')

    return labels, {'model': 'Hierarchical', 'n_clusters': n_clusters, 'silhouette': sil, 'davies_bouldin': db}

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 12: MODELOS DE CLUSTERING",
             "Segmentando pacientes")

    setup_directories()
    setup_plot_style()

    # Preparar datos
    print_section("PREPARANDO DATOS")
    X_scaled, df_orig, features = prepare_clustering_data()

    if X_scaled is None:
        print("ERROR: No se pudieron cargar los datos")
        return

    print(f"Muestras: {X_scaled.shape[0]} | Features: {X_scaled.shape[1]}")

    # Entrenar modelos
    results = []

    print_section("K-MEANS")
    labels_km, res_km = train_kmeans(X_scaled, n_clusters=4)
    results.append(res_km)

    print_section("HIERARCHICAL CLUSTERING")
    labels_hier, res_hier = train_hierarchical(X_scaled, n_clusters=4)
    results.append(res_hier)

    # Guardar resultados
    results_df = pd.DataFrame(results)
    save_dataframe(results_df, 'clustering_results.csv', 'clustering/metrics')

    # Guardar asignaciones
    df_orig['Cluster_KMeans'] = labels_km
    df_orig['Cluster_Hierarchical'] = labels_hier
    save_dataframe(df_orig[['ID_Paciente', 'Cluster_KMeans', 'Cluster_Hierarchical']]
                   if 'ID_Paciente' in df_orig.columns
                   else df_orig[['Cluster_KMeans', 'Cluster_Hierarchical']],
                   'cluster_assignments.csv', 'clustering/segments')

    print_section("RESUMEN")
    print(results_df.to_string(index=False))

    log_step("SCRIPT 12 COMPLETADO",
             f"Creados {len(results)} modelos de clustering")

if __name__ == "__main__":
    main()
