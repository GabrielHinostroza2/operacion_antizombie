"""
SCRIPT 06: ANALISIS DE REDES
CRISP-DM Fase 2: Data Understanding

Analisis de la red de contagios

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from config import *
from utils import *

def build_contagion_network(df_red):
    """Construir grafo de red de contagios"""
    print_subsection("Construyendo red de contagios")

    G = nx.DiGraph()

    for _, row in df_red.iterrows():
        if pd.notna(row.get('ID_Infectante')) and pd.notna(row.get('ID_Infectado')):
            G.add_edge(
                row['ID_Infectante'],
                row['ID_Infectado'],
                weight=row.get('Probabilidad_Contagio', 1.0)
            )

    print(f"   Nodos (pacientes): {G.number_of_nodes()}")
    print(f"   Aristas (contagios): {G.number_of_edges()}")

    return G

def analyze_network_centrality(G):
    """Analizar metricas de centralidad"""
    print_subsection("Calculando metricas de centralidad")

    # Degree centrality
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    # Top infectadores (super-spreaders)
    top_spreaders = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:20]

    # Visualizar
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top 20 super-spreaders
    if top_spreaders:
        ids, degrees = zip(*top_spreaders)
        axes[0].barh(range(len(ids)), degrees, color='red', edgecolor='black')
        axes[0].set_yticks(range(len(ids)))
        axes[0].set_yticklabels([f'Paciente {id}' for id in ids])
        axes[0].set_xlabel('Personas Contagiadas')
        axes[0].set_title('Top 20 Super-Spreaders')
        axes[0].grid(True, alpha=0.3, axis='x')

    # Distribucion de grados
    in_degrees = list(in_degree.values())
    out_degrees = list(out_degree.values())

    axes[1].hist([in_degrees, out_degrees], bins=20, label=['In-degree', 'Out-degree'],
                alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Grado')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribucion de Grados (In/Out)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, 'network_centrality.png', subdir='eda/network')

    # Guardar metricas
    centrality_df = pd.DataFrame({
        'ID_Paciente': list(in_degree.keys()),
        'In_Degree': list(in_degree.values()),
        'Out_Degree': [out_degree.get(k, 0) for k in in_degree.keys()]
    })
    save_dataframe(centrality_df, 'centrality_measures.csv',
                   subdir='eda/network')

    print(f"   Super-spreader principal: Paciente {top_spreaders[0][0]} ({top_spreaders[0][1]} contagios)")

def visualize_network(G, sample_size=100):
    """Visualizar red de contagios"""
    print_subsection("Visualizando red de contagios")

    # Si la red es muy grande, tomar muestra
    if G.number_of_nodes() > sample_size:
        # Tomar nodos con mayor degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:sample_size]
        nodes_to_keep = [n[0] for n in top_nodes]
        G_sample = G.subgraph(nodes_to_keep).copy()
        print(f"   Mostrando subgrafo con {sample_size} nodos mas conectados")
    else:
        G_sample = G

    fig, ax = plt.subplots(figsize=(14, 14))

    # Layout
    pos = nx.spring_layout(G_sample, k=2, iterations=50, seed=RANDOM_SEED)

    # Tamaño de nodos por degree
    node_sizes = [G_sample.degree(node) * 50 + 100 for node in G_sample.nodes()]

    # Dibujar
    nx.draw_networkx_nodes(G_sample, pos, node_size=node_sizes,
                          node_color='lightblue', edgecolors='black',
                          linewidths=1, alpha=0.7, ax=ax)

    nx.draw_networkx_edges(G_sample, pos, edge_color='gray',
                          arrows=True, arrowsize=10,
                          width=0.5, alpha=0.5, ax=ax)

    # Etiquetar solo nodos importantes
    top_degree = sorted(G_sample.degree(), key=lambda x: x[1], reverse=True)[:10]
    labels = {node: f'P{node}' for node, _ in top_degree}
    nx.draw_networkx_labels(G_sample, pos, labels, font_size=8, ax=ax)

    ax.set_title('Red de Contagios (nodos principales)', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    save_plot(fig, 'network_graph.png', subdir='eda/network')

def analyze_contact_patterns(df_red):
    """Analizar patrones de contacto"""
    print_subsection("Analizando patrones de contacto")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Tipo de contacto vs probabilidad
    if 'Tipo_Contacto' in df_red.columns and 'Probabilidad_Contagio' in df_red.columns:
        df_red.boxplot(column='Probabilidad_Contagio', by='Tipo_Contacto', ax=axes[0])
        axes[0].set_title('Probabilidad de Contagio por Tipo de Contacto')
        axes[0].set_xlabel('Tipo de Contacto')
        axes[0].set_ylabel('Probabilidad de Contagio')

    # Lugares de contagio
    if 'Lugar_Contagio' in df_red.columns:
        lugar_counts = df_red['Lugar_Contagio'].value_counts()
        axes[1].barh(range(len(lugar_counts)), lugar_counts.values,
                    color='coral', edgecolor='black')
        axes[1].set_yticks(range(len(lugar_counts)))
        axes[1].set_yticklabels(lugar_counts.index)
        axes[1].set_xlabel('Numero de Contagios')
        axes[1].set_title('Hotspots de Contagio')
        axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_plot(fig, 'contact_patterns.png', subdir='eda/network')

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 06: ANALISIS DE REDES",
             "Analisis de la red de contagios")

    setup_plot_style()
    setup_directories()

    try:
        red_contagios = load_dataframe('red_contagios_clean.csv')
    except FileNotFoundError:
        print("ERROR: Ejecute primero 01_data_loading.py")
        return

    print_section("CONSTRUCCION DE RED")
    G = build_contagion_network(red_contagios)

    print_section("METRICAS DE CENTRALIDAD")
    analyze_network_centrality(G)

    print_section("VISUALIZACION DE RED")
    visualize_network(G)

    print_section("PATRONES DE CONTACTO")
    analyze_contact_patterns(red_contagios)

    log_step("SCRIPT 06 COMPLETADO",
             "Visualizaciones en: resultados/eda/network/")

if __name__ == "__main__":
    main()
