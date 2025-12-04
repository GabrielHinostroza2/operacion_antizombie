"""
SCRIPT 13: ANALISIS DE REDES (MODELADO)
CRISP-DM Fase 4: Modeling

Analisis predictivo basado en redes

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import networkx as nx

from config import *
from utils import *

def build_network():
    """Construir red de contagios"""
    print_subsection("Construyendo red")

    try:
        red = load_dataframe('red_contagios_cleaned.csv')
    except:
        red = load_dataframe('red_contagios_clean.csv')

    G = nx.DiGraph()
    for _, row in red.iterrows():
        if pd.notna(row.get('ID_Infectante')) and pd.notna(row.get('ID_Infectado')):
            G.add_edge(row['ID_Infectante'], row['ID_Infectado'])

    print(f"   Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
    return G, red

def identify_super_spreaders(G):
    """Identificar super-spreaders"""
    print_subsection("Identificando super-spreaders")

    out_degree = dict(G.out_degree())
    top_spreaders = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"   Top 10 super-spreaders:")
    for node, degree in top_spreaders:
        print(f"      Paciente {node}: {degree} contagios")

    # Guardar
    spreaders_df = pd.DataFrame(top_spreaders, columns=['ID_Paciente', 'Contagios'])
    save_dataframe(spreaders_df, 'super_spreaders.csv', 'clustering/network_communities')

    return top_spreaders

def detect_communities(G):
    """Detectar comunidades"""
    print_subsection("Detectando comunidades")

    try:
        import community.community_louvain as community_louvain
        # Convertir a no dirigido para Louvain
        G_undirected = G.to_undirected()
        communities = community_louvain.best_partition(G_undirected)

        n_communities = len(set(communities.values()))
        print(f"   Comunidades detectadas: {n_communities}")

        # Guardar
        comm_df = pd.DataFrame(list(communities.items()), columns=['ID_Paciente', 'Comunidad'])
        save_dataframe(comm_df, 'network_communities.csv', 'clustering/network_communities')

        return communities
    except ImportError:
        print("   ADVERTENCIA: python-louvain no instalado")
        return None

def calculate_intervention_priority(G):
    """Calcular prioridad de intervencion"""
    print_subsection("Calculando prioridad de intervencion")

    # Calcular metricas de centralidad
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)

    # Combinar metricas
    priority = {}
    for node in G.nodes():
        priority[node] = (
            0.5 * betweenness.get(node, 0) +
            0.5 * pagerank.get(node, 0)
        )

    # Top nodos para intervenir
    top_priority = sorted(priority.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"   Top 10 nodos para intervencion:")
    for node, score in top_priority[:10]:
        print(f"      Paciente {node}: {score:.4f}")

    # Guardar
    priority_df = pd.DataFrame(top_priority, columns=['ID_Paciente', 'Priority_Score'])
    save_dataframe(priority_df, 'intervention_priority.csv', 'reports')

    return top_priority

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 13: ANALISIS DE REDES (MODELADO)",
             "Analisis predictivo de red de contagios")

    setup_directories()

    # Construir red
    print_section("CONSTRUCCION DE RED")
    G, red_df = build_network()

    # Super-spreaders
    print_section("SUPER-SPREADERS")
    super_spreaders = identify_super_spreaders(G)

    # Comunidades
    print_section("DETECCION DE COMUNIDADES")
    communities = detect_communities(G)

    # Prioridad de intervencion
    print_section("PRIORIDAD DE INTERVENCION")
    priority = calculate_intervention_priority(G)

    log_step("SCRIPT 13 COMPLETADO",
             "Analisis de redes completado")

if __name__ == "__main__":
    main()
