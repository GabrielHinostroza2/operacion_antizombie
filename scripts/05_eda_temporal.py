"""
SCRIPT 05: ANALISIS TEMPORAL
CRISP-DM Fase 2: Data Understanding

Analisis de series temporales del brote

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from utils import *

def analyze_outbreak_evolution(df):
    """Analizar evolucion del brote"""
    print_subsection("Analizando evolucion temporal del brote")

    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df = df.sort_values('Fecha')

    # 1. Casos nuevos y acumulados
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Casos nuevos
    if 'Casos_Nuevos' in df.columns:
        axes[0, 0].plot(df.index, df['Casos_Nuevos'], marker='o', linewidth=2)
        axes[0, 0].fill_between(df.index, df['Casos_Nuevos'], alpha=0.3)
        axes[0, 0].set_xlabel('Dia del Brote')
        axes[0, 0].set_ylabel('Casos Nuevos')
        axes[0, 0].set_title('Casos Nuevos Diarios')
        axes[0, 0].grid(True, alpha=0.3)

        # Media movil 7 dias
        if len(df) >= 7:
            ma7 = df['Casos_Nuevos'].rolling(window=7).mean()
            axes[0, 0].plot(df.index, ma7, 'r--', linewidth=2,
                           label='Media Movil 7 dias')
            axes[0, 0].legend()

    # Casos acumulados
    if 'Casos_Acumulados' in df.columns:
        axes[0, 1].plot(df.index, df['Casos_Acumulados'],
                       marker='o', linewidth=2, color='red')
        axes[0, 1].fill_between(df.index, df['Casos_Acumulados'],
                                alpha=0.3, color='red')
        axes[0, 1].set_xlabel('Dia del Brote')
        axes[0, 1].set_ylabel('Casos Acumulados')
        axes[0, 1].set_title('Curva de Casos Acumulados')
        axes[0, 1].grid(True, alpha=0.3)

    # Tasa de contagio R0
    if 'Tasa_Contagio_R0' in df.columns:
        axes[1, 0].plot(df.index, df['Tasa_Contagio_R0'],
                       marker='o', linewidth=2, color='green')
        axes[1, 0].axhline(y=1, color='r', linestyle='--',
                          label='R0 = 1 (umbral)')
        axes[1, 0].set_xlabel('Dia del Brote')
        axes[1, 0].set_ylabel('R0')
        axes[1, 0].set_title('Evolucion de Tasa de Contagio R0')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

    # Casos activos, recuperados, fallecidos
    if all(col in df.columns for col in ['Casos_Activos', 'Recuperados_Dia', 'Fallecidos_Dia']):
        axes[1, 1].plot(df.index, df['Casos_Activos'],
                       label='Activos', linewidth=2)
        if 'Recuperados_Dia' in df.columns:
            recuperados_acum = df['Recuperados_Dia'].cumsum()
            axes[1, 1].plot(df.index, recuperados_acum,
                           label='Recuperados (acum)', linewidth=2)
        if 'Fallecidos_Dia' in df.columns:
            fallecidos_acum = df['Fallecidos_Dia'].cumsum()
            axes[1, 1].plot(df.index, fallecidos_acum,
                           label='Fallecidos (acum)', linewidth=2)

        axes[1, 1].set_xlabel('Dia del Brote')
        axes[1, 1].set_ylabel('Cantidad de Pacientes')
        axes[1, 1].set_title('Evolucion de Casos por Estado')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

    plt.suptitle('Analisis Temporal del Brote', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'evolucion_temporal_brote.png', subdir='eda/temporal')

def analyze_epidemic_metrics(df):
    """Analizar metricas epidemiologicas"""
    print_subsection("Analizando metricas epidemiologicas")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Tasa de mortalidad
    if 'Tasa_Mortalidad' in df.columns:
        axes[0].plot(df.index, df['Tasa_Mortalidad'] * 100,
                    marker='o', linewidth=2, color='red')
        axes[0].set_xlabel('Dia del Brote')
        axes[0].set_ylabel('Tasa de Mortalidad (%)')
        axes[0].set_title('Evolucion de Tasa de Mortalidad')
        axes[0].grid(True, alpha=0.3)

    # Tasa de recuperacion
    if 'Tasa_Recuperacion' in df.columns:
        axes[1].plot(df.index, df['Tasa_Recuperacion'] * 100,
                    marker='o', linewidth=2, color='green')
        axes[1].set_xlabel('Dia del Brote')
        axes[1].set_ylabel('Tasa de Recuperacion (%)')
        axes[1].set_title('Evolucion de Tasa de Recuperacion')
        axes[1].grid(True, alpha=0.3)

    # Tratamientos administrados
    if 'Tratamientos_Administrados' in df.columns:
        axes[2].bar(df.index, df['Tratamientos_Administrados'],
                   color='steelblue', edgecolor='black')
        axes[2].set_xlabel('Dia del Brote')
        axes[2].set_ylabel('Tratamientos Administrados')
        axes[2].set_title('Tratamientos Administrados por Dia')
        axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Metricas Epidemiologicas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'metricas_epidemiologicas.png', subdir='eda/temporal')

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 05: ANALISIS TEMPORAL",
             "Analisis de series temporales del brote")

    setup_plot_style()
    setup_directories()

    try:
        evolucion = load_dataframe('evolucion_clean.csv')
    except FileNotFoundError:
        print("ERROR: Ejecute primero 01_data_loading.py")
        return

    print_section("EVOLUCION DEL BROTE")
    analyze_outbreak_evolution(evolucion)

    print_section("METRICAS EPIDEMIOLOGICAS")
    analyze_epidemic_metrics(evolucion)

    log_step("SCRIPT 05 COMPLETADO",
             "Visualizaciones en: resultados/eda/temporal/")

if __name__ == "__main__":
    main()
