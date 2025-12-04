"""
MAIN.PY - ORQUESTADOR PRINCIPAL
Operacion Anti-Zombie - Pipeline CRISP-DM

Este script ejecuta todo el pipeline de analisis de datos y machine learning
siguiendo la metodologia CRISP-DM.

Uso:
    python main.py                 # Ejecutar todo el pipeline
    python main.py --phase eda     # Solo fase de EDA
    python main.py --phase prep    # Solo preparacion de datos
    python main.py --phase model   # Solo modelado
    python main.py --phase eval    # Solo evaluacion

Autor: Proyecto Operacion Anti-Zombie
Fecha: 2025
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Agregar scripts al path
sys.path.append(str(Path(__file__).parent / 'scripts'))

from scripts.config import BANNER
from scripts.utils import log_step, format_time
import time

# ==============================================================================
# FUNCIONES DE EJECUCION POR FASE
# ==============================================================================

def run_script(script_name):
    """
    Ejecutar un script Python

    Args:
        script_name: Nombre del script (ej: '01_data_loading.py')

    Returns:
        True si exitoso, False si fallo
    """
    script_path = Path(__file__).parent / 'scripts' / script_name

    if not script_path.exists():
        print(f"   ADVERTENCIA: Script no encontrado: {script_name}")
        return False

    print(f"\n   Ejecutando: {script_name}")
    print(f"   " + "-" * 70)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"   OK: {script_name} completado exitosamente")
        return True

    except subprocess.CalledProcessError as e:
        print(f"   ERROR: {script_name} fallo con codigo {e.returncode}")
        return False

    except Exception as e:
        print(f"   ERROR: Excepcion al ejecutar {script_name}: {e}")
        return False

def run_data_understanding():
    """Fase 2 CRISP-DM: Data Understanding (EDA)"""
    log_step(
        "FASE 2: DATA UNDERSTANDING (EDA)",
        "Ejecutando analisis exploratorio de datos completo"
    )

    scripts = [
        '01_data_loading.py',
        '02_eda_univariate.py',
        '03_eda_bivariate.py',
        '04_eda_multivariate.py',
        '05_eda_temporal.py',
        '06_eda_network.py'
    ]

    results = {}
    for script in scripts:
        results[script] = run_script(script)

    # Resumen
    successful = sum(1 for v in results.values() if v)
    print(f"\n   Resumen EDA: {successful}/{len(scripts)} scripts ejecutados exitosamente")

    return all(results.values())

def run_data_preparation():
    """Fase 3 CRISP-DM: Data Preparation"""
    log_step(
        "FASE 3: DATA PREPARATION",
        "Ejecutando limpieza y preparacion de datos"
    )

    scripts = [
        '07_data_cleaning.py',
        '08_feature_engineering.py',
        '09_data_preprocessing.py'
    ]

    results = {}
    for script in scripts:
        results[script] = run_script(script)

    successful = sum(1 for v in results.values() if v)
    print(f"\n   Resumen Preparacion: {successful}/{len(scripts)} scripts ejecutados exitosamente")

    return all(results.values())

def run_modeling():
    """Fase 4 CRISP-DM: Modeling"""
    log_step(
        "FASE 4: MODELING",
        "Ejecutando entrenamiento de modelos"
    )

    scripts = [
        '10_model_classification.py',
        '11_model_regression.py',
        '12_model_clustering.py',
        '13_model_network_analysis.py'
    ]

    results = {}
    for script in scripts:
        results[script] = run_script(script)

    successful = sum(1 for v in results.values() if v)
    print(f"\n   Resumen Modelado: {successful}/{len(scripts)} scripts ejecutados exitosamente")

    return all(results.values())

def run_evaluation():
    """Fase 5 CRISP-DM: Evaluation"""
    log_step(
        "FASE 5: EVALUATION",
        "Ejecutando evaluacion y comparacion de modelos"
    )

    scripts = [
        '14_model_evaluation.py',
        '15_model_comparison.py',
        '16_results_export.py'
    ]

    results = {}
    for script in scripts:
        results[script] = run_script(script)

    successful = sum(1 for v in results.values() if v)
    print(f"\n   Resumen Evaluacion: {successful}/{len(scripts)} scripts ejecutados exitosamente")

    return all(results.values())

# ==============================================================================
# FUNCION PRINCIPAL
# ==============================================================================

def main():
    """Funcion principal del orquestador"""

    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description='Operacion Anti-Zombie - Pipeline CRISP-DM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                 # Ejecutar pipeline completo
  python main.py --phase eda     # Solo analisis exploratorio
  python main.py --phase prep    # Solo preparacion de datos
  python main.py --phase model   # Solo modelado
  python main.py --phase eval    # Solo evaluacion
        """
    )

    parser.add_argument(
        '--phase',
        choices=['all', 'eda', 'prep', 'model', 'eval'],
        default='all',
        help='Fase del pipeline a ejecutar (default: all)'
    )

    args = parser.parse_args()

    # Banner
    print(BANNER)
    print("="*80)
    print("ORQUESTADOR PRINCIPAL - PIPELINE CRISP-DM")
    print("="*80)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fase seleccionada: {args.phase}")
    print("="*80)

    # Inicio del timer
    start_time = time.time()

    # Ejecutar fases segun seleccion
    all_successful = True

    if args.phase in ['all', 'eda']:
        if not run_data_understanding():
            all_successful = False
            if args.phase != 'all':
                return 1

    if args.phase in ['all', 'prep']:
        if not run_data_preparation():
            all_successful = False
            if args.phase != 'all':
                return 1

    if args.phase in ['all', 'model']:
        if not run_modeling():
            all_successful = False
            if args.phase != 'all':
                return 1

    if args.phase in ['all', 'eval']:
        if not run_evaluation():
            all_successful = False

    # Fin del timer
    end_time = time.time()
    duration = end_time - start_time

    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL DE EJECUCION")
    print("="*80)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tiempo total de ejecucion: {format_time(duration)}")
    print(f"Estado: {'EXITOSO' if all_successful else 'COMPLETADO CON ADVERTENCIAS'}")
    print("="*80)

    if all_successful:
        print("\nTodos los scripts se ejecutaron correctamente!")
        print("\nResultados disponibles en:")
        print("  - resultados/eda/          # Visualizaciones EDA")
        print("  - resultados/classification/ # Modelos clasificacion")
        print("  - resultados/regression/   # Modelos regresion")
        print("  - resultados/clustering/   # Modelos clustering")
        print("  - resultados/reports/      # Reportes finales")
    else:
        print("\nALGUNOS SCRIPTS NO SE EJECUTARON")
        print("Revise los mensajes de error arriba")
        print("\nNota: Algunos scripts pueden no existir aun.")
        print("Ejecute los scripts disponibles individualmente:")
        print("  python scripts/01_data_loading.py")
        print("  python scripts/02_eda_univariate.py")
        print("  etc.")

    print("\n" + "="*80)
    print("FIN DEL PIPELINE")
    print("="*80)

    return 0 if all_successful else 1

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    sys.exit(main())
