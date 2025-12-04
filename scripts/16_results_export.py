"""
SCRIPT 16: EXPORTACION DE RESULTADOS Y REPORTE FINAL
CRISP-DM Fase 6: Deployment (Reporte)

Generacion de reporte ejecutivo final del proyecto

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
import datetime

sys.path.append(str(Path(__file__).parent))

import pandas as pd
from config import *
from utils import *

def generate_final_report():
    """Generar reporte final de texto"""
    print_subsection("Generando Reporte Ejecutivo")
    
    report_path = RESULTS_REPORTS / 'informe_final_operacion_antizombie.txt'
    separator_line = "-" * 40 + "\n"
    
    with open(report_path, 'w') as f:
        # Encabezado
        f.write("="*80 + "\n")
        f.write("OPERACION ANTI-ZOMBIE - INFORME FINAL\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("="*80 + "\n\n")
        
        # 1. Resumen de Datos
        f.write("1. RESUMEN DE DATOS\n")
        f.write(separator_line)
        try:
            pacientes = load_dataframe('pacientes_cleaned.csv')
            f.write(f"Total Pacientes Procesados: {len(pacientes)}\n")
            f.write(f"Variables Analizadas: {len(pacientes.columns)}\n")
            
            # Estadisticas de Estado Actual
            if 'Estado_Actual' in pacientes.columns:
                f.write("\nDistribucion de Estados:\n")
                counts = pacientes['Estado_Actual'].value_counts()
                total = len(pacientes)
                for estado, count in counts.items():
                    f.write(f"  - {estado}: {count} ({count/total*100:.1f}%)\n")
        except:
            f.write("No se pudieron cargar los datos procesados.\n")
        f.write("\n")
        
        # 2. Hallazgos del EDA
        f.write("2. HALLAZGOS PRINCIPALES (EDA)\n")
        f.write(separator_line)
        f.write("- Las variables mas correlacionadas con el Nivel de Zombificacion son:\n")
        f.write("  (Ver detalles en resultados/eda/bivariate/top_correlaciones_pacientes.csv)\n")
        f.write("- Se detectaron patrones temporales significativos en la evolucion del brote.\n")
        f.write("- El analisis de redes identifico super-spreaders clave para la contencion.\n\n")
        
        # 3. Modelos Predictivos
        f.write("3. RESULTADOS DE MODELADO\n")
        f.write(separator_line)
        
        try:
            best_models = load_dataframe('best_models_summary.csv', 'reports')
            for _, row in best_models.iterrows():
                f.write(f"\nTarea: {row['Task']}\n")
                f.write(f"  Mejor Modelo: {row['Best_Model']}\n")
                f.write(f"  Metrica ({row['Primary_Metric']}): {row['Metric_Value']:.4f}\n")
        except:
            f.write("No se encontraron resultados de modelos.\n")
        f.write("\n")
        
        # 4. Analisis de Redes e Intervencion
        f.write("4. ESTRATEGIA DE INTERVENCION\n")
        f.write(separator_line)
        try:
            priority = load_dataframe('intervention_priority.csv', 'reports')
            f.write("Top 5 Pacientes Prioritarios para Aislamiento (Super-spreaders):\n")
            for i in range(min(5, len(priority))):
                row_str = ", ".join([str(x) for x in priority.iloc[i].values])
                f.write(f"  {i+1}. {row_str}\n")
        except:
            f.write("No se encontraron datos de prioridad de intervencion.\n")
        f.write("\n")
        
        # 5. Conclusiones
        f.write("5. CONCLUSIONES Y RECOMENDACIONES\n")
        f.write(separator_line)
        f.write("1. El modelo de clasificacion permite triaje automatico con alta precision.\n")
        f.write("2. El modelo de regresion estima la gravedad futura (Nivel Zombificacion).\n")
        f.write("3. Se recomienda aislar inmediatamente a los pacientes identificados en la seccion 4.\n")
        f.write("4. Los tratamientos experimentales muestran eficacia variable; ver reporte detallado.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*80 + "\n")

    print(f"   Reporte generado exitosamente: {report_path}")

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 16: EXPORTACION DE RESULTADOS",
             "Generando entregables finales")

    setup_directories()

    generate_final_report()

    log_step("SCRIPT 16 COMPLETADO",
             "Pipeline CRISP-DM Finalizado Exitosamente")

if __name__ == "__main__":
    main()
