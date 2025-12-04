"""
SCRIPT 14: EVALUACION DE MODELOS
CRISP-DM Fase 5: Evaluation

Evaluacion detallada de los modelos entrenados (matrices de confusion, curvas ROC, residuos)

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import label_binarize
import joblib

from config import *
from utils import *

def evaluate_classification_models(X_test, y_test):
    """Evaluacion detallada de modelos de clasificacion"""
    print_subsection("Evaluando Modelos de Clasificacion")

    models_dir = RESULTS_CLASSIFICATION / 'models'
    if not models_dir.exists():
        print("   ERROR: No se encontraron modelos de clasificacion")
        return

    model_files = list(models_dir.glob('*.pkl'))
    print(f"   Encontrados {len(model_files)} modelos")

    # Para ROC multiclass
    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    eval_metrics = []

    for model_path in model_files:
        model_name = model_path.stem
        print(f"\n   Evaluando: {model_name}")
        
        try:
            model = load_model(model_path.name, 'classification/models')
            y_pred = model.predict(X_test)
            
            # Metricas basicas
            report = classification_report(y_test, y_pred, output_dict=True)
            eval_metrics.append({
                'model': model_name,
                'accuracy': report['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score']
            })

            # Matriz de Confusion
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=ESTADO_LABELS.keys(),
                       yticklabels=ESTADO_LABELS.keys())
            plt.title(f'Matriz de Confusion - {model_name}')
            plt.ylabel('Real')
            plt.xlabel('Predicho')
            save_plot(plt.gcf(), f'confusion_matrix_{model_name}.png', 'classification/visualizations')

            # Feature Importance (si aplica)
            if hasattr(model, 'feature_importances_'):
                try:
                    feature_names = load_dataframe('X_train_clf.csv').columns
                    importances = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=importances.head(10), x='importance', y='feature', palette='viridis')
                    plt.title(f'Top 10 Feature Importance - {model_name}')
                    save_plot(plt.gcf(), f'feature_importance_{model_name}.png', 'classification/visualizations')
                except Exception as e:
                    print(f"      No se pudo graficar feature importance: {e}")

        except Exception as e:
            print(f"      Error evaluando {model_name}: {e}")

    # Guardar metricas consolidadas
    pd.DataFrame(eval_metrics).to_csv(RESULTS_CLASSIFICATION / 'metrics' / 'detailed_evaluation.csv', index=False)

def evaluate_regression_models(X_test, y_test):
    """Evaluacion detallada de modelos de regresion"""
    print_subsection("Evaluando Modelos de Regresion")

    models_dir = RESULTS_REGRESSION / 'models'
    if not models_dir.exists():
        print("   ERROR: No se encontraron modelos de regresion")
        return

    model_files = list(models_dir.glob('*.pkl'))
    print(f"   Encontrados {len(model_files)} modelos")
    
    for model_path in model_files:
        model_name = model_path.stem
        print(f"\n   Evaluando: {model_name}")
        
        try:
            model = load_model(model_path.name, 'regression/models')
            y_pred = model.predict(X_test)
            
            # Grafico de Residuos vs Predichos
            residuals = y_test - y_pred.flatten()
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Predichos vs Reales
            axes[0].scatter(y_test, y_pred, alpha=0.5)
            axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0].set_xlabel('Reales')
            axes[0].set_ylabel('Predichos')
            axes[0].set_title(f'Predicho vs Real - {model_name}')
            
            # Residuos
            axes[1].scatter(y_pred, residuals, alpha=0.5)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predichos')
            axes[1].set_ylabel('Residuos')
            axes[1].set_title(f'Residuos - {model_name}')
            
            save_plot(fig, f'residuals_{model_name}.png', 'regression/visualizations')
            
            # Feature Importance (si aplica)
            if hasattr(model, 'feature_importances_'):
                try:
                    feature_names = load_dataframe('X_train_reg.csv').columns
                    importances = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=importances.head(10), x='importance', y='feature', palette='viridis')
                    plt.title(f'Top 10 Feature Importance - {model_name}')
                    save_plot(plt.gcf(), f'feature_importance_{model_name}.png', 'regression/visualizations')
                except Exception as e:
                    print(f"      No se pudo graficar feature importance: {e}")

        except Exception as e:
            print(f"      Error evaluando {model_name}: {e}")

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 14: EVALUACION DE MODELOS",
             "Evaluacion detallada y visualizacion de resultados")

    setup_directories()

    # 1. Evaluacion Clasificacion
    print_section("EVALUACION CLASIFICACION")
    try:
        X_test_clf = load_dataframe('X_test_clf.csv')
        y_test_clf = load_dataframe('y_test_clf.csv')['Estado_Actual']
        evaluate_classification_models(X_test_clf, y_test_clf)
    except FileNotFoundError:
        print("   No se encontraron datos de test para clasificacion")

    # 2. Evaluacion Regresion
    print_section("EVALUACION REGRESION")
    try:
        X_test_reg = load_dataframe('X_test_reg.csv')
        y_test_reg = load_dataframe('y_test_reg.csv')['Nivel_Zombificacion']
        evaluate_regression_models(X_test_reg, y_test_reg)
    except FileNotFoundError:
        print("   No se encontraron datos de test para regresion")

    log_step("SCRIPT 14 COMPLETADO",
             "Evaluaciones guardadas en carpetas de visualizaciones")

if __name__ == "__main__":
    main()
