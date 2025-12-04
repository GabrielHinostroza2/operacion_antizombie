"""
SCRIPT 11: MODELOS DE REGRESION
CRISP-DM Fase 4: Modeling

Entrenar modelos para predecir Nivel_Zombificacion

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import *
from utils import *

def train_models_regression(X_train, y_train, X_test, y_test):
    """Entrenar todos los modelos de regresion"""
    results = []

    # Decision Tree
    print_subsection("Decision Tree Regressor")
    dt = DecisionTreeRegressor(random_state=RANDOM_SEED)
    grid_dt = GridSearchCV(dt, MODEL_PARAMS['decision_tree_reg'], cv=CV_FOLDS, scoring='r2', n_jobs=-1)
    grid_dt.fit(X_train, y_train)
    y_pred = grid_dt.predict(X_test)
    results.append({
        'model': 'DecisionTree',
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    })
    save_model(grid_dt.best_estimator_, 'decision_tree_reg.pkl', 'regression/models')
    print(f"   R²: {results[-1]['r2']:.4f} | RMSE: {results[-1]['rmse']:.4f}")

    # Random Forest
    print_subsection("Random Forest Regressor")
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    grid_rf = GridSearchCV(rf, MODEL_PARAMS['random_forest_reg'], cv=CV_FOLDS, scoring='r2', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    y_pred = grid_rf.predict(X_test)
    results.append({
        'model': 'RandomForest',
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    })
    save_model(grid_rf.best_estimator_, 'random_forest_reg.pkl', 'regression/models')
    print(f"   R²: {results[-1]['r2']:.4f} | RMSE: {results[-1]['rmse']:.4f}")

    # Gradient Boosting
    print_subsection("Gradient Boosting Regressor")
    gb = GradientBoostingRegressor(random_state=RANDOM_SEED)
    grid_gb = GridSearchCV(gb, MODEL_PARAMS['xgboost_reg'], cv=CV_FOLDS, scoring='r2', n_jobs=-1)
    grid_gb.fit(X_train, y_train)
    y_pred = grid_gb.predict(X_test)
    results.append({
        'model': 'GradientBoosting',
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    })
    save_model(grid_gb.best_estimator_, 'gradient_boosting_reg.pkl', 'regression/models')
    print(f"   R²: {results[-1]['r2']:.4f} | RMSE: {results[-1]['rmse']:.4f}")

    # Ridge
    print_subsection("Ridge Regression")
    ridge = Ridge(random_state=RANDOM_SEED)
    grid_ridge = GridSearchCV(ridge, MODEL_PARAMS['ridge'], cv=CV_FOLDS, scoring='r2', n_jobs=-1)
    grid_ridge.fit(X_train, y_train)
    y_pred = grid_ridge.predict(X_test)
    results.append({
        'model': 'Ridge',
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    })
    save_model(grid_ridge.best_estimator_, 'ridge_reg.pkl', 'regression/models')
    print(f"   R²: {results[-1]['r2']:.4f} | RMSE: {results[-1]['rmse']:.4f}")

    return results

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 11: MODELOS DE REGRESION",
             "Entrenando modelos para Nivel_Zombificacion")

    setup_directories()

    # Cargar datos
    print_section("CARGANDO DATOS")
    try:
        X_train = load_dataframe('X_train_reg.csv')
        X_test = load_dataframe('X_test_reg.csv')
        y_train = load_dataframe('y_train_reg.csv')['Nivel_Zombificacion']
        y_test = load_dataframe('y_test_reg.csv')['Nivel_Zombificacion']
    except FileNotFoundError:
        print("ERROR: Ejecute primero 09_data_preprocessing.py")
        return

    # Entrenar modelos
    print_section("ENTRENANDO MODELOS")
    results = train_models_regression(X_train, y_train, X_test, y_test)

    # Guardar resultados
    results_df = pd.DataFrame(results)
    save_dataframe(results_df, 'regression_results.csv', 'regression/metrics')

    print_section("RESUMEN DE MODELOS")
    print(results_df.to_string(index=False))

    log_step("SCRIPT 11 COMPLETADO",
             f"Entrenados {len(results)} modelos de regresion")

if __name__ == "__main__":
    main()
