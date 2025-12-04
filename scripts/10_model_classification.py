"""
SCRIPT 10: MODELOS DE CLASIFICACION
CRISP-DM Fase 4: Modeling

Entrenar modelos para predecir Estado_Actual

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

from config import *
from utils import *

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Entrenar Decision Tree"""
    print_subsection("Decision Tree Classifier")

    param_grid = MODEL_PARAMS['decision_tree_clf']
    dt = DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight=CLASS_WEIGHT)

    grid = GridSearchCV(dt, param_grid, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"   Best params: {grid.best_params_}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-score: {f1:.4f}")

    save_model(best_model, 'decision_tree_clf.pkl', 'classification/models')

    return {'model': 'DecisionTree', 'accuracy': acc, 'f1_score': f1, 'params': grid.best_params_}

def train_random_forest(X_train, y_train, X_test, y_test):
    """Entrenar Random Forest"""
    print_subsection("Random Forest Classifier")

    param_grid = MODEL_PARAMS['random_forest_clf']
    rf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight=CLASS_WEIGHT)

    grid = GridSearchCV(rf, param_grid, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"   Best params: {grid.best_params_}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-score: {f1:.4f}")

    save_model(best_model, 'random_forest_clf.pkl', 'classification/models')

    return {'model': 'RandomForest', 'accuracy': acc, 'f1_score': f1, 'params': grid.best_params_}

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Entrenar Gradient Boosting"""
    print_subsection("Gradient Boosting Classifier")

    param_grid = MODEL_PARAMS['xgboost_clf']
    gb = GradientBoostingClassifier(random_state=RANDOM_SEED)

    grid = GridSearchCV(gb, param_grid, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"   Best params: {grid.best_params_}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-score: {f1:.4f}")

    save_model(best_model, 'gradient_boosting_clf.pkl', 'classification/models')

    return {'model': 'GradientBoosting', 'accuracy': acc, 'f1_score': f1, 'params': grid.best_params_}

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Entrenar Logistic Regression"""
    print_subsection("Logistic Regression")

    param_grid = MODEL_PARAMS['logistic_regression']
    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight=CLASS_WEIGHT)

    grid = GridSearchCV(lr, param_grid, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"   Best params: {grid.best_params_}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-score: {f1:.4f}")

    save_model(best_model, 'logistic_regression_clf.pkl', 'classification/models')

    return {'model': 'LogisticRegression', 'accuracy': acc, 'f1_score': f1, 'params': grid.best_params_}

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 10: MODELOS DE CLASIFICACION",
             "Entrenando modelos para Estado_Actual")

    setup_directories()

    # Cargar datos
    print_section("CARGANDO DATOS")
    try:
        X_train = load_dataframe('X_train_clf.csv')
        X_test = load_dataframe('X_test_clf.csv')
        y_train = load_dataframe('y_train_clf.csv')['Estado_Actual']
        y_test = load_dataframe('y_test_clf.csv')['Estado_Actual']
    except FileNotFoundError:
        print("ERROR: Ejecute primero 09_data_preprocessing.py")
        return

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Entrenar modelos
    results = []

    print_section("ENTRENANDO MODELOS")
    results.append(train_decision_tree(X_train, y_train, X_test, y_test))
    results.append(train_random_forest(X_train, y_train, X_test, y_test))
    results.append(train_gradient_boosting(X_train, y_train, X_test, y_test))
    results.append(train_logistic_regression(X_train, y_train, X_test, y_test))

    # Guardar resultados
    results_df = pd.DataFrame(results)
    save_dataframe(results_df, 'classification_results.csv', 'classification/metrics')

    # Mostrar resumen
    print_section("RESUMEN DE MODELOS")
    print(results_df[['model', 'accuracy', 'f1_score']].to_string(index=False))

    log_step("SCRIPT 10 COMPLETADO",
             f"Entrenados {len(results)} modelos de clasificacion")

if __name__ == "__main__":
    main()
