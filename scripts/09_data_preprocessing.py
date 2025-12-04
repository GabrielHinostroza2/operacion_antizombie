"""
SCRIPT 09: PREPROCESAMIENTO FINAL
CRISP-DM Fase 3: Data Preparation

Codificacion, escalado y train-test split

Autor: Proyecto Operacion Anti-Zombie
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

from config import *
from utils import *

def encode_categorical_variables(df):
    """Codificar variables categoricas"""
    print_subsection("Codificando variables categoricas")

    # One-hot encoding para nominales
    nominal_cols = ['Departamento', 'Edificio', 'Tratamiento_Recibido', 'Tipo_Sangre']
    nominal_cols = [col for col in nominal_cols if col in df.columns]

    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, prefix=nominal_cols, drop_first=True)
        print(f"   One-hot encoding aplicado a {len(nominal_cols)} columnas")

    # Label encoding para target
    if 'Estado_Actual' in df.columns:
        le = LabelEncoder()
        df['Estado_Actual_Encoded'] = le.fit_transform(df['Estado_Actual'])

        # Guardar encoder
        joblib.dump(le, DATA_PROCESSED / 'label_encoder_estado.pkl')
        print(f"   Label encoding aplicado a Estado_Actual")

    return df

def scale_features(X_train, X_test):
    """Escalar features numericas"""
    print_subsection("Escalando features")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar scaler
    joblib.dump(scaler, DATA_PROCESSED / 'scaler.pkl')
    print(f"   StandardScaler aplicado y guardado")

    return X_train_scaled, X_test_scaled, scaler

def prepare_classification_data(df):
    """Preparar datos para clasificacion"""
    print_subsection("Preparando datos para clasificacion")

    # Features y target
    feature_cols = [col for col in FEATURES_CLASSIFICATION if col in df.columns]

    if 'Estado_Actual_Encoded' in df.columns:
        target_col = 'Estado_Actual_Encoded'
    elif 'Estado_Actual' in df.columns:
        le = LabelEncoder()
        df['Estado_Actual_Encoded'] = le.fit_transform(df['Estado_Actual'])
        target_col = 'Estado_Actual_Encoded'
    else:
        print("   ERROR: Target no encontrado")
        return None

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Eliminar filas con target nulo
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Train-test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols

def prepare_regression_data(df, target_col='Nivel_Zombificacion'):
    """Preparar datos para regresion"""
    print_subsection(f"Preparando datos para regresion ({target_col})")

    feature_cols = [col for col in FEATURES_CLASSIFICATION if col in df.columns and col != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None

    if y is None:
        print(f"   ERROR: Target {target_col} no encontrado")
        return None

    # Eliminar filas con target nulo
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_cols

def main():
    """Funcion principal"""
    print(BANNER)
    log_step("SCRIPT 09: PREPROCESAMIENTO FINAL",
             "Codificacion, escalado y splits")

    setup_directories()

    # Cargar datos con features
    print_section("CARGANDO DATOS")
    try:
        df = load_dataframe('features_engineered.csv')
    except FileNotFoundError:
        print("ERROR: Ejecute primero 08_feature_engineering.py")
        return

    # Codificar categoricas
    print_section("CODIFICACION")
    df = encode_categorical_variables(df)

    # Preparar para clasificacion
    print_section("PREPARANDO DATOS PARA CLASIFICACION")
    clf_data = prepare_classification_data(df)

    if clf_data:
        X_train_clf, X_test_clf, y_train_clf, y_test_clf, clf_features = clf_data

        # Guardar splits
        save_dataframe(pd.DataFrame(X_train_clf, columns=clf_features), 'X_train_clf.csv')
        save_dataframe(pd.DataFrame(X_test_clf, columns=clf_features), 'X_test_clf.csv')
        save_dataframe(pd.DataFrame({'Estado_Actual': y_train_clf}), 'y_train_clf.csv')
        save_dataframe(pd.DataFrame({'Estado_Actual': y_test_clf}), 'y_test_clf.csv')

    # Preparar para regresion
    print_section("PREPARANDO DATOS PARA REGRESION")
    reg_data = prepare_regression_data(df, 'Nivel_Zombificacion')

    if reg_data:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg, reg_features = reg_data

        # Guardar splits
        save_dataframe(pd.DataFrame(X_train_reg, columns=reg_features), 'X_train_reg.csv')
        save_dataframe(pd.DataFrame(X_test_reg, columns=reg_features), 'X_test_reg.csv')
        save_dataframe(pd.DataFrame(y_train_reg, columns=['Nivel_Zombificacion']), 'y_train_reg.csv')
        save_dataframe(pd.DataFrame(y_test_reg, columns=['Nivel_Zombificacion']), 'y_test_reg.csv')

    # Guardar lista de features
    print_section("GUARDANDO CONFIGURACION")
    with open(DATA_PROCESSED / 'feature_names_classification.txt', 'w') as f:
        f.write('\n'.join(clf_features if clf_data else []))

    with open(DATA_PROCESSED / 'feature_names_regression.txt', 'w') as f:
        f.write('\n'.join(reg_features if reg_data else []))

    log_step("SCRIPT 09 COMPLETADO",
             "Datos preprocesados y listos para modelado")

if __name__ == "__main__":
    main()
