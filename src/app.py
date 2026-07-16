from utils import db_connect
engine = db_connect()

# your code here
"""
Pipeline de clasificación - Predicción de Diabetes
Dataset: Pima Indians Diabetes

Objetivo: predecir si un paciente tiene diabetes (Outcome: 0/1)
basado en medidas de diagnóstico.

Modelo: Regresión Logística (default, sin tuning de hiperparámetros)
con ajuste de threshold priorizando recall (piso >= 0.80), dado el
contexto de diagnóstico médico donde el falso negativo es el error
más costoso.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)

# -----------------------------------------------------------------
# 1. Carga de datos
# -----------------------------------------------------------------
df = pd.read_csv('diabetes.csv')

# -----------------------------------------------------------------
# 2. Limpieza: tratar ceros inválidos como missing values
# -----------------------------------------------------------------
cols_con_cero_invalido = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df_clean = df.copy()
df_clean[cols_con_cero_invalido] = df_clean[cols_con_cero_invalido].replace(0, np.nan)

# Indicadores de "faltante" para las columnas con alto % de missing
df_clean['Insulin_was_missing'] = df_clean['Insulin'].isnull().astype(int)
df_clean['SkinThickness_was_missing'] = df_clean['SkinThickness'].isnull().astype(int)

# Imputación simple (bajo % de missing)
cols_mediana_simple = ['Glucose', 'BloodPressure', 'BMI']
for col in cols_mediana_simple:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Imputación agrupada por Outcome (alto % de missing)
cols_mediana_grupo = ['SkinThickness', 'Insulin']
for col in cols_mediana_grupo:
    df_clean[col] = df_clean.groupby('Outcome')[col].transform(
        lambda x: x.fillna(x.median())
    )

# -----------------------------------------------------------------
# 3. Split train/test
# -----------------------------------------------------------------
X = df_clean.drop(columns=['Outcome'])
y = df_clean['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------
# 4. Escalado
# -----------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

# -----------------------------------------------------------------
# 5. Modelo base: Regresión Logística (default)
# -----------------------------------------------------------------
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred_default = model.predict(X_test_scaled)

print("=== Métricas con threshold default (0.5) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_default):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_default):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_default):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# -----------------------------------------------------------------
# 6. Ajuste de threshold (piso de recall = 0.80)
# -----------------------------------------------------------------
recall_floor = 0.80

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
mask = recalls[:-1] >= recall_floor
thresholds_validos = thresholds[mask]
precisions_validas = precisions[:-1][mask]

idx_optimo = np.argmax(precisions_validas)
threshold_optimo = thresholds_validos[idx_optimo]

y_pred_final = (y_pred_proba >= threshold_optimo).astype(int)

print(f"\n=== Threshold ajustado (recall >= {recall_floor}): {threshold_optimo:.4f} ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_final):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_final):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_final))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_final))

# -----------------------------------------------------------------
# 7. Coeficientes del modelo (interpretación)
# -----------------------------------------------------------------
print("\n=== Coeficientes (escala estandarizada) ===")
for feature, coef in sorted(
    zip(X_train_scaled.columns, model.coef_[0]), key=lambda x: -abs(x[1])
):
    print(f"  {feature}: {coef:.4f}")