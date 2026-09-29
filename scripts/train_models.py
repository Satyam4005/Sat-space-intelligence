"""
Satyam — Phase 3: Core ML Training Pipeline
Trains:
1. Prospectivity Classifier: XGBoost on satellite features (MAI, SAR, Optical)
2. Shortfall Forecaster: Multi-class XGBoost on operational risk telemetry
Saves trained models to D:\Manganese\models\
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb

models_dir = r"D:\Manganese\models"
os.makedirs(models_dir, exist_ok=True)

# ==============================================================
# MODEL 1: MANGANESE PROSPECTIVITY CLASSIFIER
# ==============================================================
print("\n" + "="*55)
print("🛰️ TRAINING MODEL 1: SATELLITE PROSPECTIVITY CLASSIFIER")
print("="*55)

prospectivity_path = r"D:\Manganese\data\processed\manganese_prospectivity.csv"
if not os.path.exists(prospectivity_path):
    print(f"❌ Error: {prospectivity_path} not found!")
    exit(1)

df_prop = pd.read_csv(prospectivity_path)

# Features: All optical bands, SAR radar backscatter, and custom indices
feature_cols = [
    'B2', 'B3', 'B4', 'B8', 'B8A', 'B11', 'B12',
    'MAI', 'IronOxide', 'ClayIndex', 'NDVI', 'SWIR_ratio',
    'VV', 'VH', 'VV_VH_ratio'
]
# Keep only features present in the CSV
feature_cols = [c for c in feature_cols if c in df_prop.columns]
target_col = 'manganese_target'

X = df_prop[feature_cols]
y = df_prop[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train):,} pixels | Testing on {len(X_test):,} pixels")

# Train XGBoost Binary Classifier
prop_model = xgb.XGBClassifier(
    n_estimators=120,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
prop_model.fit(X_train, y_train)

# Evaluation
y_pred = prop_model.predict(X_test)
y_prob = prop_model.predict_proba(X_test)[:, 1]

print("\n📊 Prospectivity Model Performance:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model
model_1_path = os.path.join(models_dir, "prospectivity_xgb.joblib")
joblib.dump({"model": prop_model, "features": feature_cols}, model_1_path)
print(f"💾 Model 1 saved to: {model_1_path}")


# ==============================================================
# MODEL 2: PRODUCTION SHORTFALL RISK FORECASTER
# ==============================================================
print("\n" + "="*55)
print("🚜 TRAINING MODEL 2: MINE PRODUCTION SHORTFALL FORECASTER")
print("="*55)

risk_path = r"D:\Manganese\data\processed\mine_production_risk.csv"
if not os.path.exists(risk_path):
    print(f"❌ Error: {risk_path} not found! Run generate_production_data.py first.")
    exit(1)

df_risk = pd.read_csv(risk_path)

risk_features = ['rainfall_mm', 'active_trucks', 'excavator_hours']
risk_target = 'shortfall_risk_class'

Xr = df_risk[risk_features]
yr = df_risk[risk_target]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42, stratify=yr
)

# Train Multi-Class XGBoost
shortfall_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    objective='multi:softprob',
    num_class=3,
    random_state=42
)
shortfall_model.fit(Xr_train, yr_train)

yr_pred = shortfall_model.predict(Xr_test)

print("\n📊 Shortfall Model Performance:")
print(f"  Accuracy : {accuracy_score(yr_test, yr_pred)*100:.2f}%")
print("\nClassification Report (0=Normal, 1=Moderate, 2=Severe Shortfall):\n", 
      classification_report(yr_test, yr_pred))

# Save Model
model_2_path = os.path.join(models_dir, "shortfall_xgb.joblib")
joblib.dump({"model": shortfall_model, "features": risk_features}, model_2_path)
print(f"💾 Model 2 saved to: {model_2_path}")

print("\n" + "="*55)
print("🎉 PHASE 3 COMPLETE! Both AI Models Trained & Saved.")
print("="*55)