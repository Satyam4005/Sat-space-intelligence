"""
Satyam — Phase 5: High-Performance FastAPI Backend
Unifies:
1. /api/v1/prospectivity/predict (Satellite features + ESG Eco-mask + SHAP transparency)
2. /api/v1/production/forecast  (Weather + Fleet + Prescriptive What-If Mitigation)
3. /api/v1/edge/benchmark        (Offline latency metrics)
"""

import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import shap

# --- INITIALIZE APP ---
app = FastAPI(
    title="Satyam Space Intelligence API",
    description="AI-powered manganese prospectivity mapping & prescriptive production planning",
    version="1.0.0"
)

# Enable CORS for React / Streamlit frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODELS & SHAP EXPLAINER ---
models_dir = r"D:\Manganese\models"
prop_pkg = joblib.load(os.path.join(models_dir, "prospectivity_xgb.joblib"))
shortfall_pkg = joblib.load(os.path.join(models_dir, "shortfall_xgb.joblib"))

prop_model = prop_pkg["model"]
prop_features = prop_pkg["features"]

shortfall_model = shortfall_pkg["model"]
shortfall_features = shortfall_pkg["features"]

explainer = shap.TreeExplainer(prop_model)


# --- REQUEST & RESPONSE SCHEMAS ---
class ProspectivityRequest(BaseModel):
    latitude: float = 21.849
    longitude: float = 80.203
    B2: float = Field(709.0, description="Blue reflectance")
    B3: float = Field(920.0, description="Green reflectance")
    B4: float = Field(1150.0, description="Red reflectance")
    B8: float = Field(1930.0, description="NIR reflectance")
    B8A: float = Field(2100.0, description="RedEdge reflectance")
    B11: float = Field(2400.0, description="SWIR1 reflectance")
    B12: float = Field(1800.0, description="SWIR2 reflectance")
    MAI: float = Field(0.09, description="Manganese Alteration Index")
    IronOxide: float = Field(0.24, description="Iron Oxide Index")
    ClayIndex: float = Field(0.15, description="Clay Alteration Index")
    NDVI: float = Field(0.32, description="Vegetation Index")
    SWIR_ratio: float = Field(1.14, description="SWIR Ratio")
    VV: float = Field(-11.5, description="SAR VV backscatter")
    VH: float = Field(-17.2, description="SAR VH backscatter")
    VV_VH_ratio: float = Field(0.66, description="SAR Structural Roughness")


class ProductionRiskRequest(BaseModel):
    rainfall_mm: float = Field(38.5, description="Forecasted / Current Rainfall (mm)")
    active_trucks: int = Field(17, description="Number of operable dump trucks (Max: 25)")
    excavator_hours: float = Field(11.5, description="Scheduled excavator operational hours")


# --- ROUTES ---
@app.get("/")
def root():
    return {
        "platform": "Satyam Space Intelligence Engine",
        "status": "Online",
        "district": "Balaghat, Madhya Pradesh",
        "docs_url": "/docs"
    }


@app.post("/api/v1/prospectivity/predict")
def predict_prospectivity(req: ProspectivityRequest):
    """
    Evaluates satellite spectral signature, checks ESG eco-constraints,
    and returns SHAP feature attribution breakdown.
    """
    # 1. ESG Eco-Constraint Check
    if req.NDVI > 0.62:
        return {
            "status": "ECO_RESTRICTED",
            "manganese_probability": 0.0,
            "esg_violation": True,
            "message": "🚨 STRICT NEGATIVE CONSTRAINT TRIGGERED: Target lies in protected dense biosphere reserve / high-density canopy. Exploration legally prohibited."
        }

    # 2. Prepare feature vector
    input_dict = req.model_dump()
    row = pd.DataFrame([input_dict])[prop_features]

    prob = float(prop_model.predict_proba(row)[0, 1])

    # 3. SHAP Explainability
    shap_vals = explainer.shap_values(row)[0]
    drivers = []
    for feat, val, s_val in zip(prop_features, row.iloc[0], shap_vals):
        drivers.append({
            "feature": feat,
            "measured_value": round(float(val), 4),
            "shap_impact_pct": round(float(s_val * 100), 2)
        })
    drivers.sort(key=lambda x: abs(x["shap_impact_pct"]), reverse=True)

    return {
        "latitude": req.latitude,
        "longitude": req.longitude,
        "manganese_probability": round(prob * 100, 2),
        "is_viable_deposit": bool(prob >= 0.5),
        "esg_compliant": True,
        "top_shap_drivers": drivers[:5]
    }


@app.post("/api/v1/production/forecast")
def forecast_production_shortfall(req: ProductionRiskRequest):
    """
    Forecasts supply shortfalls and outputs prescriptive dispatch commands.
    """
    row = pd.DataFrame([{
        "rainfall_mm": req.rainfall_mm,
        "active_trucks": req.active_trucks,
        "excavator_hours": req.excavator_hours
    }])[shortfall_features]

    risk_class = int(shortfall_model.predict(row)[0])
    probs = shortfall_model.predict_proba(row)[0]

    QUOTA = 850.0
    est_production = (req.active_trucks * 18) + (req.excavator_hours * 30)
    if req.rainfall_mm > 25:
        est_production -= (req.rainfall_mm * 3.2)
    est_production = max(est_production, 0)
    deficit = max(QUOTA - est_production, 0)

    risk_names = ["NORMAL (Low Risk)", "MODERATE SHORTFALL", "CRITICAL SHORTFALL"]
    
    actions = []
    if risk_class == 0:
        actions.append("Operations nominal. Maintain scheduled haul routes.")
    else:
        trucks_needed = int(np.ceil(deficit / 35.0))
        hours_needed = round(deficit / 30.0, 1)

        if req.rainfall_mm > 25:
            actions.append(f"🌧️ WEATHER HAZARD ({req.rainfall_mm}mm rain): Ramp slippage risk. Reroute {min(trucks_needed, 4)} dump trucks to paved South Sector B benches.")
        if req.active_trucks < 20:
            actions.append(f"⚠️ FLEET DEFICIT: {25 - req.active_trucks} trucks offline. Mobilize auxiliary maintenance crew.")
        actions.append(f"⚡ OVERTIME DISPATCH: Authorize +{min(hours_needed, 2.5)} excavator overtime hours on high-grade Pit #4 to salvage ~{round(deficit)} tons.")

    return {
        "risk_level": risk_names[risk_class],
        "confidence_breakdown": {
            "normal": round(float(probs[0]*100), 1),
            "moderate": round(float(probs[1]*100), 1),
            "critical": round(float(probs[2]*100), 1)
        },
        "forecasted_production_tons": round(float(est_production), 1),
        "quota_tons": QUOTA,
        "projected_deficit_tons": round(float(deficit), 1),
        "prescriptive_mitigation_actions": actions
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)