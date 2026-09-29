"""
Satyam — Phase 2: Mine Operational Telemetry Dataset
Generates mine_production_risk.csv based on real-world mining constraints:
- Monsoon rainfall penalties
- Dump truck fleet availability
- Excavator operational hours
- Risk classification (Normal / Moderate / Critical Shortfall)
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Generate 3 years of daily mine operations
dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
n_days = len(dates)
months = dates.month

# 1. Weather: Monsoon rainfall (June - September)
rainfall = np.where((months >= 6) & (months <= 9), 
                    np.random.exponential(scale=18.0, size=n_days), 
                    np.random.exponential(scale=1.5, size=n_days))

# 2. Fleet: Active dump trucks (Base 25 trucks)
active_trucks = 25 - np.random.poisson(lam=1.8, size=n_days)
active_trucks = np.clip(active_trucks, 12, 25)

# 3. Heavy Machinery: Excavator hours per day (Max 18 hours)
excavator_hours = 18 - (rainfall / 15) - np.random.normal(0, 1.2, n_days)
excavator_hours = np.clip(excavator_hours, 2.0, 18.0)

# 4. Daily Production (Tons)
base_prod = (active_trucks * 18) + (excavator_hours * 30)
flood_penalty = np.where(rainfall > 25, rainfall * 3.2, 0)
production_tons = base_prod - flood_penalty + np.random.normal(0, 20, n_days)
production_tons = np.clip(production_tons, 50, None)

# 5. Shortfall Risk Target Label:
# Quota = 850 tons/day
# 0 = Normal (>750T), 1 = Moderate Risk (550-750T), 2 = High Risk (<550T)
risk_class = np.where(production_tons < 550, 2,
             np.where(production_tons < 750, 1, 0))

df = pd.DataFrame({
    'date': dates,
    'rainfall_mm': np.round(rainfall, 2),
    'active_trucks': active_trucks,
    'excavator_hours': np.round(excavator_hours, 1),
    'daily_production_tons': np.round(production_tons, 1),
    'shortfall_risk_class': risk_class
})

output_dir = r"D:\Manganese\data\processed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "mine_production_risk.csv")
df.to_csv(output_path, index=False)

print("\n" + "="*50)
print(f"✅ SUCCESS! Generated: {output_path}")
print(f"Total Operational Days: {len(df):,}")
print("Risk Class Breakdown:")
print("  0 (Normal):", (df['shortfall_risk_class'] == 0).sum())
print("  1 (Moderate Shortfall):", (df['shortfall_risk_class'] == 1).sum())
print("  2 (Severe Shortfall):", (df['shortfall_risk_class'] == 2).sum())
print("="*50)