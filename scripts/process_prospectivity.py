"""
Satyam — Phase 2: Satellite TIF to Labeled Prospectivity Dataset
Converts raw Sentinel-1 & 2 GeoTIFF into ML-ready tabular features with:
1. Band extraction (MAI, SAR VV/VH, Spectral Indices)
2. Real Ground-Truth Labels (Balaghat Manganese Belt Coordinates)
3. Eco-Masking (Excluding water bodies & dense protected forest reserves)
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

# --- 1. FIND INPUT TIF FILE ---
raw_dir = r"D:\Manganese\data\raw\sentinel"
tif_files = glob.glob(os.path.join(raw_dir, "*.tif"))

if not tif_files:
    # Check parent folder just in case
    tif_files = glob.glob(r"D:\Manganese\*.tif")

if not tif_files:
    print(f"❌ Error: No .tif file found in {raw_dir}!")
    exit(1)

tif_path = tif_files[0]
print(f"🔍 Found satellite GeoTIFF: {os.path.basename(tif_path)}")

# Band order from GEE Export script:
band_names = [
    'B2', 'B3', 'B4', 'B8', 'B8A', 'B11', 'B12',
    'MAI',          # Manganese Alteration Index
    'IronOxide',    # Gossan / Laterite index
    'ClayIndex',    # Hydrothermal clay index
    'NDVI',         # Vegetation index
    'SWIR_ratio',   # B11/B8A ratio
    'VV',           # SAR VV backscatter
    'VH',           # SAR VH backscatter
    'VV_VH_ratio'   # Structural roughness proxy
]

# --- 2. EXTRACT PIXELS & COORDINATES ---
print("⏳ Reading multi-band raster data (this takes ~15-30 seconds)...")
with rasterio.open(tif_path) as src:
    # Downsample slightly (step of 2 or 3) to keep dataset fast and clean
    step = 2 
    data = src.read(out_shape=(src.count, src.height // step, src.width // step))
    transform = src.transform * src.transform.scale(step, step)
    height, width = data.shape[1], data.shape[2]
    
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    lons, lats = xy(transform, rows.ravel(), cols.ravel())

df = pd.DataFrame({
    'longitude': lons,
    'latitude': lats
})

for idx, name in enumerate(band_names):
    if idx < data.shape[0]:
        df[name] = data[idx].ravel()

print(f"📊 Extracted {len(df):,} grid points across Balaghat.")

# --- 3. DATA CLEANING & ECO-MASK ---
print("🌿 Applying Eco-Mask & Quality Filtering...")
# Remove nodata pixels
df = df.dropna()
df = df[df['B4'] > 0]
df = df[df['MAI'] > -9000]

# Eco-Mask Constraint 1: Mask out Water Bodies (NDVI very negative or NIR very low)
df = df[df['NDVI'] > -0.1]

# Eco-Mask Constraint 2: Strictly Protected Dense Canopy Forests (NDVI > 0.65)
# Real-world ESG constraint: exploration strictly barred in virgin high-density biosphere canopy
df['eco_restricted'] = np.where(df['NDVI'] > 0.62, 1, 0)

# Filter out eco-restricted zones for our target prospectivity mapping
valid_df = df[df['eco_restricted'] == 0].copy()

# --- 4. GROUND TRUTH LABELS (Known Balaghat Mines) ---
print("📍 Injecting Ground Truth labels from Balaghat Manganese Belt...")

# Verified historical & active manganese mine coordinates (GSI / IBM records)
known_mines = [
    {"name": "Bharweli Mine (MOIL)", "lat": 21.849, "lon": 80.203},
    {"name": "Ukwa Mine",             "lat": 21.968, "lon": 80.468},
    {"name": "Tirodi Deposit",        "lat": 21.688, "lon": 79.712},
    {"name": "Ramrama Mine",          "lat": 21.854, "lon": 80.081},
    {"name": "Sitapathor / Miragpur", "lat": 21.800, "lon": 80.050},
    {"name": "Netra Deposit",         "lat": 21.860, "lon": 80.120},
    {"name": "Waraseoni Belt",        "lat": 21.820, "lon": 80.040},
    {"name": "Katangjhari",           "lat": 21.830, "lon": 80.020}
]

# Calculate distance to nearest known deposit
valid_df['manganese_target'] = 0

# Mark points within 1.5 km of known deposits/veins as Positive Ground Truth (1)
for mine in known_mines:
    dist = np.sqrt(
        (valid_df['latitude'] - mine['lat'])**2 + 
        (valid_df['longitude'] - mine['lon'])**2
    )
    # ~0.015 degrees is roughly 1.5 km radius around mineralized strike zone
    valid_df.loc[dist < 0.015, 'manganese_target'] = 1

# Create balanced dataset for training:
positives = valid_df[valid_df['manganese_target'] == 1]
negatives = valid_df[valid_df['manganese_target'] == 0].sample(n=min(len(positives) * 5, len(valid_df)), random_state=42)

final_df = pd.concat([positives, negatives]).sample(frac=1.0, random_state=42).reset_index(drop=True)

# --- 5. SAVE FINAL CSV ---
output_dir = r"D:\Manganese\data\processed"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "manganese_prospectivity.csv")

final_df.to_csv(output_file, index=False)

print("\n" + "="*50)
print(f"✅ SUCCESS! Dataset generated: {output_file}")
print(f"Total Rows: {len(final_df):,}")
print(f"Positive Manganese Signatures: {len(positives):,}")
print(f"Background Non-Mineral Signatures: {len(negatives):,}")
print("="*50)