import pandas as pd
import numpy as np
import os

def load_and_merge_indicators(
    base_path=r"C:\smarts-n-yieldpredict.git\data",
    modis_file="MODIS_VI_Mali_2020_2025_mali_20250821_1503.csv",
    spei_file="SPEI_Mali_ADM2_20250821_1546.csv",
    soil_file="fusion_SMAP_SoilGrids.csv"
):
    # Construct full paths
    modis_path = os.path.join(base_path, modis_file)
    spei_path = os.path.join(base_path, spei_file)
    soil_path = os.path.join(base_path, soil_file)

    # Load datasets
    modis = pd.read_csv(modis_path)
    spei = pd.read_csv(spei_path)
    soil = pd.read_csv(soil_path)

    # Harmonize ADM2_ID and date
    for df in [modis, spei]:
        df['ADM2_ID'] = df['ADM2_ID'].astype(str)
        df['date'] = pd.to_datetime(df[['year', 'month']].astype(str).agg('-'.join, axis=1), errors='coerce')

    # Merge MODIS + SPEI on ADM2_ID + date
    merged = pd.merge(modis, spei, on=['ADM2_ID', 'date'], how='outer')

    # Merge soil data (static, no date) on ADM2_ID
    soil['ADM2_ID'] = soil['ADM2_ID'].astype(str)
    merged = pd.merge(merged, soil, on='ADM2_ID', how='left')

    # QA: Drop rows with invalid ADM2_ID or missing date
    merged = merged.dropna(subset=['ADM2_ID', 'date'])

    # QA: Rename key columns for clarity
    merged.rename(columns={
        'NDVI': 'ndvi',
        'EVI': 'evi',
        'SPEI_03': 'spei_03',
        'SPEI_06': 'spei_06',
        'SPEI_12': 'spei_12',
        'SMAP_moisture': 'smap_moisture',
        'sand': 'soil_sand',
        'clay': 'soil_clay',
        'silt': 'soil_silt',
        'pH': 'soil_ph'
    }, inplace=True)

    # QA: Add missing flags
    merged['ndvi_missing'] = merged['ndvi'].isnull()
    merged['smap_missing'] = merged['smap_moisture'].isnull()
    merged['spei_missing'] = merged['spei_03'].isnull()

    # Optional: Fill missing values with interpolation
    merged.sort_values(['ADM2_ID', 'date'], inplace=True)
    merged[['ndvi', 'evi']] = merged.groupby('ADM2_ID')[['ndvi', 'evi']].transform(lambda x: x.interpolate(limit_direction='forward'))

    return merged
