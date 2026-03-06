#!/usr/bin/env python
"""
SCIENTIFIC METADATA REFINEMENT
Adresses 3 critical issues in dataset_pro/metadata.csv

Issues Fixed:
1. 82% Unknown Crop → Use heuristics + group unknown crops
2. Abiotic Unknown → Separate into 6 valid biological classes
3. Class imbalance → Add class weights + reduce low-freq classes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path('.')
INPUT_CSV = ROOT / 'dataset_pro' / 'metadata.csv'
OUTPUT_CSV = ROOT / 'dataset_pro' / 'metadata_refined.csv'
CLASS_WEIGHTS_JSON = ROOT / 'dataset_pro' / 'class_weights.json'

print("="*80)
print("SCIENTIFIC METADATA REFINEMENT")
print("="*80)

# Load original metadata
df = pd.read_csv(INPUT_CSV)
print(f"\nOriginal: {len(df)} images")

# ============================================================================
# PROBLEM 1: ABIOTIC UNKNOWN → Separate into 6 biological classes
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 1: Abiotic Classification")
print("="*80)

# Extract abiotic indicators from filename
def classify_abiotic(filename, image_path):
    """Separate abiotic unknown into 6 valid classes."""
    fname_lower = filename.lower()
    path_lower = image_path.lower()
    
    # Nutrient deficiency indicators
    if any(x in fname_lower for x in ['deficiency', 'chlorosis', 'yellowing', 'nitrogen', 'iron', 'magnesium']):
        return 'Nutrient Deficiency'
    
    # Drought stress
    if any(x in fname_lower for x in ['drought', 'wilt', 'dry', 'wilting', 'desiccation']):
        return 'Drought Stress'
    
    # Salinity damage
    if any(x in fname_lower for x in ['salt', 'salinity', 'halophyte']):
        return 'Salinity Stress'
    
    # Sunburn/temperature
    if any(x in fname_lower for x in ['sunburn', 'sun', 'frost', 'chilling', 'temperature']):
        return 'Temperature/Sun Damage'
    
    # Mechanical damage
    if any(x in fname_lower for x in ['mechanical', 'bruise', 'wound', 'broken', 'damaged']):
        return 'Mechanical Damage'
    
    # Chemical injury (pesticide, herbicide)
    if any(x in fname_lower for x in ['chemical', 'herbicide', 'pesticide', 'phytotoxic']):
        return 'Chemical Injury'
    
    # Default: Physiological disorder (unspecified)
    return 'Physiological Disorder'

# Refine abiotic class
abiotic_mask = df['agent_type'] == 'Abiotic'
df.loc[abiotic_mask, 'refined_abiotic_class'] = df[abiotic_mask].apply(
    lambda row: classify_abiotic(row['filename'], row['image_path']), 
    axis=1
)

print(f"\nAbiotic breakdown ({abiotic_mask.sum()} images):")
print(df[abiotic_mask]['refined_abiotic_class'].value_counts())

# ============================================================================
# PROBLEM 2: CROP UNKNOWN (82%) → Better extraction + standardization
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 2: Crop Classification")
print("="*80)

CROP_MAPPING = {
    'tomato': 'Tomato',
    'bean': 'Bean',
    'citrus': 'Citrus',
    'cucumber': 'Cucumber',
    'banana': 'Banana',
    'grape': 'Grape',
    'corn': 'Corn',
    'maize': 'Maize',
    'rice': 'Rice',
    'wheat': 'Wheat',
    'potato': 'Potato',
    'carrot': 'Carrot',
    'cabbage': 'Cabbage',
    'lettuce': 'Lettuce',
    'pepper': 'Pepper',
    'eggplant': 'Eggplant',
    'spinach': 'Spinach',
    'broccoli': 'Broccoli',
    'cassava': 'Cassava',
    'soybean': 'Soybean',
}

def extract_crop_improved(filename, crop_current):
    """Better crop extraction with fallback strategy."""
    fname_lower = filename.lower().replace('_', ' ')
    
    # If already identified, trust it
    if crop_current != 'Unknown':
        return crop_current
    
    # Try to extract from filename
    for keyword, crop_name in CROP_MAPPING.items():
        if keyword in fname_lower:
            return crop_name
    
    # Return Unknown if still not found
    return 'Unknown'

df['crop_refined'] = df.apply(
    lambda row: extract_crop_improved(row['filename'], row['crop']),
    axis=1
)

known_crops = df[df['crop_refined'] != 'Unknown']
unknown_crops = df[df['crop_refined'] == 'Unknown']

print(f"\nCrop coverage:")
print(f"  Known:   {len(known_crops):6d} ({100*len(known_crops)/len(df):5.1f}%)")
print(f"  Unknown: {len(unknown_crops):6d} ({100*len(unknown_crops)/len(df):5.1f}%)")
print(f"\nTop crops:")
print(df[df['crop_refined'] != 'Unknown']['crop_refined'].value_counts().head(10))

# ============================================================================
# PROBLEM 3: CLASS IMBALANCE → Reduce low-frequency classes + compute weights
# ============================================================================

print("\n" + "="*80)
print("PROBLEM 3: Class Imbalance & Reduction")
print("="*80)

# Agent type distribution
agent_counts = df['agent_type'].value_counts()
print(f"\nAgent Type distribution:")
print(agent_counts)
print(f"Ratio (max/min): {agent_counts.max() / agent_counts.min():.1f}x imbalance")

# Symptom type - REDUCE TO VALID APS CLASSES
APS_SYMPTOM_TYPES = {
    'Leaf spot': ['spot'],
    'Rot': ['rot'],
    'Wilt': ['wilt'],
    'Blight': ['blight'],
    'Rust': ['rust'],
    'Mildew': ['mildew'],
    'Mosaic': ['mosaic'],
    'Curl': ['curl'],
    'Canker': ['canker'],
    'Necrosis': ['necrosis'],
    'Chlorosis': ['chlorosis'],
    'Deformation': ['deformation', 'gall'],
    'General Damage': ['chewing', 'sucking', 'general'],
    'Unknown': ['unknown', 'specific', 'streak', 'blast', 'mold', 'scab'],
}

def map_to_aps_symptom(symptom_type):
    """Map symptoms to standardized APS classes."""
    symptom_lower = symptom_type.lower()
    
    for aps_class, keywords in APS_SYMPTOM_TYPES.items():
        if any(kw in symptom_lower for kw in keywords):
            return aps_class
    
    return 'Unknown'

df['symptom_type_refined'] = df['symptom_type'].apply(map_to_aps_symptom)

print(f"\nSymptom Type (refined to APS standard):")
symptom_counts = df['symptom_type_refined'].value_counts()
print(symptom_counts)

# Pathogen classes - REDUCE to avoid sparse classes
# Only keep classes with ≥ 20 images
pathogen_counts = df['pathogen_name'].value_counts()
pathogen_threshold = 20
frequent_pathogens = set(pathogen_counts[pathogen_counts >= pathogen_threshold].index)

print(f"\nPathogen classes:")
print(f"  Total unique: {len(pathogen_counts)}")
print(f"  Keep (≥{pathogen_threshold} images): {len(frequent_pathogens)}")
print(f"  Merge (< {pathogen_threshold}): {len(pathogen_counts) - len(frequent_pathogens)}")
print(f"\nTop pathogens (keeping):")
print(pathogen_counts[pathogen_counts >= pathogen_threshold].head(15))

df['pathogen_name_refined'] = df['pathogen_name'].apply(
    lambda x: x if x in frequent_pathogens else 'Other'
)

# ============================================================================
# COMPUTE CLASS WEIGHTS
# ============================================================================

print("\n" + "="*80)
print("CLASS WEIGHTS (for training)")
print("="*80)

def compute_class_weights(labels):
    """Compute inverse frequency weights."""
    counts = labels.value_counts()
    total = len(labels)
    weights = {}
    for class_name, count in counts.items():
        # Inverse frequency: rare classes get higher weight
        weights[class_name] = total / (len(counts) * count)
    return weights

# Compute weights for each task
agent_weights = compute_class_weights(df['agent_type'])
symptom_weights = compute_class_weights(df['symptom_type_refined'])
crop_weights = compute_class_weights(df['crop_refined'])

print("\nAgent Type weights:")
for agent, weight in sorted(agent_weights.items(), key=lambda x: -x[1]):
    count = len(df[df['agent_type'] == agent])
    print(f"  {agent:15} weight={weight:.3f} (n={count:5d})")

print("\nSymptom Type weights (refined):")
for symptom, weight in sorted(symptom_weights.items(), key=lambda x: -x[1]):
    count = len(df[df['symptom_type_refined'] == symptom])
    print(f"  {symptom:20} weight={weight:.3f} (n={count:5d})")

print("\nCrop weights (first 10):")
for crop, weight in sorted(crop_weights.items(), key=lambda x: -x[1])[:10]:
    count = len(df[df['crop_refined'] == crop])
    print(f"  {crop:20} weight={weight:.3f} (n={count:5d})")

# ============================================================================
# CREATE REFINED METADATA
# ============================================================================

print("\n" + "="*80)
print("GENERATING REFINED METADATA")
print("="*80)

# Build final dataframe
df_refined = pd.DataFrame({
    'image_id': df['image_id'],
    'filename': df['filename'],
    'image_path': df['image_path'],
    'agent_type': df['agent_type'],
    'symptom_type': df['symptom_type_refined'],
    'pathogen_name': df['pathogen_name_refined'],
    'crop': df['crop_refined'],
    'plant_part': df['plant_part'],
})

# Add abiotic subclass for abiotic images
df_refined['abiotic_class'] = df.get('refined_abiotic_class', pd.Series([''] * len(df)))

# Save refined metadata
df_refined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"\n✅ Refined metadata saved: {OUTPUT_CSV}")

# Save class weights as JSON
import json

weights_dict = {
    'agent_type': {str(k): float(v) for k, v in agent_weights.items()},
    'symptom_type': {str(k): float(v) for k, v in symptom_weights.items()},
    'crop': {str(k): float(v) for k, v in crop_weights.items()},
}

with open(CLASS_WEIGHTS_JSON, 'w') as f:
    json.dump(weights_dict, f, indent=2)
print(f"✅ Class weights saved: {CLASS_WEIGHTS_JSON}")

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("REFINED DATASET SUMMARY")
print("="*80)

print(f"\nTotal images: {len(df_refined)}")
print(f"\nAgent Type distribution:")
print(df_refined['agent_type'].value_counts())

print(f"\nSymptom Type distribution (APS standardized):")
print(df_refined['symptom_type'].value_counts())

print(f"\nCrop distribution (improved):")
print(df_refined['crop'].value_counts().head(15))

print(f"\nPathogen distribution (sparse classes merged):")
print(df_refined['pathogen_name'].value_counts().head(15))

if (df_refined['agent_type'] == 'Abiotic').any():
    print(f"\nAbiotic subclass distribution:")
    print(df_refined[df_refined['agent_type'] == 'Abiotic']['abiotic_class'].value_counts())

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "="*80)
print("VALIDATION")
print("="*80)

print(f"\nMissing values:")
print(df_refined.isnull().sum())

print(f"\nClass balance improvement:")
agent_ratio = df['agent_type'].value_counts().max() / df['agent_type'].value_counts().min()
pathogen_ratio = df['pathogen_name'].value_counts().max() / df['pathogen_name'].value_counts().min()
print(f"  Agent type imbalance (before): ∞ (6 balanced classes)")
print(f"  Pathogen imbalance (before): {pathogen_ratio:.1f}x")
pathogen_ratio_after = df_refined['pathogen_name'].value_counts().max() / df_refined['pathogen_name'].value_counts().min()
print(f"  Pathogen imbalance (after): {pathogen_ratio_after:.1f}x (merged sparse classes)")

print(f"\nCrop Unknown reduction:")
unknown_before = (df['crop'] == 'Unknown').sum()
unknown_after = (df_refined['crop'] == 'Unknown').sum()
print(f"  Before: {unknown_before:5d} ({100*unknown_before/len(df):.1f}%)")
print(f"  After:  {unknown_after:5d} ({100*unknown_after/len(df):.1f}%)")
print(f"  Improvement: {unknown_before - unknown_after:5d} images reclassified")

print("\n" + "="*80)
print("✅ SCIENTIFIC REFINEMENT COMPLETE")
print("="*80)
print("\nNext: Use dataset_pro/metadata_refined.csv for multi-task training")
print("      with class_weights loaded from class_weights.json")
