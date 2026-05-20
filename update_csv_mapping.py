import pandas as pd
import csv

# Charger le CSV existant
csv_path = r"C:\Users\moham\Videos\dataset_map_fixed.csv"
df = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL)

print("🔄 MISE À JOUR DU MAPPING CSV\n")

# Ajouter les nouveaux mappings pour les noms nettoyés
additional_mappings = {
    "Cherry_including_sour_healthy": "Cherry_Healthy",
    "Cherry_including_sour_Powdery_mildew": "Cherry_Powdery_Mildew",
    "Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot": "Corn_Gray_Leaf_Spot",
    "Corn_maize_Common_rust": "Corn_Common_Rust",
    "Corn_maize_healthy": "Corn_Healthy",
    "Corn_maize_Northern_Leaf_Blight": "Corn_Northern_Leaf_Blight",
    "Grape_Esca_Black_Measles": "Grape_Esca",
    "Grape_Leaf_blight_Isariopsis_Leaf_Spot": "Grape_Leaf_Blight",
    "Orange_Haunglongbing_Citrus_greening": "Orange_Citrus_Greening",
    '"Pepper,_bell_Bacterial_spot"': "Pepper_Bacterial_Spot",
    '"Pepper,_bell_healthy"': "Pepper_Healthy",
    "Tomato_healthy": "Tomato_Healthy"
}

# Ajouter les nouvelles entrées
new_rows = []
for original, new in additional_mappings.items():
    new_rows.append({"original_name": original, "new_name": new})

# Créer un nouveau DataFrame avec les ajouts
df_new = pd.DataFrame(new_rows)
df_updated = pd.concat([df, df_new], ignore_index=True)

# Sauvegarder
updated_csv_path = r"C:\Users\moham\Videos\dataset_map_updated.csv"
df_updated.to_csv(updated_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"✅ CSV mis à jour sauvegardé: {updated_csv_path}")
print(f"Total mappings: {len(df_updated)}")
print("\nNouveaux mappings ajoutés:")
for original, new in additional_mappings.items():
    print(f"  {original} → {new}")