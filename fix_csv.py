import csv

csv_path = r"C:\Users\moham\Videos\dataset_map.csv"
corrected_path = r"C:\Users\moham\Videos\dataset_map_fixed.csv"

print("🔧 CORRECTION DU CSV AVEC GUILLEMETS\n")

# Lire et corriger le CSV
corrected_lines = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            # Mettre des guillemets autour des champs qui contiennent des virgules
            original_name = row[0]
            new_name = row[1]

            # Si le nom original contient une virgule, l'encadrer de guillemets
            if ',' in original_name:
                original_name = f'"{original_name}"'

            corrected_line = f"{original_name},{new_name}"
            corrected_lines.append(corrected_line)

# Sauvegarder la version corrigée
with open(corrected_path, 'w', encoding='utf-8', newline='') as f:
    f.write('\n'.join(corrected_lines))

print(f"✅ Version corrigée sauvegardée: {corrected_path}")

# Tester la lecture
import pandas as pd
try:
    df = pd.read_csv(corrected_path, quoting=csv.QUOTE_MINIMAL)
    print(f"✅ Lecture réussie: {len(df)} lignes")
    print("Aperçu:")
    print(df.head(10))

    # Sauvegarder le mapping pour utilisation
    mapping = dict(zip(df["original_name"], df["new_name"]))
    print(f"\n📋 Mapping créé avec {len(mapping)} entrées")

except Exception as e:
    print(f"❌ Erreur de lecture: {e}")

print("\n💡 Utilisation:")
print("Remplace dans ton script 1.py:")
print(f'csv_path = r"{corrected_path}"')
print('df = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL)')