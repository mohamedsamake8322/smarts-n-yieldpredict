import csv
import pandas as pd

csv_path = r"C:\Users\moham\Videos\dataset_map.csv"

print("🔍 DIAGNOSTIC DU CSV\n")

# Essayer de lire ligne par ligne pour identifier le problème
print("Lecture ligne par ligne:")
with open(csv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Compter les virgules
        comma_count = line.count(',')
        print(f"Ligne {i}: {comma_count} virgules - {line[:80]}...")

        if i == 39:
            print(f"  → PROBLÈME À LA LIGNE 39: {line}")
            fields = line.split(',')
            print(f"  → Champs détectés: {len(fields)}")
            for j, field in enumerate(fields):
                print(f"    Champ {j+1}: '{field}'")

# Essayer différentes options de lecture pandas
print("\n" + "="*50)
print("TESTS DE LECTURE PANDAS:")

try:
    # Test 1: Lecture normale
    df1 = pd.read_csv(csv_path)
    print("✅ Lecture normale: OK")
except Exception as e:
    print(f"❌ Lecture normale: {e}")

try:
    # Test 2: Avec sep=','
    df2 = pd.read_csv(csv_path, sep=',')
    print("✅ Avec sep=',': OK")
except Exception as e:
    print(f"❌ Avec sep=',': {e}")

try:
    # Test 3: Avec quoting
    df3 = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL)
    print("✅ Avec quoting: OK")
except Exception as e:
    print(f"❌ Avec quoting: {e}")

try:
    # Test 4: Avec engine='python'
    df4 = pd.read_csv(csv_path, engine='python')
    print("✅ Avec engine='python': OK")
except Exception as e:
    print(f"❌ Avec engine='python': {e}")

# Créer version corrigée
print("\n" + "="*50)
print("CRÉATION VERSION CORRIGÉE:")

corrected_lines = []
with open(csv_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Nettoyer les caractères problématiques
        line = line.replace('*', '').replace('(', '').replace(')', '')

        # S'assurer qu'il n'y a que 2 champs
        parts = line.split(',')
        if len(parts) > 2:
            # Fusionner les parties supplémentaires
            parts = [parts[0], ','.join(parts[1:])]

        corrected_line = ','.join(parts)
        corrected_lines.append(corrected_line)

# Sauvegarder la version corrigée
corrected_path = csv_path.replace('.csv', '_corrected.csv')
with open(corrected_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(corrected_lines))

print(f"✅ Version corrigée sauvegardée: {corrected_path}")

# Tester la version corrigée
try:
    df_corrected = pd.read_csv(corrected_path)
    print(f"✅ Version corrigée lisible: {len(df_corrected)} lignes")
    print("Colonnes:", list(df_corrected.columns))
    print("Aperçu:")
    print(df_corrected.head())
except Exception as e:
    print(f"❌ Version corrigée encore problématique: {e}")