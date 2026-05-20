from pathlib import Path
import csv

# Chemin du dataset à renommer
DATASET_ROOT = Path(r"C:\Downloads\Lychee Disease Augmentation Dataset")
# Chemin du mapping CSV créé
MAPPING_CSV = Path(__file__).with_name("lychee_rename_mapping.csv")

if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset root introuvable: {DATASET_ROOT}")

if not MAPPING_CSV.exists():
    raise FileNotFoundError(f"Mapping CSV introuvable: {MAPPING_CSV}")

with MAPPING_CSV.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Dataset root: {DATASET_ROOT}")
print(f"Mapping CSV: {MAPPING_CSV}")
print(f"Mappings à appliquer: {len(rows)}\n")

renamed = 0
skipped = 0
for row in rows:
    old_name = row["original_name"].strip()
    new_name = row["new_name"].strip()
    old_path = DATASET_ROOT / old_name
    new_path = DATASET_ROOT / new_name

    if not old_path.exists():
        print(f"⚠️  Source introuvable, ignoré: {old_name}")
        skipped += 1
        continue
    if new_path.exists():
        print(f"⚠️  Destination existe déjà, ignoré: {new_name}")
        skipped += 1
        continue

    old_path.rename(new_path)
    print(f"✅  Renommé: {old_name} -> {new_name}")
    renamed += 1

print(f"\nTerminé: {renamed} renommages effectués, {skipped} ignorés.")
