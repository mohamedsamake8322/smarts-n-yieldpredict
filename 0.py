import os

dataset_path = r"C:\smarts-n-yieldpredict.git\dataset_final"
json_path = r"C:\Downloads\BLIP2"

# Dossiers du dataset
dataset_folders = {
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
}

# Fichiers json existants
json_files = {
    os.path.splitext(f)[0] for f in os.listdir(json_path)
    if f.endswith(".json")
}

# JSON manquants
missing = dataset_folders - json_files

print("Nombre dossiers dataset :", len(dataset_folders))
print("Nombre json :", len(json_files))

print("\nFICHIERS JSON MANQUANTS :\n")

for m in sorted(missing):
    print(f"{m}.json")

print("\nTotal manquant :", len(missing))