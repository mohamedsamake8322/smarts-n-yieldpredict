import os
import json
import re
import pandas as pd

# 📁 Dossier contenant les fichiers
DATA_FOLDER = r"C:\smarts-n-yieldpredict.git"
JSON_PATH = os.path.join(DATA_FOLDER, "vectorized_dataset.json")
PARQUET_PATH = os.path.join(DATA_FOLDER, "vectorized_dataset.parquet")

# 🧼 Fonction de nettoyage du texte
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\bPage\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(Figure\s*\d+.*|Table\s*\d+.*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# 🔄 Nettoyage du JSON
def clean_json_file(path):
    print(f"\n🧼 Nettoyage du fichier JSON : {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        entry["text"] = clean_text(entry.get("text", ""))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON nettoyé et sauvegardé.")

# 🔄 Nettoyage du Parquet
def clean_parquet_file(path):
    print(f"\n🧼 Nettoyage du fichier Parquet : {path}")
    df = pd.read_parquet(path)
    df["text"] = df["text"].apply(clean_text)
    df.to_parquet(path, index=False)
    print(f"✅ Parquet nettoyé et sauvegardé.")

# 🚀 Exécution
clean_json_file(JSON_PATH)
clean_parquet_file(PARQUET_PATH)
print("\n🎉 Nettoyage terminé pour les deux fichiers.")
