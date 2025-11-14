.\env311\Scripts\Activate.ps1

script pour connaitre les noms des Fichiers d'un dossier

import os

# Chemin du dossier

folder_path = r"C:\plateforme-agricole-complete-v2\pages"

# Vérifie si le dossier existe

if os.path.exists(folder_path):
print(f"Contenu du dossier : {folder_path}\n")
for item in os.listdir(folder_path):
item_path = os.path.join(folder_path, item)
if os.path.isfile(item_path):
print(f"📄 Fichier : {item}")
elif os.path.isdir(item_path):
print(f"📁 Dossier : {item}")
else:
print("❌ Le dossier spécifié n'existe pas.")

Pour vérier NaNimport pandas as pd
import os

# 📂 Dossier contenant les fichiers

folder = r"C:\Users\moham\Music\faostat"

# Liste des fichiers à vérifier

files = [
"FAOSTAT_data_en_8-21-2025 (1).csv",
"FAOSTAT_data_en_8-21-2025 (2).csv",
"FAOSTAT_data_en_8-21-2025 (3).csv",
"FAOSTAT_data_en_8-21-2025 (4).csv",
"FAOSTAT_data_en_8-21-2025.csv"
]

print(f"[INFO] Vérification des NaN dans {len(files)} fichiers...")

for f in files:
path = os.path.join(folder, f)
try:
df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
total_rows = len(df)
missing = df.replace({"": pd.NA}).isna().sum()
total_missing = missing.sum()
print(f"\n=== {f} ===")
print(f"Lignes totales : {total_rows}")
print(f"Valeurs manquantes totales : {total_missing}")
print("Colonnes avec NaN :")
print(missing[missing>0].sort_values(ascending=False))
except Exception as e:
print(f"[ERREUR] Impossible de lire {f} : {e}")

# Pour recoonaitre la structre

import os
import pandas as pd

# 📁 Dossier contenant les fichiers CSV

folder_path = r"C:\Users\moham\Music\Moh"

# 📊 Dictionnaire pour stocker les colonnes par fichier

schema_dict = {}

# 🔍 Parcours des fichiers

for filename in os.listdir(folder_path):
if filename.endswith(".csv"):
file_path = os.path.join(folder_path, filename)
try:
df = pd.read_csv(file_path, nrows=5) # Lecture rapide
schema_dict[filename] = list(df.columns)
except Exception as e:
schema_dict[filename] = [f"Erreur de lecture: {e}"]

# 📋 Analyse des colonnes

all_columns = set()
for cols in schema_dict.values():
if isinstance(cols, list):
all_columns.update(cols)

# 🧠 Rapport fusion

report_lines = []
report_lines.append("🧾 Rapport de colonnes pour fusion\n")
report_lines.append(f"📁 Dossier analysé : {folder_path}\n")
report_lines.append("📦 Colonnes par fichier :\n")

for file, cols in schema_dict.items():
report_lines.append(f"\n➡️ {file} :")
if isinstance(cols, list):
for col in cols:
report_lines.append(f" - {col}")
else:
report_lines.append(f" ⚠️ {cols}")

report_lines.append("\n🧮 Colonnes totales détectées :")
for col in sorted(all_columns):
report_lines.append(f" - {col}")

# 📝 Sauvegarde du rapport

report_path = os.path.join(folder_path, "fusion_schema_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
f.write("\n".join(report_lines))

print(f"✅ Rapport généré : {report_path}")

import pandas as pd

# Charger le fichier CSV

fichier = r"C:\Users\moham\Music\faostat\NDVI_NDMI_Merged_Mali_2021_2024.csv"
df = pd.read_csv(fichier)

# Afficher la structure générale

print("🔹 Colonnes du fichier :")
print(df.columns.tolist())
print("\n🔹 Aperçu des 5 premières lignes :")
print(df.head())

# Nombre de lignes et colonnes

print("\n🔹 Dimensions du fichier :", df.shape)

# Vérification des valeurs manquantes

print("\n🔹 Valeurs manquantes par colonne :")
print(df.isna().sum())

# Statistiques générales

print("\n🔹 Statistiques descriptives :")
print(df.describe(include="all"))

Pour Fusionner les chiers climat et sol

import pandas as pd
import glob
import os

# ----------------------------

# 1️⃣ Charger et fusionner tous les fichiers CSV

# ----------------------------

folder = r"C:\Users\moham\Music\Boua" # chemin vers tes fichiers CSV
csv_files = glob.glob(os.path.join(folder, "\*.csv"))

dfs = []
for file in csv_files:
df = pd.read_csv(file)

    # Extraire le nom du pays à partir du fichier
    country_name = os.path.basename(file).split("_")[-1].replace(".csv","")
    df["country"] = country_name

    dfs.append(df)

# Fusionner tous les DataFrames (union des colonnes)

merged_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

# ----------------------------

# 2️⃣ Supprimer colonnes inutiles

# ----------------------------

cols_to_drop = ["Mangrove2000", ".geo", "system:index"]
merged_df = merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns])

# ----------------------------

# 3️⃣ Gérer les NaN

# ----------------------------

# Option 1 : Remplacer NaN par 0 pour les variables explicatives

merged_df.fillna(0, inplace=True)

# Option 2 (si tu veux ne garder que les lignes où la cible est connue)

# merged_df = merged_df[merged_df["mean"].notna()]

# ----------------------------

# 4️⃣ Préparer les colonnes pour XGBoost

# ----------------------------

# Variable cible

target_col = "mean"

# Colonnes explicatives : toutes sauf cible et pays

feature_cols = [c for c in merged_df.columns if c != target_col and c != "country" and c != "ADM0_NAME" and c != "ADM1_NAME"]

X = merged_df[feature_cols]
y = merged_df[target_col]

# ----------------------------

# 5️⃣ Sauvegarder le dataset final

# ----------------------------

output_file = os.path.join(folder, "Merged_XGBoost_Dataset.csv")
merged_df.to_csv(output_file, index=False)

print(f"✅ Fusion terminée ! Dataset prêt pour XGBoost : {output_file}")
print(f"Dimensions finales : {merged_df.shape}")
print("Aperçu des colonnes :", merged_df.columns.tolist())

Fusuion complète
import pandas as pd
import glob
import os

# Dossier contenant tes CSVs

folder_path = r"C:\Users\moham\Music\Boua"
all_files = glob.glob(os.path.join(folder_path, "\*.csv"))

dfs = []

for file in all_files:
print(f"📄 Lecture du fichier : {file}") # Lecture avec dtype=object pour éviter les conflits et low_memory=False
df = pd.read_csv(file, dtype=object, low_memory=False)

    # Supprimer les colonnes volumineuses ou inutiles
    cols_to_drop = ["Mangrove2000", ".geo", "system:index"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Ajouter le dataframe à la liste
    dfs.append(df)

# Fusionner tous les fichiers

merged_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
print(f"✅ Fusion terminée ! Dimensions : {merged_df.shape}")

# Conversion des colonnes numériques après fusion

non_numeric_cols = ['ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME', 'country'] if 'ADM2_NAME' in merged_df.columns else ['ADM0_NAME', 'ADM1_NAME', 'country']
num_cols = [c for c in merged_df.columns if c not in non_numeric_cols]

merged_df[num_cols] = merged_df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Sauvegarder le dataset fusionné

output_file = os.path.join(folder_path, "Merged_XGBoost_Dataset.csv")
merged_df.to_csv(output_file, index=False)
print(f"✅ Dataset prêt pour XGBoost : {output_file}")

import pandas as pd

# -----------------------------

# Chemins des fichiers

# -----------------------------

ndvi_file = r"C:\Downloads\New folder (2)\NDVI_NDMI_Mali_2021_2024_merged.csv"
faostat_file = r"C:\Downloads\New folder (2)\merged_all_advanced.csv"
smap_file = r"C:\Downloads\New folder (2)\SMAP_SoilMoisture_Mali.csv"
worldclim_bio_file = r"C:\Downloads\New folder (2)\WorldClim BIO MLI Variables V1.csv"
worldclim_monthly_file = r"C:\Downloads\New folder (2)\WorldClim Mali Monthly V1.csv"
chirps_smap_file = r"C:\Downloads\New folder (2)\CHIRPS_SMAP_DAILY_PENTMli.csv"
gedi_file = r"C:\Downloads\New folder (2)\GEDI_Mangrove_CSV_Mali_Single.csv"

# -----------------------------

# Charger les fichiers CSV

# -----------------------------

print("Chargement des fichiers...")
ndvi_df = pd.read_csv(ndvi_file)
faostat_df = pd.read_csv(faostat_file, low_memory=False) # Evite les warnings
smap_df = pd.read_csv(smap_file)
worldclim_bio_df = pd.read_csv(worldclim_bio_file)
worldclim_monthly_df = pd.read_csv(worldclim_monthly_file)
chirps_smap_df = pd.read_csv(chirps_smap_file)
gedi_df = pd.read_csv(gedi_file)

# -----------------------------

# Harmoniser les colonnes clés

# -----------------------------

# FAOSTAT : ajouter ADM0_NAME et renommer Area -> ADM1_NAME

if "ADM0_NAME" not in faostat_df.columns:
faostat_df["ADM0_NAME"] = "Mali"
faostat_df = faostat_df.rename(columns={"Area": "ADM1_NAME", "Year": "Year"})

# NDVI : renommer year -> Year, month -> Month

ndvi_df = ndvi_df.rename(columns={"year": "Year", "month": "Month"})

# SMAP, WorldClim, CHIRPS : EXP1_YEAR -> Year

for df in [smap_df, worldclim_bio_df, worldclim_monthly_df, chirps_smap_df]:
df.rename(columns={"EXP1_YEAR": "Year"}, inplace=True)

# -----------------------------

# Sélection des colonnes utiles pour éviter les doublons

# -----------------------------

# SMAP

smap_df = smap_df[["ADM0_NAME","ADM1_NAME","Year","mean"]]

# WorldClim BIO

worldclim_bio_df = worldclim_bio_df[["ADM0_NAME","ADM1_NAME","Year"] + [f"bio{i:02d}" for i in range(1,20)]]

# WorldClim Monthly

worldclim_monthly_df = worldclim_monthly_df[["ADM0_NAME","ADM1_NAME","Year","prec","tavg","tmax","tmin"]]

# CHIRPS + SMAP

chirps_smap_df = chirps_smap_df[["ADM0_NAME","ADM1_NAME","Year","CHIRPS_Daily","CHIRPS_Pentad","SMAP_SoilMoisture"]]

# GEDI

gedi_df = gedi_df[["ADM0_NAME","ADM1_NAME","GEDI_CanopyHeight","Mangrove2000","TidalWetlands2019"]]

# -----------------------------

# Fusionner les fichiers progressivement

# -----------------------------

print("Fusion NDVI + FAOSTAT...")
df_merged = pd.merge(ndvi_df, faostat_df, how="outer", on=["ADM0_NAME", "ADM1_NAME", "Year"])

print("Fusion avec SMAP...")
df_merged = pd.merge(df_merged, smap_df, how="outer", on=["ADM0_NAME", "ADM1_NAME", "Year"])

print("Fusion avec WorldClim BIO...")
df_merged = pd.merge(df_merged, worldclim_bio_df, how="outer", on=["ADM0_NAME", "ADM1_NAME", "Year"])

print("Fusion avec WorldClim Monthly...")
df_merged = pd.merge(df_merged, worldclim_monthly_df, how="outer", on=["ADM0_NAME", "ADM1_NAME", "Year"])

print("Fusion avec CHIRPS + SMAP...")
df_merged = pd.merge(df_merged, chirps_smap_df, how="outer", on=["ADM0_NAME", "ADM1_NAME", "Year"])

print("Fusion avec GEDI...")
df_merged = pd.merge(df_merged, gedi_df, how="left", on=["ADM0_NAME", "ADM1_NAME"]) # pas d'année

# -----------------------------

# Résumé final

# -----------------------------

print("Fusion terminée !")
print(f"Colonnes finales : {df_merged.columns.tolist()}")
print(f"Nombre de lignes : {len(df_merged)}")

# -----------------------------

# Sauvegarder le dataset global

# -----------------------------

output_file = r"C:\Downloads\New folder (2)\Mali_Global_Dataset.csv"
df_merged.to_csv(output_file, index=False)
print(f"Dataset global sauvegardé : {output_file}")

Nettoyage de FAPSTAT

# eda_faostat.py

# Exploratory Data Analysis (EDA) pour faostat_fusion_clean.csv

# Sorties: CSV de synthèse, graphiques PNG, rapport Markdown

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_data(csv_path: Path) -> pd.DataFrame:
print(f"[INFO] Lecture du fichier: {csv_path}")
df = pd.read_csv(
csv_path,
encoding="utf-8-sig",
low_memory=False,
na_values=["", "NA", "N/A", "NaN", "null", "None", ".."]
) # Normaliser noms de colonnes
df.columns = [c.strip() for c in df.columns]
return df

def ensure_types(df: pd.DataFrame) -> pd.DataFrame: # Année
if "Year" in df.columns:
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Valeur (conserver Value, créer Value_num)
    if "Value" in df.columns:
        # Retirer séparateurs éventuels (espace, virgule)
        val = df["Value"].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        df["Value_num"] = pd.to_numeric(val, errors="coerce")
    else:
        df["Value_num"] = np.nan

    # Codes en texte propre
    for c in ["Area Code", "Item Code", "Element Code", "Domain Code", "Year Code"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df

def make*outdir(base: Path) -> Path:
stamp = datetime.now().strftime("%Y%m%d*%H%M%S")
outdir = base / f"faostat*eda_report*{stamp}"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "figs").mkdir(exist_ok=True)
(outdir / "tables").mkdir(exist_ok=True)
return outdir

def overview(df: pd.DataFrame) -> dict:
info = {
"n_rows": int(df.shape[0]),
"n_cols": int(df.shape[1]),
"columns": df.columns.tolist(),
"dtypes": df.dtypes.astype(str).to_dict(),
}
return info

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
miss = df.isna().sum().rename("missing_count").to_frame()
miss["missing_pct"] = (miss["missing_count"] / len(df)) \* 100
miss = miss.sort_values("missing_pct", ascending=False)
return miss.reset_index().rename(columns={"index": "column"})

def duplicate_summary(df: pd.DataFrame, key_cols=None) -> dict:
out = {}
out["total_duplicate_rows"] = int(df.duplicated().sum())
if key_cols and all(k in df.columns for k in key_cols): # Duplicats sur la clé (plusieurs lignes partagent la même clé)
key_dups = df.duplicated(subset=key_cols, keep=False)
out["duplicate_key_rows"] = int(key_dups.sum()) # Comptage des clés en double
if key_dups.any():
dup_keys_df = (
df.loc[key_dups, key_cols]
.value_counts()
.rename("count")
.reset_index()
.sort_values("count", ascending=False)
)
else:
dup_keys_df = pd.DataFrame(columns=key_cols + ["count"])
else:
out["duplicate_key_rows"] = None
dup_keys_df = pd.DataFrame(columns=(key_cols or []) + ["count"])
return out, dup_keys_df

def counts_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
if col not in df.columns:
return pd.DataFrame(columns=[col, "n_rows"])
s = df[col].fillna("NA")
return s.value_counts().rename("n_rows").reset_index().rename(columns={"index": col})

def year_coverage(df: pd.DataFrame) -> pd.DataFrame:
if "Year" not in df.columns:
return pd.DataFrame(columns=["Year", "n_rows"])
return df["Year"].dropna().astype(int).value_counts().sort_index().rename("n_rows").reset_index().rename(columns={"index": "Year"})

def flag_summary(df: pd.DataFrame) -> pd.DataFrame:
if "Flag" not in df.columns:
return pd.DataFrame(columns=["Flag", "Flag Description", "n_rows"])
fd = "Flag Description" if "Flag Description" in df.columns else None
if fd:
tbl = df.groupby(["Flag", fd]).size().rename("n_rows").reset_index().sort_values("n_rows", ascending=False)
else:
tbl = df["Flag"].value_counts().rename("n_rows").reset_index().rename(columns={"index": "Flag"})
return tbl

def value_stats(df: pd.DataFrame) -> pd.DataFrame: # Stats globales sur Value_num
v = df["Value_num"]
desc = v.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame(name="Value_num").reset_index().rename(columns={"index": "metric"})
return desc

def value_stats_by_element(df: pd.DataFrame) -> pd.DataFrame:
if "Element" not in df.columns:
return pd.DataFrame()
g = df.groupby("Element")["Value_num"].describe(percentiles=[0.05, 0.5, 0.95])
g.columns = [c if isinstance(c, str) else c[1] for c in g.columns]
return g.reset_index()

def save_table(df: pd.DataFrame, outdir: Path, name: str):
path = outdir / "tables" / f"{name}.csv"
df.to_csv(path, index=False, encoding="utf-8-sig")
print(f"[OK] Table: {path}")

def plot_year_hist(df: pd.DataFrame, outdir: Path):
if "Year" not in df.columns:
return
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["Year"].dropna().astype(int), bins=50, ax=ax, color="#2563eb")
ax.set_title("Répartition des enregistrements par année")
ax.set_xlabel("Année")
ax.set_ylabel("Nombre de lignes")
fig.tight_layout()
p = outdir / "figs" / "year_hist.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"[OK] Figure: {p}")

def plot*top_bar(df: pd.DataFrame, col: str, outdir: Path, top_n=20, metric="n_rows"):
tbl = counts_by(df, col)
if tbl.empty:
return
tbl = tbl.head(top_n)
fig, ax = plt.subplots(figsize=(10, max(4, 0.35 \* len(tbl))))
sns.barplot(data=tbl, y=col, x=metric, ax=ax, color="#059669")
ax.set_title(f"Top {top_n} {col} par nombre de lignes")
ax.set_xlabel("Nombre de lignes")
ax.set_ylabel(col)
fig.tight_layout()
p = outdir / "figs" / f"top*{col}\_bar.png"
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"[OK] Figure: {p}")

def plot_value_hist(df: pd.DataFrame, outdir: Path):
if "Value_num" not in df.columns:
return
v = df["Value_num"].dropna()
if v.empty:
return
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(v, bins=60, ax=axes[0], color="#7c3aed")
axes[0].set_title("Distribution de Value_num")
axes[0].set_xlabel("Value_num")

    # Log-scale (en filtrant les <= 0)
    v_pos = v[v > 0]
    if not v_pos.empty:
        sns.histplot(np.log10(v_pos), bins=60, ax=axes[1], color="#ef4444")
        axes[1].set_title("Distribution log10(Value_num) (Value_num > 0)")
        axes[1].set_xlabel("log10(Value_num)")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    p = outdir / "figs" / "value_histograms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[OK] Figure: {p}")

def plot_value_by_element_box(df: pd.DataFrame, outdir: Path, max_elements=20):
if "Element" not in df.columns or "Value_num" not in df.columns:
return # Prendre les éléments les plus fréquents
top_elems = df["Element"].value_counts().head(max_elements).index
sub = df[df["Element"].isin(top_elems)].copy()
if sub.empty:
return # Clip pour limiter l'effet des extrêmes
q01, q99 = sub["Value_num"].quantile([0.01, 0.99])
sub["Value_num_clip"] = sub["Value_num"].clip(lower=q01, upper=q99)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(top_elems))))
    sns.boxplot(data=sub, y="Element", x="Value_num_clip", ax=ax, color="#0ea5e9")
    ax.set_title("Value_num (clippé 1-99%) par Element (Top)")
    ax.set_xlabel("Value_num (clippé)")
    ax.set_ylabel("Element")
    fig.tight_layout()
    p = outdir / "figs" / "value_by_element_box.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[OK] Figure: {p}")

def write_markdown_report(outdir: Path, info: dict, miss_tbl: pd.DataFrame,
dup_info: dict, key_dup_tbl: pd.DataFrame,
year_tbl: pd.DataFrame,
area_counts: pd.DataFrame,
item_counts: pd.DataFrame,
elem_counts: pd.DataFrame,
flag_tbl: pd.DataFrame,
val_desc: pd.DataFrame,
val_by_elem: pd.DataFrame):
report = outdir / "report.md"
def top_preview(df, n=10):
if df is None or df.empty:
return "Aucune donnée."
return df.head(n).to_markdown(index=False)

    with open(report, "w", encoding="utf-8") as f:
        f.write(f"# Rapport EDA — FAOSTAT\n\n")
        f.write(f"**Lignes:** {info['n_rows']}  \n")
        f.write(f"**Colonnes:** {info['n_cols']}\n\n")

        f.write("## Colonnes et types\n\n")
        f.write(pd.DataFrame({"column": list(info["dtypes"].keys()), "dtype": list(info["dtypes"].values())}).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Valeurs manquantes (Top)\n\n")
        f.write(top_preview(miss_tbl))
        f.write("\n\n")

        f.write("## Doublons\n\n")
        f.write(f"- **Lignes dupliquées (exactes):** {dup_info['total_duplicate_rows']}\n\n")
        if dup_info.get("duplicate_key_rows") is not None:
            f.write(f"- **Lignes dupliquées sur la clé (Area Code, Item Code, Element Code, Year):** {dup_info['duplicate_key_rows']}\n\n")
            f.write("### Clés dupliquées (aperçu)\n\n")
            f.write(top_preview(key_dup_tbl))
            f.write("\n\n")

        f.write("## Couverture par année\n\n")
        f.write(top_preview(year_tbl, n=50))
        f.write("\n\n")

        f.write("## Top zones, items et éléments (par lignes)\n\n")
        f.write("### Zones (Area)\n\n")
        f.write(top_preview(area_counts))
        f.write("\n\n### Items\n\n")
        f.write(top_preview(item_counts))
        f.write("\n\n### Elements\n\n")
        f.write(top_preview(elem_counts))
        f.write("\n\n")

        f.write("## Flags\n\n")
        f.write(top_preview(flag_tbl))
        f.write("\n\n")

        f.write("## Statistiques sur Value_num\n\n")
        f.write(top_preview(val_desc, n=20))
        f.write("\n\n### Par Element\n\n")
        f.write(top_preview(val_by_elem, n=50))
        f.write("\n\n")

        f.write("## Fichiers générés\n\n")
        f.write("- Tables CSV dans `tables/`\n")
        f.write("- Figures PNG dans `figs/`\n")

    print(f"[OK] Rapport Markdown: {report}")

def main():
parser = argparse.ArgumentParser(description="EDA pour faostat_fusion_clean.csv")
parser.add_argument(
"--csv",
type=str,
default=r"C:\Users\moham\Music\faostat\faostat_fusion_clean.csv",
help="Chemin du CSV fusionné FAOSTAT"
)
parser.add_argument(
"--outdir",
type=str,
default=None,
help="Dossier de sortie (par défaut: à côté du CSV, dossier horodaté)"
)
args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    base_out = Path(args.outdir) if args.outdir else csv_path.parent
    outdir = make_outdir(base_out)

    df = read_data(csv_path)
    df = ensure_types(df)

    # Clé standard FAOSTAT pour contrôle doublons
    key_cols = ["Area Code", "Item Code", "Element Code", "Year"]
    if not all(k in df.columns for k in key_cols):
        print("[WARN] Les colonnes de clé standard ne sont pas toutes présentes, le contrôle de doublons par clé sera partiel.")

    # Synthèses
    info = overview(df)
    miss_tbl = missing_summary(df)
    dup_info, key_dup_tbl = duplicate_summary(df, key_cols=key_cols)

    year_tbl = year_coverage(df)
    area_counts = counts_by(df, "Area")
    item_counts = counts_by(df, "Item")
    elem_counts = counts_by(df, "Element")
    flag_tbl = flag_summary(df)
    val_desc = value_stats(df)
    val_by_elem = value_stats_by_element(df)

    # Sauvegardes tables
    save_table(miss_tbl, outdir, "missing_by_column")
    save_table(key_dup_tbl, outdir, "duplicate_keys")
    save_table(year_tbl, outdir, "year_coverage")
    save_table(area_counts, outdir, "counts_by_area")
    save_table(item_counts, outdir, "counts_by_item")
    save_table(elem_counts, outdir, "counts_by_element")
    save_table(flag_tbl, outdir, "flags_summary")
    save_table(val_desc, outdir, "value_num_descriptive")
    save_table(val_by_elem, outdir, "value_num_by_element")

    # Figures
    sns.set_theme(style="whitegrid")
    plot_year_hist(df, outdir)
    plot_top_bar(df, "Area", outdir)
    plot_top_bar(df, "Item", outdir)
    plot_top_bar(df, "Element", outdir)
    plot_value_hist(df, outdir)
    plot_value_by_element_box(df, outdir)

    # Rapport Markdown
    write_markdown_report(
        outdir, info, miss_tbl, dup_info, key_dup_tbl,
        year_tbl, area_counts, item_counts, elem_counts,
        flag_tbl, val_desc, val_by_elem
    )

    print(f"[DONE] Rapport EDA prêt dans: {outdir}")

if **name** == "**main**":
main()

sion vraie
import pandas as pd
import numpy as np
from pathlib import Path

# 📁 Chemins vers tes fichiers

PATHS = {
"ndvi": r"C:\Users\moham\Music\faostat\NDVI_NDMI_Merged_Mali_2021_2024.csv",
"chirps_smap": r"C:\Users\moham\Music\faostat\CHIRPS MLI.csv",
"worldclim_monthly": r"C:\Users\moham\Music\faostat\WorldClim MLI.csv",
"worldclim_bio": r"C:\Users\moham\Music\faostat\WorldClim BIO MLI.csv",
"smap_soil": r"C:\Users\moham\Music\faostat\SMAP_SoilMoisture_All.csv",
"gedi": r"C:\Users\moham\Music\faostat\Gedi MLI.csv",
"wapor": r"C:\Users\moham\Music\faostat\WAPOR_MLI.csv"
}

# 📆 Saison par défaut (à adapter selon culture)

SEASON_MONTHS = [6, 7, 8, 9, 10] # Exemple à adapter

# 📌 Fonction générique pour générer un identifiant admin

def make_adm_id(df):
cols = df.columns

    if "NDVI_ADM1_NAME" in cols and "ADM2_NAME" in cols:
        df["adm_id"] = df["NDVI_ADM1_NAME"].str.strip() + "_" + df["ADM2_NAME"].str.strip()
    elif "ADM1_NAME" in cols and "ADM2_NAME" in cols:
        df["adm_id"] = df["ADM1_NAME"].str.strip() + "_" + df["ADM2_NAME"].str.strip()
    elif "ADM1_NAME" in cols and "ADM0_NAME" in cols:
        df["adm_id"] = df["ADM1_NAME"].str.strip()
    elif "ADM0_NAME" in cols:
        df["adm_id"] = df["ADM0_NAME"].str.strip()
    else:
        raise ValueError(f"Impossible de créer adm_id, colonnes trouvées: {cols}")

    return df

# 📊 Agrégation NDVI/NDMI

def process_ndvi(path):
df = pd.read_csv(path)
df = make_adm_id(df)

    # Filtrer par mois de saison
    if "month" in df.columns:
        df = df[df["month"].isin(SEASON_MONTHS)]

    agg_dict = {}
    for col in ["NDVI_mean", "NDVI_max", "NDVI_min", "NDVI_stdDev",
                "NDMI_mean", "NDMI_max", "NDMI_min", "NDMI_stdDev"]:
        if col in df.columns:
            agg_dict[col] = "mean" if "mean" in col or "stdDev" in col else "max"

    grouped = df.groupby(["adm_id", "year"]).agg(agg_dict).reset_index()
    return grouped

# 🌧️ Agrégation CHIRPS + SMAP

def process_chirps_smap(path):
df = pd.read_csv(path)
df = make_adm_id(df)

    # Harmoniser les colonnes
    if "EXP1_YEAR" in df.columns:
        df.rename(columns={"EXP1_YEAR": "year"}, inplace=True)

    if "SMAP_SoilMoisture" in df.columns:
        df.rename(columns={"SMAP_SoilMoisture": "soil_moisture"}, inplace=True)
    elif "mean" in df.columns:  # cas du fichier SMAP seul
        df.rename(columns={"mean": "soil_moisture"}, inplace=True)

    agg_dict = {}
    if "CHIRPS_Daily" in df.columns:
        agg_dict["CHIRPS_Daily"] = "mean"
    if "CHIRPS_Pentad" in df.columns:
        agg_dict["CHIRPS_Pentad"] = "mean"
    if "soil_moisture" in df.columns:
        agg_dict["soil_moisture"] = "mean"

    grouped = df.groupby(["adm_id", "year"]).agg(agg_dict).reset_index()
    return grouped

# 🌡️ Agrégation WorldClim mensuel

def process_worldclim_monthly(path):
df = pd.read_csv(path)
df = make_adm_id(df)

    if "EXP1_YEAR" in df.columns:
        df.rename(columns={"EXP1_YEAR": "year"}, inplace=True)

    agg_dict = {}
    for col in ["prec", "tavg", "tmax", "tmin"]:
        if col in df.columns:
            agg_dict[col] = "mean"

    grouped = df.groupby(["adm_id", "year"]).agg(agg_dict).reset_index()
    return grouped

# 🧬 WorldClim BIO (statique)

def process_worldclim_bio(path):
df = pd.read_csv(path)
df = make_adm_id(df)

    bio_cols = [col for col in df.columns if col.startswith("bio")]
    df = df[["adm_id"] + bio_cols].drop_duplicates(subset=["adm_id"])
    return df

# 🌱 SMAP Soil Moisture (statique seul)

def process_smap_soil(path):
df = pd.read_csv(path)
df = make_adm_id(df)

    if "mean" in df.columns:
        df = df[["adm_id", "mean"]].rename(columns={"mean": "soil_moisture_mean"})
    elif "SMAP_SoilMoisture" in df.columns:
        df = df[["adm_id", "SMAP_SoilMoisture"]].rename(columns={"SMAP_SoilMoisture": "soil_moisture_mean"})

    df = df.drop_duplicates(subset=["adm_id"])
    return df

# 🌳 GEDI + Mangrove (statique)

def process_gedi(path):
df = pd.read_csv(path)
df = make_adm_id(df)

    keep_cols = [c for c in ["GEDI_CanopyHeight", "Mangrove2000", "TidalWetlands2019"] if c in df.columns]
    df = df[["adm_id"] + keep_cols].drop_duplicates(subset=["adm_id"])
    return df

# 💧 WAPOR (statique, indicateurs hydriques et productivité)

def process*wapor(path):
df = pd.read_csv(path) # Création de l'identifiant administratif
df["adm_id"] = df["ADM1_NAME"].str.strip() + "*" + df["ADM0_NAME"].str.strip()

    # On garde uniquement les colonnes utiles
    wapor_cols = [
        "AETI_2", "AETI_3", "DRET_2", "DRET_3",
        "Evap_3a", "Interception_2", "Interception_3a",
        "Evap_3b", "RET_2", "NPP_2", "RET_3",
        "Transp_2", "Transp_3"
    ]

    # Nettoyage et suppression des doublons
    df = df[["adm_id"] + wapor_cols].drop_duplicates(subset=["adm_id"])

    return df

# 🔗 Fusion finale

def fuse_all():
ndvi = process_ndvi(PATHS["ndvi"])
chirps = process_chirps_smap(PATHS["chirps_smap"])
clim = process_worldclim_monthly(PATHS["worldclim_monthly"])
bio = process_worldclim_bio(PATHS["worldclim_bio"])
soil = process_smap_soil(PATHS["smap_soil"])
gedi = process_gedi(PATHS["gedi"])

    # Fusion spatio-temporelle
    df = ndvi.merge(chirps, on=["adm_id", "year"], how="outer")
    df = df.merge(clim, on=["adm_id", "year"], how="outer")

    # Fusion statique
    df = df.merge(bio, on="adm_id", how="left")
    df = df.merge(soil, on="adm_id", how="left")
    df = df.merge(gedi, on="adm_id", how="left")

    # Sauvegarde
    out_path = Path(PATHS["ndvi"]).parent / "agro_dataset_mali.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Dataset fusionné sauvegardé : {out_path}")
    print(f"📊 Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")

fuse_all()

VériFication
import pandas as pd

# --- Charger le dataset satellite ---

file_path = r"C:\Users\moham\Music\faostat\agro_dataset_mali.csv"
df = pd.read_csv(file_path)

print("📥 Dimensions :", df.shape)
print("\n🔎 Aperçu des colonnes :")
print(df.columns.tolist())

# --- Vérification de base ---

print("\n🔁 Doublons :", df.duplicated().sum())
print("❌ Valeurs manquantes par colonne :\n", df.isna().sum())
print("📅 Années uniques :", df['year'].unique() if 'year' in df.columns else "⚠️ Pas de colonne 'year'")
print("🌍 Pays uniques :", df['country'].unique() if 'country' in df.columns else "⚠️ Pas de colonne 'country'")

# --- Nettoyage ---

# Supprimer doublons

df = df.drop_duplicates()

# Supprimer années aberrantes si 'year' existe

if 'year' in df.columns:
df = df[(df['year'] >= 1980) & (df['year'] <= 2100)]

# Vérifier colonnes numériques

num_cols = df.select_dtypes(include=['float64','int64']).columns
print("\n📊 Colonnes numériques :", num_cols.tolist())

# Sauvegarder dataset nettoyé

output_file = r"C:\Users\moham\Music\faostat\agro_dataset_mali_clean.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n✅ Dataset nettoyé sauvegardé : {output_file}")
print("📊 Dimensions finales :", df.shape)

Fusion FOASTAT ET NDVI ET AUTRES
import pandas as pd

# 📥 Chargement des datasets

agro_path = r"C:\Users\moham\Music\faostat\agro_dataset_mali_clean.csv"
faostat_path = r"C:\Users\moham\Music\faostat\faostat_fusion_clean.csv"

agro_df = pd.read_csv(agro_path)
fao_df = pd.read_csv(faostat_path)

# 🏷 Normalisation des colonnes FAOSTAT

fao_df = fao_df.rename(columns={"Area": "country", "Year": "year", "Value": "value"})

# ⚠ Filtrer uniquement Mali et années présentes dans agro_df

years_in_agro = agro_df["year"].unique()
fao_df = fao_df[(fao_df["country"] == "Mali") & (fao_df["year"].isin(years_in_agro))]

# 🔄 Créer colonne combinée Element + Item pour pivot

fao*df["element_item"] = fao_df["Element"] + "*" + fao_df["Item"]

# 🔀 Pivot pour transformer lignes en colonnes

fao_pivot = fao_df.pivot_table(
index=["country", "year"],
columns="element_item",
values="value",
aggfunc="first"
).reset_index()

# 🧩 Fusion avec le dataset agro

merged_df = pd.merge(
agro_df,
fao_pivot,
how="left",
left_on=["year"],
right_on=["year"]
)

# 💾 Sauvegarde

output_path = r"C:\Users\moham\Music\faostat\agro_dataset_mali_faostat.csv"
merged_df.to_csv(output_path, index=False)
print(f"✅ Fusion réussie, fichier sauvegardé : {output_path}")
print(f"📊 Dimensions finales : {merged_df.shape}")

Fusionnnn
import pandas as pd

# ⚙️ Paths

agro_path = r"C:\Users\moham\Music\faostat\agro_dataset_mali_clean.csv"
faostat_path = r"C:\Users\moham\Music\faostat\faostat_fusion_clean.csv"
output_path = r"C:\Users\moham\Music\faostat\agro_dataset_mali_faostat.csv"

print("📥 Chargement des fichiers...")
agro_df = pd.read_csv(agro_path)
fao_df = pd.read_csv(faostat_path, low_memory=False) # évite DtypeWarning

print(f"Agro-environnemental : {agro_df.shape}")
print(f"FAOSTAT : {fao_df.shape}")

# 🧹 Nettoyage dataset agro

agro_df = agro_df.dropna(subset=['year'])
agro_df['year'] = agro_df['year'].astype(int)
print(f"✅ Dataset agro nettoyé : {agro_df.shape}")

# 🧹 Pivot FAOSTAT : Item x Element

fao_pivot = fao_df.pivot_table(
index=['Area','Year'],
columns=['Item','Element'],
values='Value',
aggfunc='sum'
).reset_index()

# 🧹 Aplatir les colonnes multi-index

fao*pivot.columns = [
'*'.join(map(str, col)).strip() if isinstance(col, tuple) else col
for col in fao_pivot.columns.values
]

# Renommer les colonnes pour le merge

if 'Area*' in fao_pivot.columns: fao_pivot = fao_pivot.rename(columns={'Area*':'Area'})
if 'Year*' in fao_pivot.columns: fao_pivot = fao_pivot.rename(columns={'Year*':'Year'})

print(f"✅ FAOSTAT pivoté : {fao_pivot.shape}")

# 🔗 Fusion des datasets

final_df = agro_df.merge(
fao_pivot,
left_on=['adm_id','year'],
right_on=['Area','Year'],
how='left'
)

# Supprimer les colonnes redondantes après merge

final_df = final_df.drop(columns=['Area','Year'], errors='ignore')

# 💾 Sauvegarde

final_df.to_csv(output_path, index=False)
print(f"✅ Fusion réussie, fichier sauvegardé : {output_path}")
print(f"📊 Dimensions finales : {final_df.shape}")

# Pour VériFier si il n'ya pas de données manquyante

import os
import pandas as pd

# 📁 Dossier contenant les fichiers CSV

folder_path = r"C:\Users\moham\Music\Moh"
missing_threshold = 0.3 # Seuil de 30% pour alerter sur les colonnes très incomplètes

def analyze_missing_data(file_path):
print(f"\n📄 Analyse du fichier : {os.path.basename(file_path)}")
try:
df = pd.read_csv(file_path)
except Exception as e:
print(f"❌ Erreur de lecture : {e}")
return

    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    missing_ratio = total_missing / total_cells

    print(f"🔢 Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"❌ Valeurs manquantes totales : {total_missing} ({missing_ratio:.2%})")

    if missing_ratio > missing_threshold:
        print("⚠️ Beaucoup de données manquantes dans ce fichier")
    else:
        print("✅ Données globalement complètes")

    print("\n📊 Détail par colonne :")
    missing_per_column = df.isnull().mean().sort_values(ascending=False)
    for col, ratio in missing_per_column.items():
        status = "⚠️" if ratio > missing_threshold else "✅"
        print(f"  {status} {col}: {ratio:.2%} manquantes")

# 🔍 Analyse de tous les fichiers CSV dans le dossier

for filename in os.listdir(folder_path):
if filename.lower().endswith(".csv"):
file_path = os.path.join(folder_path, filename)
analyze_missing_data(file_path)

Contenu du dossier : C:\Users\moham\Music\Voice assistance

Dossier : .git
Dossier : .local
Fichier : .replit
Fichier : advanced_ai.py
Fichier : advanced_document_processor.py
Fichier : app.py
Fichier : auth_utils.py
Fichier : document_processor.py
Fichier : external_integrations.py
Fichier : language_detector.py
Fichier : main.py
Fichier : models.py
Fichier : pyproject.toml
Fichier : replit.md
Dossier : static
Dossier : templates
Fichier : uv.lock
Fichier : vector_store.py
Fichier : voice_assistant.py
Dossier : **pycache**
