import pandas as pd # type: ignore
import os
import rasterio # type: ignore
REQUIRED_COLUMNS = [
    "country", "year", "latitude", "longitude",
    "GWETPROF", "GWETROOT", "GWETTOP", "WD10M", "WS10M_RANGE",
    "Export quantity", "Import quantity", "Production", "pesticides_use",
    "culture", "yield_target"
]
def load_ard(file_path):
    """
    Charge un jeu de données ARD (Analytical Ready Data) depuis un fichier raster.

    Args:
        file_path (str): Chemin vers le fichier raster (ex: GeoTIFF).

    Returns:
        tuple: (data, profile, transform, crs)
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # lit la première bande
            profile = src.profile
            transform = src.transform
            crs = src.crs
        return data, profile, transform, crs
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du fichier ARD : {e}")

def load_agricultural_data(path="data/dataset_agricole_prepared.csv"):
    """
    Charge et nettoie le dataset agricole.

    Args:
        path (str): chemin vers le fichier CSV.

    Returns:
        pd.DataFrame: table prête pour analyse agro-intelligente.

    Raises:
        FileNotFoundError: si le fichier est introuvable.
        ValueError: si des colonnes essentielles sont manquantes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier '{path}' est introuvable.")

    df = pd.read_csv(path)

    # Vérification des colonnes attendues
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing_cols}")

    # Nettoyage des cultures
    df['culture'] = df['culture'].astype(str).str.strip()

    # Détection et suppression des doublons
    df = df.drop_duplicates(subset=["country", "year", "latitude", "longitude", "culture"])

    # Conversion des coordonnées
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Indexation temporelle potentielle
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Nettoyage valeurs manquantes critiques
    df = df.dropna(subset=["latitude", "longitude", "culture", "yield_target"])

    return df
