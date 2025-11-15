import os
import logging
import rasterio  # type: ignore
import numpy as np # type: ignore
import xarray as xr  # type: ignore
from smart_agro_tools.ndvi_engine.config import MISSION_PRIORITY
from smart_agro_tools.ndvi_engine.dataset_loader import load_agricultural_data  # fonction existante
from smart_agro_tools.ndvi_engine.ndvi_core import calculate_indices  # type: ignore
from smart_agro_tools.ndvi_engine.masking_utils import masking  # type: ignore


def extract_ndvi_from_dataset(dataset: xr.Dataset) -> dict:
    """
    Extrait le profil NDVI masqué à partir d'un Dataset satellite.

    Args:
        dataset (xr.Dataset): Données satellites avec les bandes nécessaires.

    Returns:
        dict: Dictionnaire contenant le NDVI masqué et les timestamps.
    """
    if dataset is None or 'time' not in dataset.dims or dataset.time.size == 0:
        raise ValueError("Le dataset est vide ou ne contient pas de dimension temporelle.")

    try:
        ndvi = calculate_indices(dataset, index='NDVI')
        ndvi_masked = masking(ndvi)
        valid_ndvi = ndvi_masked.where(~ndvi_masked.isnull(), drop=True)

        if valid_ndvi.time.size == 0:
            raise ValueError("Aucune donnée NDVI valide après masquage.")

        return {
            "ndvi": valid_ndvi,
            "dates": valid_ndvi.time.values
        }

    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'extraction du profil NDVI : {e}")


def extract_valid_ndvi(lat, lon, year, mission_priority=MISSION_PRIORITY):
    """
    Extraction NDVI avec fallback sur différentes missions satellites.

    Args:
        lat (float): latitude du point.
        lon (float): longitude du point.
        year (int): année cible.

    Returns:
        dict: { "ndvi": DataArray NDVI nettoyée, "source": mission utilisée }
    """
    for mission in mission_priority:
        try:
            logging.info(f"🔍 Tentative avec mission : {mission}")
            ds = load_agricultural_data(lat=lat, lon=lon, year=year, mission=mission)

            if ds is None or not hasattr(ds, "time") or ds.time.size == 0:
                logging.warning(f"⚠️ Aucune donnée pour {mission} à ({lat}, {lon}) en {year}")
                continue

            ndvi = calculate_indices(ds, index='NDVI')
            ndvi_masked = masking(ndvi)

            if ndvi_masked.isnull().all():
                logging.warning(f"❌ NDVI entièrement nul après masquage pour {mission}")
                continue

            logging.info(f"✅ NDVI extrait avec succès depuis {mission}")
            return {"ndvi": ndvi_masked, "source": mission}

        except Exception as e:
            logging.error(f"❗Erreur lors de l'extraction avec {mission} : {e}")
            continue

    raise ValueError("🚫 Aucune donnée NDVI exploitable pour cette zone/année.")


def extract_ndvi_profile(lat: float, lon: float, ndvi_folder: str) -> list[float]:
    """
    Extrait un profil NDVI depuis un dossier de rasters GeoTIFF pour une position donnée.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        ndvi_folder (str): Chemin du dossier contenant des fichiers NDVI .tif.

    Returns:
        list[float]: Liste de valeurs NDVI normalisées (0-1).
    """
    ndvi_values = []

    try:
        tif_files = sorted([f for f in os.listdir(ndvi_folder) if f.endswith(".tif")])
        if not tif_files:
            logging.warning(f"⚠️ Aucun fichier .tif trouvé dans {ndvi_folder}")
            return []

        for file in tif_files:
            tif_path = os.path.join(ndvi_folder, file)
            with rasterio.open(tif_path) as src:
                row, col = src.index(lon, lat)
                value = src.read(1)[row, col]

                # Normalisation NDVI
                if value > 1:
                    value = value / 10000.0

                ndvi_values.append(round(float(value), 3))

    except Exception as e:
        logging.error(f"⚠️ Erreur extraction NDVI pour ({lat}, {lon}) : {e}")

    return ndvi_values
