import sys
import os
import logging
import numpy as np  # type: ignore
import psycopg2  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, Any, List

# --- Imports locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from smart_agro_tools.ndvi_engine.extractor import extract_ndvi_profile
from smart_agro_tools.ndvi_engine.validator import check as ndvi_check
from smart_agro_tools.input_recommender.climate_filter import adjust_for_climate
from smart_agro_tools.input_recommender.stress_detector import detect_stress_from_ndvi
from smart_agro_tools.input_recommender.soil_matcher import adjust_for_soil
from smart_agro_tools.input_recommender.recommender import suggest_inputs
from smart_agro_tools.db_interface.ndvi_storage import store_ndvi_profile

# --- Configuration ---
CSV_PATH = r"C:\plateforme-agricole-complete-v2\data\dataset_agricole_prepared.csv"
NDVI_FOLDER = r"C:\plateforme-agricole-complete-v2\data\ndvi_rasters"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_ndvi_stats(profile: List[float]) -> Dict[str, Any]:
    """Compute basic statistics from NDVI profile."""
    arr = np.array(profile)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "std": float(np.std(arr)),
        "peak_index": int(np.argmax(arr)),
    }


def process_ndvi_data(conn: psycopg2.extensions.connection) -> None:
    """Process NDVI data and insert into PostgreSQL database."""
    logging.info(f"📥 Chargement du CSV : {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    for idx, row in df.iterrows():
        lat, lon, year = float(row["latitude"]), float(row["longitude"]), int(row["year"])
        crop = str(row.get("culture", "unknown"))
        soil_data = {
            "GWETPROF": row.get("GWETPROF"),
            "GWETROOT": row.get("GWETROOT"),
            "GWETTOP": row.get("GWETTOP"),
        }
        climate_data = {
            "WD10M": row.get("WD10M"),
            "WS10M_RANGE": row.get("WS10M_RANGE"),
        }

        try:
            # Step 1: NDVI Extraction
            profile = extract_ndvi_profile(lat, lon, NDVI_FOLDER)
            if not profile:
                logging.warning(f"[{idx}] ❌ NDVI vide pour {crop} ({lat}, {lon})")
                continue

            # Step 2: NDVI Validation
            if not ndvi_check(profile):
                logging.warning(f"[{idx}] ❌ NDVI invalide pour {crop} ({lat}, {lon})")
                continue

            # Step 3: Input Recommendation
            soil_factor = adjust_for_soil(soil_data)
            climate_factor = adjust_for_climate(climate_data)
            stress = detect_stress_from_ndvi(profile)
            recommendation = suggest_inputs(profile, soil_data, climate_data, crop, row.get("yield_target"))
            stats = compute_ndvi_stats(profile)

            # Step 4: Store NDVI Profile
            store_ndvi_profile(conn, lat, lon, profile, "Sentinel-2", year, stats)

            logging.info(
                f"[{idx}] ✅ {crop} ({lat}, {lon}) | NDVI stats: {stats} | NPK Recommendation: {recommendation}"
            )

        except Exception as e:
            logging.error(f"[{idx}] ❌ Erreur sur {crop} ({lat}, {lon}): {e}")


def get_database_connection() -> psycopg2.extensions.connection:
    """Establish a PostgreSQL connection."""
    return psycopg2.connect(
        host="localhost",
        dbname="datacube",
        user="mohamedsamake2000",
        password="70179877Moh#",  # 🔒 TIP: Use os.environ.get("DB_PASSWORD") for production
        port=5432
    )


if __name__ == "__main__":
    try:
        with get_database_connection() as conn:
            process_ndvi_data(conn)
    except Exception as e:
        logging.critical(f"❌ Failed to connect or process NDVI data: {e}")
