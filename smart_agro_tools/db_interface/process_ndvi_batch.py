import pandas as pd
import numpy as np
import geopandas as gpd
import pyproj
from shapely.geometry import Point
from db_interface.connector import connect_db
from db_interface.ndvi_storage import store_ndvi_profile
import os

# Forcer le chemin PROJ si nécessaire
proj_path = r"C:\Users\moham\anaconda3\envs\smartgeo310\Library\share\proj"
if os.path.exists(proj_path):
    pyproj.datadir.set_data_dir(proj_path)

print(f"📌 Chemin PROJ actif : {pyproj.datadir.get_data_dir()}")

# NDVI synthétique ou calculé — ici exemple fixe
def simulate_ndvi_profile():
    return [round(np.random.uniform(0.1, 0.6), 2) for _ in range(6)]

def compute_ndvi_stats(profile):
    profile = np.array(profile)
    return {
        "mean": float(np.mean(profile)),
        "max": float(np.max(profile)),
        "min": float(np.min(profile)),
        "std": float(np.std(profile)),
        "peak_index": int(np.argmax(profile)),
    }

# Traitement par lot
def process_all_ndvi(conn, csv_path="../data/dataset_agricole_prepared.csv"):
    df_agri = pd.read_csv(csv_path)

    for idx, row in df_agri.iterrows():
        lat, lon, year = row["latitude"], row["longitude"], row["year"]
        profile = simulate_ndvi_profile()
        stats = compute_ndvi_stats(profile)

        culture = row.get("culture", "unknown")
        yield_target = row.get("yield_target", None)

        print(f"[{idx}] Insertion 📍({lat}, {lon}) | {culture}, {year} | NDVI stats: {stats}")
        store_ndvi_profile(
            conn,
            lat,
            lon,
            profile,
            mission="Sentinel-2",
            year=int(year)
        )

if __name__ == "__main__":
    conn = connect_db(
        host="localhost",
        dbname="datacube",
        user="mohamedsamake2000",
        password="70179877Moh#",
        port=5432
    )
    process_all_ndvi(conn)
