import pandas as pd
import numpy as np
import geopandas as gpd
import pyproj
from shapely.geometry import Point
from db_interface.connector import connect_db
from db_interface.ndvi_storage import store_ndvi_profile
import os
import pyproj

# Forcer le chemin PROJ correct
proj_path = r"C:\Users\moham\anaconda3\envs\smartgeo310\Library\share\proj"
if os.path.exists(proj_path):
    pyproj.datadir.set_data_dir(proj_path)
else:
    print(f"⚠️ Le chemin PROJ n'existe pas : {proj_path}")

print(f"Chemin PROJ actif : {pyproj.datadir.get_data_dir()}")

# Vérification et configuration du chemin PROJ si nécessaire
# Dans pyproj >= 3.4, get_default_data_dir() n'existe plus.
# En général, pyproj trouve les données PROJ automatiquement.
# Si besoin, on peut forcer un chemin.
if not pyproj.datadir.get_data_dir():
    try:
        pyproj.datadir.set_data_dir("C:/Users/moham/anaconda3/envs/smartgeo310/Library/share/proj")
    except Exception as e:
        print(f"⚠️ Impossible de définir le chemin PROJ : {e}")

# Chargement du fichier CSV
df_agri = pd.read_csv("../data/dataset_agricole_prepared.csv")

# Création d'une GeoDataFrame
gdf_agri = gpd.GeoDataFrame(
    df_agri,
    geometry=gpd.points_from_xy(df_agri.longitude, df_agri.latitude),
    crs="EPSG:4326"
)

# Fonction d'enrichissement NDVI
def compute_ndvi_stats(profile):
    profile = np.array(profile)
    return {
        "mean": float(np.mean(profile)),
        "max": float(np.max(profile)),
        "min": float(np.min(profile)),
        "std": float(np.std(profile)),
        "peak_index": int(np.argmax(profile)),
    }

# Exemple d’insertion enrichie
def process_single_ndvi(conn):
    profile = [0.12, 0.15, 0.18, 0.20, 0.17, 0.13, 0.10]
    lat, lon, year = 19.66, 4.3, 2021

    stats = compute_ndvi_stats(profile)
    match = gdf_agri[
        (gdf_agri.year == year) &
        (gdf_agri.latitude == lat) &
        (gdf_agri.longitude == lon)
    ]

    if not match.empty:
        culture = match.iloc[0]["culture"]
        yield_target = match.iloc[0]["yield_target"]
        print(f"Enrichi avec culture = {culture}, yield_target = {yield_target}, stats = {stats}")
        store_ndvi_profile(conn, lat, lon, profile, mission="Sentinel-2", year=year)

if __name__ == "__main__":
    conn = connect_db(
        host="localhost",
        dbname="datacube",
        user="mohamedsamake2000",
        password="70179877Moh#",
        port=5432
    )
    process_single_ndvi(conn)
