import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import psycopg2  # type: ignore
import psycopg2.extras # type: ignore
import os

# Fonction pour stocker le profil NDVI et stats dans la base
def store_ndvi_profile(conn, lat, lon, profile, mission, year, stats):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ndvi_profiles
            (latitude, longitude, ndvi_profile, mission, year, mean, max, min, std, peak_index)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                lat,
                lon,
                profile,
                mission,
                year,
                stats["mean"],
                stats["max"],
                stats["min"],
                stats["std"],
                stats["peak_index"],
            )
        )
    conn.commit()

# Simule un profil NDVI avec 6 valeurs aléatoires entre 0.1 et 0.6
def simulate_ndvi_profile():
    return [round(np.random.uniform(0.1, 0.6), 2) for _ in range(6)]

# Calcule quelques stats sur le profil NDVI (facultatif)
def compute_ndvi_stats(profile):
    profile = np.array(profile)
    return {
        "mean": float(np.mean(profile)),
        "max": float(np.max(profile)),
        "min": float(np.min(profile)),
        "std": float(np.std(profile)),
        "peak_index": int(np.argmax(profile)),
    }

# Traite tout le CSV, insert ligne par ligne
def process_all_ndvi(
    conn,
    csv_path=r"C:\smarts-n-yieldpredict.git\data\dataset_agricole_prepared.csv",
):
    df_agri = pd.read_csv(csv_path)

    for idx, row in df_agri.iterrows():
        lat, lon, year = row["latitude"], row["longitude"], row["year"]
        culture = row.get("culture", "unknown")

        profile = simulate_ndvi_profile()
        stats = compute_ndvi_stats(profile)

        print(f"[{idx}] Insertion 📍({lat}, {lon}) | {culture}, {year} | NDVI stats: {stats}")

        # Ici on passe la liste Python directement ; psycopg2 gère la conversion en array PostgreSQL
        store_ndvi_profile(
            conn=conn,
            lat=lat,
            lon=lon,
            profile=profile,
            mission="Sentinel-2",
            year=int(year),
            stats=stats,
        )


if __name__ == "__main__":
    conn = psycopg2.connect(
        host="localhost",
        dbname="datacube",
        user="mohamedsamake2000",
        password="70179877Moh#",  # Remplace par ton mot de passe réel
        port=5432,
    )
    try:
        process_all_ndvi(conn)
    finally:
        conn.close()
