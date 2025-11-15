import psycopg2 # type: ignore
import psycopg2.extras # type: ignore

def store_ndvi_profile(conn, lat, lon, profile, mission, year, stats):
    """
    Stocke un profil NDVI et ses statistiques dans la base de données PostgreSQL.

    Args:
        conn: Connexion psycopg2.
        lat (float): Latitude.
        lon (float): Longitude.
        profile (list[float]): Liste des valeurs NDVI.
        mission (str): Source (Sentinel-2, Landsat...).
        year (int): Année.
        stats (dict): Dictionnaire {mean, max, min, std, peak_index}.
    """
    query = """
        INSERT INTO ndvi_profiles
        (latitude, longitude, ndvi_profile, mission, year, mean, max, min, std, peak_index)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                query,
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
                ),
            )
        conn.commit()
        print(f"✅ Profil NDVI inséré pour ({lat}, {lon}) | {mission} - {year}")
    except Exception as e:
        print(f"⚠️ Erreur insertion NDVI: {e}")
        conn.rollback()
