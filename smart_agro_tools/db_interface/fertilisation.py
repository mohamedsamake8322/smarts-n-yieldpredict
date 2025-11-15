import psycopg2 # type: ignore

def insert_fertilisation(conn, lat, lon, npk, crop, year, source="auto"):
    """
    Insère une recommandation NPK dans la table fertilisation_recommandee.

    Args:
        conn: Connexion psycopg2.
        lat (float): Latitude du champ.
        lon (float): Longitude du champ.
        npk (dict): Dictionnaire contenant les clés 'N', 'P', 'K'.
        crop (str): Nom de la culture.
        year (int): Année de la recommandation.
        source (str): Source de la recommandation (par défaut "auto").
    """
    query = """
        INSERT INTO fertilisation_recommandee
        (geom, crop, year, n, p, k, source)
        VALUES (ST_SetSRID(ST_Point(%s, %s), 4326), %s, %s, %s, %s, %s, %s)
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (
                lon, lat,
                crop,
                year,
                npk.get("N", 0),
                npk.get("P", 0),
                npk.get("K", 0),
                source
            ))
        conn.commit()
        print(f"✅ Recommandation NPK insérée pour {crop} ({lat}, {lon})")
    except Exception as e:
        print(f"⚠️ Erreur insertion fertilisation: {e}")
        conn.rollback()
