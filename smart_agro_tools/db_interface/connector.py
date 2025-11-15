import psycopg2 # type: ignore
def connect_db(host, dbname, user, password, port):
    try:
        conn = psycopg2.connect(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=port
        )
        print("Connexion réussie à la base PostgreSQL ✅")
        return conn
    except Exception as e:
        print("Erreur de connexion ❌ :", e)
        return None

