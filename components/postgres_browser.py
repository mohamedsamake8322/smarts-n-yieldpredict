import streamlit as st
import psycopg2
import pandas as pd

def show_db_browser():
    st.subheader("📦 Lecture PostgreSQL - fertilisation")
    try:
        conn = psycopg2.connect(
            host="localhost",
            dbname="datacube",
            user="mohamedsamake2000",
            password="70179877Moh#",
            port=5432
        )
        query = "SELECT crop, year, n, p, k FROM fertilisation_recommandee LIMIT 100;"
        df = pd.read_sql(query, conn)
        st.dataframe(df)
        conn.close()
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
