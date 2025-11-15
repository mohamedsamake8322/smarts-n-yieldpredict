# smart_agro_tools/db_interface/__main__.py

from db_interface.connector import connect_db

if __name__ == '__main__':
    conn = connect_db(
    host="localhost",
    dbname="datacube",
    user="mohamedsamake2000",
    password="70179877Moh#",
    port=5432
)
