import streamlit as st # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import psycopg2 # type: ignore
import json
import plotly.express as px # type: ignore
import datetime
import ee # type: ignore
import geemap.foliumap as geemap # type: ignore

# ====================================
#        CONFIGURATION STREAMLIT
# ====================================
st.set_page_config(page_title="🛰️ Satellite NDVI & Soil Viewer", layout="wide")
st.title("🛰️ NDVI & Soil & Satellite Viewer - Sama AgroLink Africa")
st.markdown("### Analyse combinée **NDVI (Base + Satellite)** et **Sol**")

# ====================================
#       INITIALISATION GEE
# ====================================
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# ====================================
#       CONNEXION POSTGRESQL
# ====================================
def get_ndvi_data(lat=None, lon=None, limit=10):
    """Récupère les profils NDVI stockés dans PostgreSQL."""
    conn = psycopg2.connect(
        host="localhost",
        dbname="datacube",
        user="mohamedsamake2000",
        password="Motdepasse",  # Modifier par ton mot de passe réel
        port=5432
    )
    cur = conn.cursor()

    if lat and lon:
        query = """
            SELECT id, latitude, longitude, year, ndvi_profile, mean, max, min, std
            FROM ndvi_profiles
            WHERE ROUND(latitude::numeric, 2) = ROUND(%s::numeric, 2)
              AND ROUND(longitude::numeric, 2) = ROUND(%s::numeric, 2)
            ORDER BY year DESC
            LIMIT %s;
        """
        cur.execute(query, (lat, lon, limit))
    else:
        query = """
            SELECT id, latitude, longitude, year, ndvi_profile, mean, max, min, std
            FROM ndvi_profiles
            ORDER BY inserted_at DESC
            LIMIT %s;
        """
        cur.execute(query, (limit,))

    rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=["id", "latitude", "longitude", "year", "ndvi_profile", "mean", "max", "min", "std"])
    return df

# ====================================
#       CHARGEMENT DES DONNÉES
# ====================================
@st.cache_data
def load_regions():
    return gpd.read_file("africa_admin_level2.geojson")

@st.cache_data
def load_soil_profile():
    df_soil = pd.read_csv("soil_profile_africa.csv")
    soil_gdf = gpd.GeoDataFrame(df_soil, geometry=gpd.points_from_xy(df_soil.x, df_soil.y), crs="EPSG:4326")
    return df_soil, soil_gdf, [col for col in df_soil.columns if "_" in col and col != "geometry"]

regions = load_regions()
df_soil, soil_gdf, soil_cols = load_soil_profile()

# ====================================
#       SÉLECTION DE LA ZONE
# ====================================
mode = st.radio("🎯 Mode d'analyse NDVI", ["GPS Coordinates", "Administrative Region"])

lat, lon = None, None
geometry = None
poly_geom = None

if mode == "GPS Coordinates":
    lat = st.number_input("Latitude", value=11.174, format="%.6f")
    lon = st.number_input("Longitude", value=-1.562, format="%.6f")
    buffer_m = st.slider("Buffer autour du champ (mètres)", 100, 2000, 1000)
    geometry = ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds()
    poly_geom = gpd.GeoSeries([gpd.points_from_xy([lon], [lat])[0].buffer(buffer_m/111000)], crs="EPSG:4326")

else:
    countries = sorted(regions["GID_0"].dropna().unique())
    selected_country = st.selectbox("🌍 Pays", countries)
    filtered = regions[regions["GID_0"] == selected_country]
    region_names = sorted(filtered["NAME_2"].dropna().unique())
    selected_region = st.selectbox("🏢 Région", region_names)
    selected_geom = filtered[filtered["NAME_2"] == selected_region].geometry.iloc[0]
    bounds = selected_geom.bounds
    minx, miny, maxx, maxy = bounds
    geometry = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    poly_geom = gpd.GeoSeries([selected_geom], crs="EPSG:4326")

# ====================================
#      PARAMÈTRES D'ANALYSE
# ====================================
start_date = st.date_input("📅 Date de début", value=datetime.date(2023, 6, 1))
end_date = st.date_input("📅 Date de fin", value=datetime.date(2023, 7, 1))

crop = st.selectbox("🌾 Type de culture", [
    "Maize", "Rice", "Wheat", "Sorghum", "Tomato", "Potato", "Soybean",
    "Sunflower", "Banana", "Mango", "Orange", "Coffee", "Tea", "Cocoa"
])
agro_zone = st.text_input("Zone agroécologique (ex: Sudan West)", "Sudan West")

selected_soil_col = st.selectbox("🧪 Propriété du sol", soil_cols)

# ====================================
#       FONCTION MASQUAGE NUAGES
# ====================================
def mask_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
    return image.updateMask(cloud_mask)

# ====================================
#       BOUTONS D'ACTION
# ====================================

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Afficher NDVI stockés"):
        df_ndvi = get_ndvi_data(lat, lon, limit=5)
        if df_ndvi.empty:
            st.warning("⚠️ Aucun profil NDVI trouvé pour cette localisation.")
        else:
            st.subheader("📊 Profils NDVI (Base de données)")
            st.dataframe(df_ndvi)

            # Graphique NDVI (1er profil)
            ndvi_values = df_ndvi.iloc[0]["ndvi_profile"]
            if isinstance(ndvi_values, list):
                fig = px.line(
                    y=ndvi_values,
                    markers=True,
                    title=f"Profil NDVI - Année {df_ndvi.iloc[0]['year']}",
                    labels={"y": "NDVI", "x": "Indice (Mois)"}
                )
                st.plotly_chart(fig)

            # Option de téléchargement CSV
            csv = df_ndvi.to_csv(index=False)
            st.download_button(
                "📥 Télécharger NDVI (CSV)",
                data=csv,
                file_name="ndvi_profiles.csv",
                mime="text/csv"
            )

with col2:
    if st.button("🔍 Générer NDVI via Google Earth Engine"):
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(geometry) \
            .filterDate(str(start_date), str(end_date)) \
            .map(mask_clouds) \
            .sort("CLOUDY_PIXEL_PERCENTAGE")

        image = collection.first()
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndvi_params = {'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}

        # Carte interactive
        if mode == "GPS Coordinates":
            Map = geemap.Map(center=[lat, lon], zoom=12)
            Map.addLayer(ee.Geometry.Point([lon, lat]), {}, 'Field')
        else:
            center_lat = (miny + maxy) / 2
            center_lon = (minx + maxx) / 2
            Map = geemap.Map(center=[center_lat, center_lon], zoom=6)
            Map.addLayer(selected_geom.__geo_interface__, {}, 'Selected Region')

        Map.addLayer(ndvi, ndvi_params, 'NDVI')

        # Données sol
        soil_within = soil_gdf[soil_gdf.within(poly_geom.iloc[0])]
        Map.add_points_from_xy(
            soil_within,
            column=selected_soil_col,
            color_column=selected_soil_col,
            color_palette="viridis",
            layer_name=f"Soil: {selected_soil_col}",
            radius=5,
            info_mode="on_hover"
        )

        Map.to_streamlit(height=600)

        url = ndvi.getThumbURL({
            'min': 0,
            'max': 1,
            'region': geometry,
            'dimensions': 512,
            'format': 'png'
        })
        st.markdown(f"📥 [Télécharger l'image NDVI]({url})")

        # Résumé JSON
        soil_stats = soil_within.mean(numeric_only=True).to_dict() if not soil_within.empty else {}

        result_json = {
            "mode": mode,
            "latitude": lat if mode == "GPS Coordinates" else None,
            "longitude": lon if mode == "GPS Coordinates" else None,
            "country": selected_country if mode == "Administrative Region" else None,
            "region": selected_region if mode == "Administrative Region" else None,
            "crop": crop,
            "agroecological_zone": agro_zone,
            "ndvi_url": url,
            "period": {"start": str(start_date), "end": str(end_date)},
            "soil_property": selected_soil_col,
            "soil_profile": soil_stats,
        }

    st.subheader("🧪 NDVI + Soil Data Output (for SènèYield API)")
    st.code(json.dumps(result_json, indent=2), language='json')
