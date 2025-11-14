import streamlit as st
import folium
from streamlit_folium import st_folium

def show_location_picker(lat=19.66, lon=4.3):
    m = folium.Map(location=[lat, lon], zoom_start=5)
    folium.Marker([lat, lon], popup="Zone cible").add_to(m)
    st_folium(m, width=700, height=400)
