import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.earth_engine_loader import get_agro_indicators
import ee
from utils import AdvancedAI

SERVICE_ACCOUNT = "323629817646-compute@developer.gserviceaccount.com"  # ton email de service account
KEY_FILE = "utils/keys/gee-key.json"  # chemin vers le JSON téléchargé

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials)

print("✅ Google Earth Engine initialisé avec Service Account")

# ---------------------------
# Page config FIRST
# ---------------------------
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# ---------------------------
# Imports from local utils with safe fallback
# ---------------------------
try:
    from utils.visualization import create_overview_charts, create_trend_analysis  # noqa: F401
except Exception:
    # Optional utils; continue without them
    create_overview_charts = None
    create_trend_analysis = None

try:
    from utils.data_processing import (
        load_and_merge_indicators,
        get_sample_agricultural_data,
    )
except ImportError:
    # Provide a robust fallback for sample data
    def get_sample_agricultural_data(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic agricultural dataset with common columns."""
        rng = np.random.default_rng(seed)
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
        ndvi = np.clip(0.5 + 0.1 * np.sin(np.linspace(0, 3*np.pi, n_days)) + rng.normal(0, 0.03, n_days), 0, 1)
        rainfall = np.clip(rng.gamma(shape=2.0, scale=3.0, size=n_days), 0, None)  # mm/day
        soil_moisture = np.clip(0.25 + 0.05 * np.cos(np.linspace(0, 2*np.pi, n_days)) + rng.normal(0, 0.01, n_days), 0, 1)
        temperature = 18 + 7*np.sin(np.linspace(0, 2*np.pi, n_days)) + rng.normal(0, 1.5, n_days)  # °C
        evapotranspiration = np.clip(2 + 0.5*np.sin(np.linspace(0, 2*np.pi, n_days)) + rng.normal(0, 0.2, n_days), 0, None)
        area = np.clip(rng.normal(4, 1.2, n_days), 0.5, None)  # ha
        yield_t_ha = np.clip(2.5 + 0.6*np.sin(np.linspace(0, 2*np.pi, n_days)) + 0.8*ndvi + rng.normal(0, 0.3, n_days), 0, None)
        profit = (yield_t_ha * area * 150) - (area * 40) + rng.normal(0, 50, n_days)
        df = pd.DataFrame({
            "date": dates,
            "ndvi": ndvi,
            "rainfall": rainfall,
            "soil_moisture": soil_moisture,
            "temperature": temperature,
            "evapotranspiration": evapotranspiration,
            "area": area,
            "yield": yield_t_ha,
            "profit": profit,
            "crop_type": rng.choice(["Maize", "Rice", "Sorghum", "Cotton"], size=n_days),
            "ADM2_NAME": rng.choice(["Bamako", "Kayes", "Koulikoro"], size=n_days),
            "smap_moisture": np.clip(0.2 + 0.1*np.sin(np.linspace(0, 1.5*np.pi, n_days)) + rng.normal(0, 0.02, n_days), 0, 1),
            "soil_sand": np.clip(40 + 10*rng.normal(0.0, 1.0, n_days), 10, 90),
            "soil_clay": np.clip(20 + 8*rng.normal(0.0, 1.0, n_days), 5, 70),
            "spei_03": np.clip(-1 + 2*np.sin(np.linspace(0, 2*np.pi, n_days)) + rng.normal(0, 0.3, n_days), -3, 3),
            "region": rng.choice(["Mali", "Senegal", "Côte d’Ivoire"], size=n_days),
        })
        return df
def load_and_merge_indicators(base_path=None, n_days: int = 30):
    """
    Charge les indicateurs agricoles depuis Google Earth Engine pour une zone prédéfinie
    et calcule des colonnes supplémentaires comme rendement et profit.

    Args:
        base_path: Chemin de base pour d'éventuelles données locales (non utilisé ici)
        n_days: Nombre de jours à considérer pour récupérer les indicateurs
    Returns:
        pd.DataFrame avec les indicateurs et colonnes calculées
    """
    # Exemple : rectangle autour de Manisa
    # Remarque : geometry est défini globalement dans get_agro_indicators
    df = get_agro_indicators(n_days=n_days)

    # Ajout de colonnes calculées
    df["crop_type"] = "Maize"       # Placeholder
    df["area"] = 4.0                 # hectares, placeholder
    df["yield"] = df["NDVI"] * 3.2   # Simple proxy pour rendement
    df["profit"] = df["yield"] * df["area"] * 150 - df["area"] * 40

    return df


# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def _get_sample_cached(n_days: int = 120):
    return get_sample_agricultural_data(n_days=n_days)


def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        return df.dropna(subset=[col])
    return df


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


# ---------------------------
# Header
# ---------------------------
st.title("📊 Agricultural Dashboard")
st.markdown("### Comprehensive overview of your agricultural operations")

# ---------------------------
# Data loader panel
# ---------------------------
if "agricultural_data" not in st.session_state:
    st.warning("No data available. Please upload data, use sample data, or load Mali indicators.")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Upload Data", use_container_width=True):
            # Requires Streamlit >= 1.31 experimental API; adjust if needed
            try:
                st.switch_page("pages/5_Data_Upload.py")
            except Exception:
                st.error("Couldn't switch page automatically. Open '5_Data_Upload' from the sidebar.")
    with col2:
        if st.button("Use Sample Data", use_container_width=True):
            st.session_state.agricultural_data = _get_sample_cached()
            st.rerun()
    with col3:
        if st.button("Load Mali Indicators", use_container_width=True):
            # Prefer relative project path over absolute Windows path for portability
            base_path = "data"  # adjust to your repo structure
            st.session_state.agricultural_data = load_and_merge_indicators(base_path=base_path)
            st.rerun()
    st.stop()

# ---------------------------
# Main dataset
# ---------------------------
data = st.session_state.agricultural_data
if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

# Cleanups
data = _ensure_datetime(data, "date")
if "ADM2_NAME" in data.columns:
    data["ADM2_NAME"] = data["ADM2_NAME"].astype(str)

st.caption(f"Records: **{len(data):,}** · Columns: **{len(data.columns)}**")

# ---------------------------
# KPI Section
# ---------------------------
st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if "yield" in data.columns and data["yield"].notna().any():
        avg_yield = data["yield"].mean()
        median_yield = data["yield"].median()
        st.metric("Average Yield (t/ha)", f"{avg_yield:.2f}", delta=f"{(avg_yield - median_yield):.2f}")
    else:
        st.metric("Average Yield", "--", help="Yield data not available")

with col2:
    if "area" in data.columns and data["area"].notna().any():
        total_area = data["area"].sum()
        st.metric("Total Area (ha)", f"{total_area:.1f}", help="Total cultivated area")
    else:
        st.metric("Total Area", "--", help="Area data not available")

with col3:
    if "crop_type" in data.columns:
        crop_diversity = int(data["crop_type"].nunique())
        st.metric("Crop Varieties", crop_diversity, help="Number of different crops grown")
    else:
        st.metric("Crop Varieties", "--", help="Crop data not available")

with col4:
    if "profit" in data.columns and data["profit"].notna().any():
        total_profit = data["profit"].sum()
        per_record = total_profit / max(len(data), 1)
        st.metric("Total Profit ($)", f"${total_profit:,.0f}", delta=f"${per_record:,.0f} per record")
    else:
        st.metric("Total Profit", "--", help="Profit data not available")

# ---------------------------
# Charts
# ---------------------------
st.markdown("---")
st.subheader("Data Visualization")

TABS = [
    "Yield Analysis",
    "Crop Distribution",
    "Seasonal Trends",
    "Performance Metrics",
    "NDVI/EVI Trends",
    "SPEI Drought Index",
    "Soil Properties",
]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(TABS)

with tab1:
    if _has_cols(data, ["yield", "crop_type"]):
        fig_yield = px.box(data, x="crop_type", y="yield", title="Yield Distribution by Crop Type")
        st.plotly_chart(fig_yield, use_container_width=True)
    else:
        st.info("'yield' and/or 'crop_type' columns are missing.")

with tab2:
    if "crop_type" in data.columns:
        crop_counts = data["crop_type"].astype(str).value_counts()
        if not crop_counts.empty:
            fig_pie = px.pie(values=crop_counts.values, names=crop_counts.index, title="Crop Distribution")
            fig_bar = px.bar(x=crop_counts.index, y=crop_counts.values, title="Crop Count by Type")
            st.plotly_chart(fig_pie, use_container_width=True)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No crop_type values to display.")
    else:
        st.info("'crop_type' column is missing.")

with tab3:
    if _has_cols(data, ["date", "yield"]):
        tmp = data[["date", "yield"]].dropna().copy()
        if not tmp.empty:
            monthly_yield = (
                tmp.assign(month=lambda d: d["date"].dt.to_period("M"))
                .groupby("month")["yield"].mean()
                .reset_index()
            )
            monthly_yield["date"] = monthly_yield["month"].dt.to_timestamp()
            fig_trend = px.line(monthly_yield, x="date", y="yield", title="Monthly Yield Trends")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No valid dates/yields to plot.")
    else:
        st.info("'date' and/or 'yield' columns are missing.")

with tab4:
    if _has_cols(data, ["area", "yield"]):
        color_kw = {"color": "crop_type"} if "crop_type" in data.columns else {}
        fig_scatter = px.scatter(data, x="area", y="yield", title="Yield vs Area", **color_kw)
        st.plotly_chart(fig_scatter, use_container_width=True)

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation matrix.")
    else:
        st.info("'yield' and/or 'area' columns are missing.")

with tab5:
    if _has_cols(data, ["date", "ndvi"]) and ("ADM2_NAME" in data.columns):
        tmp = data.dropna(subset=["date", "ndvi"]).copy()
        if not tmp.empty:
            fig_ndvi = px.line(tmp, x="date", y="ndvi", color="ADM2_NAME", title="NDVI Trends by Region")
            st.plotly_chart(fig_ndvi, use_container_width=True)
        else:
            st.info("No NDVI data available to plot.")
    else:
        st.info("Columns required: 'date', 'ndvi', and 'ADM2_NAME'.")

with tab6:
    if _has_cols(data, ["date", "spei_03"]) and ("ADM2_NAME" in data.columns):
        tmp = data.dropna(subset=["date", "spei_03"]).copy()
        if not tmp.empty:
            fig_spei = px.line(tmp, x="date", y="spei_03", color="ADM2_NAME", title="SPEI-03 Drought Index")
            st.plotly_chart(fig_spei, use_container_width=True)
        else:
            st.info("No SPEI data available to plot.")
    else:
        st.info("Columns required: 'date', 'spei_03', and 'ADM2_NAME'.")

with tab7:
    need = ["soil_sand", "soil_clay"]
    if _has_cols(data, need):
        size_col = "smap_moisture" if "smap_moisture" in data.columns else None
        color_col = "ADM2_NAME" if "ADM2_NAME" in data.columns else None
        fig_soil = px.scatter(
            data,
            x="soil_sand",
            y="soil_clay",
            color=color_col,
            size=size_col,
            title="Soil Texture vs Moisture",
            labels={"soil_sand": "Sand (%)", "soil_clay": "Clay (%)", "smap_moisture": "Moisture"},
        )
        st.plotly_chart(fig_soil, use_container_width=True)
    else:
        st.info("Columns required: 'soil_sand' and 'soil_clay'.")

# ---------------------------
# Data summary
# ---------------------------
st.markdown("---")
st.subheader("Data Summary")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Dataset Overview:**")
    st.write(f"- Total records: {len(data):,}")
    st.write(f"- Columns: {len(data.columns):,}")
    dtypes_count = data.dtypes.astype(str).value_counts().to_dict()
    st.write(f"- Data types: {dtypes_count}")

with col2:
    st.markdown("**Data Quality:**")
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.warning("Missing data detected:")
        for col, missing in missing_data[missing_data > 0].items():
            st.write(f"- {col}: {int(missing):,} missing values")
    else:
        st.success("No missing data detected")

# ---------------------------
# Export options
# ---------------------------
st.markdown("---")
st.subheader("Export Options")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Export Dashboard Report", use_container_width=True):
        st.info("Report export functionality would be implemented here")

with col2:
    if st.button("📈 Export Charts", use_container_width=True):
        st.info("Chart export functionality would be implemented here")

with col3:
    if st.button("📋 Export Data Summary", use_container_width=True):
        st.info("Data summary export functionality would be implemented here")

with col4:
    if st.button("📁 Export Excel File", use_container_width=True):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            data.to_excel(writer, index=False, sheet_name="Agricultural Data")
        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name="agricultural_dashboard_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
