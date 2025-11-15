import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import urllib.parse
from utils import AdvancedAI

# Page configuration
st.set_page_config(page_title="Agro Monitoring", page_icon="🛰️", layout="wide")
st.title("🛰️ Integrated Agro-Climatic Monitoring")
st.markdown("### Visualization of climate, vegetation, and soil by region")

# 📁 Loading files
data_path = "C:\\smarts-n-yieldpredict.git\\data"
spei_file = os.path.join(data_path, "SPEI_Mali_ADM2_20250821_1546.csv")
modis_file = os.path.join(data_path, "MODIS_VI_Mali_2020_2025_mali_20250821_1503.csv")
soil_file = os.path.join(data_path, "fusion_completesoil.csv")
wapor_file = os.path.join(data_path, "WAPOR_fusion_long_clean_clean.csv")  # optional if available

# Data loading (safe)
def safe_read(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Unable to read {os.path.basename(csv_path)} : {e}")
        return pd.DataFrame()

spei_df = safe_read(spei_file)
modis_df = safe_read(modis_file)
soil_df = safe_read(soil_file)
wapor_df = safe_read(wapor_file) if os.path.exists(wapor_file) else pd.DataFrame()

# Normalize column names to lowercase to avoid case errors
for df in (spei_df, modis_df, soil_df, wapor_df):
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()

# Create 'date' column in SPEI if possible
if {"year", "month"}.issubset(spei_df.columns):
    try:
        spei_df["date"] = pd.to_datetime(spei_df["year"].astype(str) + "-" + spei_df["month"].astype(str) + "-01")
    except Exception:
        spei_df["date"] = pd.NaT
else:
    spei_df["date"] = pd.NaT  # fallback

# --- Build adm2_name field if absent in spei_df by mapping with modis/wapor on adm2_id
if "adm2_name" not in spei_df.columns:
    if "adm2_id" in spei_df.columns:
        mapping = None
        if (not modis_df.empty) and {"adm2_id", "adm2_name"}.issubset(modis_df.columns):
            mapping = modis_df[["adm2_id", "adm2_name"]].drop_duplicates().set_index("adm2_id")["adm2_name"]
        elif (not wapor_df.empty) and {"adm2_id", "adm2_name"}.issubset(wapor_df.columns):
            mapping = wapor_df[["adm2_id", "adm2_name"]].drop_duplicates().set_index("adm2_id")["adm2_name"]

        if mapping is not None:
            spei_df["adm2_name"] = spei_df["adm2_id"].map(mapping)
        else:
            # fallback: use adm2_id as text if no name info
            spei_df["adm2_name"] = spei_df["adm2_id"].astype(str)
    else:
        # no admin2 info in this df
        spei_df["adm2_name"] = pd.NA

# Build list of possible regions (unique, non-NA)
regions = []
if "adm2_name" in spei_df.columns:
    regions = pd.Series(spei_df["adm2_name"].dropna().unique()).astype(str).tolist()

# if still empty, check modis/wapor
if not regions:
    if (not modis_df.empty) and "adm2_name" in modis_df.columns:
        regions = pd.Series(modis_df["adm2_name"].dropna().unique()).astype(str).tolist()
    elif (not wapor_df.empty) and "adm2_name" in wapor_df.columns:
        regions = pd.Series(wapor_df["adm2_name"].dropna().unique()).astype(str).tolist()

# if still empty -> warning and exit (empty selectbox)
if not regions:
    st.error("No ADM2 region found in SPEI/MODIS/WAPOR files. Check your source files.")
    st.stop()

regions = sorted(regions)
# 🔍 Region selection
region = st.selectbox("📍 Select a region", regions)

# helper function to filter by adm2 name across different dframes that may use different column names
def filter_by_region(df, region_name):
    if df.empty:
        return df
    # try common adm2 name columns
    for col in ["adm2_name", "aluminium multi-profondeur_adm2_name", "calcium_adm2_name"]:
        if col in df.columns:
            # generic match
            return df[df[col].astype(str) == str(region_name)]
    # some files may use adm2_id only; try mapping via modis
    if "adm2_id" in df.columns:
        # try to map region_name -> id using modis or wapor
        mapping_df = None
        if (not modis_df.empty) and {"adm2_id", "adm2_name"}.issubset(modis_df.columns):
            mapping_df = modis_df[["adm2_id", "adm2_name"]].dropna().drop_duplicates()
        elif (not wapor_df.empty) and {"adm2_id", "adm2_name"}.issubset(wapor_df.columns):
            mapping_df = wapor_df[["adm2_id", "adm2_name"]].dropna().drop_duplicates()
        if mapping_df is not None and not mapping_df.empty:
            possible_ids = mapping_df[mapping_df["adm2_name"].astype(str) == str(region_name)]["adm2_id"].unique()
            return df[df["adm2_id"].isin(possible_ids)]
    # last fallback: return empty
    return df.iloc[0:0]

# 🧭 Tabs
tab1, tab2, tab3 = st.tabs(["🌦 Climate & Vegetation", "🧪 Soil Profile", "🚨 Agricultural Alert"])

# Initialize diagnostics variables so tab3 can read them even if tab1/2 are empty
ndvi = None
spei_val = None
ph = None
clay = None
carbon = None

# 🌦 Tab 1 : Climate & Vegetation
with tab1:
    st.subheader("🌦 Climate trends and vegetative stress")

    region_spei = filter_by_region(spei_df, region)
    region_modis = filter_by_region(modis_df, region)

    # normalize column names for values used
    # spei value can be 'value' or 'value' lowercase after normalization
    spei_value_col = "value" if "value" in region_spei.columns else ("value" if "value" in spei_df.columns else None)

    # merge on adm2_name + year if possible (after normalization, 'adm2_name' exists)
    if not region_spei.empty and not region_modis.empty and "adm2_name" in region_spei.columns and "adm2_name" in region_modis.columns:
        # ensure presence of 'year' columns
        common_on = []
        if "adm2_name" in region_spei.columns and "adm2_name" in region_modis.columns:
            common_on.append("adm2_name")
        if "year" in region_spei.columns and "year" in region_modis.columns:
            common_on.append("year")
        try:
            fusion_df = pd.merge(region_spei, region_modis, on=common_on, how="inner") if common_on else pd.DataFrame()
        except Exception:
            fusion_df = pd.DataFrame()
    else:
        fusion_df = pd.DataFrame()

    # SPEI Chart
    fig_spei = go.Figure()
    if ("value" in region_spei.columns) and (not region_spei["date"].isna().all()):
        y_vals = region_spei["value"].astype(float)
        fig_spei.add_trace(go.Scatter(x=region_spei["date"], y=y_vals, mode='lines', name="SPEI"))
        fig_spei.update_layout(title=f"SPEI Trend for {region}", xaxis_title="Date", yaxis_title="SPEI Index", yaxis=dict(range=[-3, 3]))
        st.plotly_chart(fig_spei, use_container_width=True)
        # take latest spei for diagnostics (most recent date)
        try:
            latest_spei_row = region_spei.sort_values("date").dropna(subset=["date"]).iloc[-1]
            spei_val = float(latest_spei_row["value"])
        except Exception:
            spei_val = None
    else:
        st.info("Not enough temporal SPEI data to draw the chart.")

    # NDVI vs SPEI Chart
    if not fusion_df.empty and "ndvi_mean" in fusion_df.columns and "value" in fusion_df.columns:
        # ensure sorting by year if possible
        x = fusion_df["year"] if "year" in fusion_df.columns else fusion_df.index
        fig_modis = go.Figure()
        fig_modis.add_trace(go.Scatter(x=x, y=fusion_df["ndvi_mean"].astype(float), mode="lines", name="NDVI"))
        fig_modis.add_trace(go.Scatter(x=x, y=fusion_df["value"].astype(float), mode="lines", name="SPEI", yaxis="y2"))
        fig_modis.update_layout(
            title=f"NDVI vs SPEI for {region}",
            xaxis_title="Year",
            yaxis=dict(title="NDVI", range=[0, 1]),
            yaxis2=dict(title="SPEI", overlaying="y", side="right", range=[-3, 3])
        )
        st.plotly_chart(fig_modis, use_container_width=True)

        # Simplified interpretation — take the last row that has ndvi & value
        try:
            latest = fusion_df.dropna(subset=["ndvi_mean", "value"]).sort_values("year").iloc[-1]
            ndvi = float(latest["ndvi_mean"])
            spei_val = float(latest["value"])
        except Exception:
            ndvi = None
            # spei_val may already be defined by SPEI chart
    else:
        st.info("Not enough merged data (NDVI + SPEI) to compare.")

    # Interpretation & WhatsApp summary (only displays if we have at least one value)
    st.markdown("### 🧠 Interpretation")
    advice = "Insufficient data for diagnosis."
    if (spei_val is not None) or (ndvi is not None):
        # apply rules defensively
        try:
            if (spei_val is not None and spei_val < -1) and (ndvi is not None and ndvi < 0.3):
                st.error("🚨 Vegetative stress confirmed: drought + low NDVI")
                advice = "🚨 Stress confirmed: dry soil + weak plants"
            elif (spei_val is not None and spei_val < -1):
                st.warning("⚠️ Drought detected, monitor vegetation")
                advice = "⚠️ Dry soil: monitor crops"
            elif (ndvi is not None and ndvi < 0.3):
                st.warning("🌿 Low NDVI - possible vegetative stress")
                advice = "🌿 Weak plants: check irrigation"
            else:
                st.success("✅ Normal conditions")
                advice = "✅ All good: good moisture and vegetation"
        except Exception:
            st.info("Unable to interpret values — check data.")
    else:
        st.info("No SPEI/NDVI available to provide interpretation.")

    st.markdown("### 🔊 Simplified summary for producers")
    st.markdown(advice)
    # build whatsapp message using available numeric values (format safely)
    msg_parts = [f"🌿 Region : {region}"]
    if ndvi is not None:
        msg_parts.append(f"NDVI : {ndvi:.2f}")
    if spei_val is not None:
        msg_parts.append(f"SPEI : {spei_val:.2f}")
    msg_parts.append(advice)
    message = "\n".join(msg_parts)
    encoded_msg = urllib.parse.quote(message)
    whatsapp_url = f"https://wa.me/?text={encoded_msg}"
    st.markdown(f"[📱 Send on WhatsApp]({whatsapp_url})", unsafe_allow_html=True)

# 🧪 Tab 2 : Soil Profile
with tab2:
    st.subheader("🧪 Soil properties")

    # soil_df has columns like ph_adm2_name, claycontent_adm2_name, carbonorganic_adm2_name, etc.
    # We will filter by trying ph_adm2_name then other common columns
    soil_region = pd.DataFrame()
    if not soil_df.empty:
        # find an adm2_name column in soil_df
        adm2_name_cols = [c for c in soil_df.columns if "adm2_name" in c]
        if adm2_name_cols:
            # take the simplest column if exists 'ph_adm2_name' preferred
            col_to_use = None
            for prefer in ["ph_adm2_name", "sand_adm2_name", "claycontent_adm2_name", "carbonorganic_adm2_name"]:
                if prefer in soil_df.columns:
                    col_to_use = prefer
                    break
            if not col_to_use:
                col_to_use = adm2_name_cols[0]
            soil_region = soil_df[soil_df[col_to_use].astype(str) == str(region)]
        else:
            # try mapping via adm2_id (if possible)
            if "adm2_id" in soil_df.columns:
                # map region->id using modis/wapor
                mapping_df = None
                if (not modis_df.empty) and {"adm2_id", "adm2_name"}.issubset(modis_df.columns):
                    mapping_df = modis_df[["adm2_id", "adm2_name"]].dropna().drop_duplicates()
                elif (not wapor_df.empty) and {"adm2_id", "adm2_name"}.issubset(wapor_df.columns):
                    mapping_df = wapor_df[["adm2_id", "adm2_name"]].dropna().drop_duplicates()
                if mapping_df is not None and not mapping_df.empty:
                    possible_ids = mapping_df[mapping_df["adm2_name"].astype(str) == str(region)]["adm2_id"].unique()
                    soil_region = soil_df[soil_df["adm2_id"].isin(possible_ids)]

    if soil_region.empty:
        st.info("No soil profile found for this region in fusion_completesoil.csv.")
    else:
        # take the first useful row
        row = soil_region.iloc[0]
        ph = row.get("ph_mean_0_20_mean", None)
        clay = row.get("claycontent_mean_0_20_mean", None)
        carbon = row.get("carbonorganic_mean_0_20_mean", None)

        col1, col2, col3 = st.columns(3)
        col1.metric("pH (0–20cm)", f"{float(ph):.2f}" if pd.notna(ph) else "N/A")
        col2.metric("Clay (%)", f"{float(clay):.1f}" if pd.notna(clay) else "N/A")
        col3.metric("Organic carbon", f"{float(carbon):.2f}" if pd.notna(carbon) else "N/A")

        st.markdown("### ⚠️ Soil alerts")
        if pd.notna(ph) and float(ph) < 5.5:
            st.warning("🧪 Acidic soil - risk for certain crops")
        if pd.notna(clay) and float(clay) > 40:
            st.warning("🧱 Very clayey soil - slow drainage")
        if pd.notna(carbon) and float(carbon) < 1.0:
            st.warning("🌱 Low organic matter - limited fertility")
        if (
            pd.notna(ph) and float(ph) >= 5.5 and
            pd.notna(clay) and float(clay) <= 40 and
            pd.notna(carbon) and float(carbon) >= 1.0
        ):
            st.success("✅ Soil favorable for cultivation")

# 🚨 Tab 3 : Agricultural Alert
with tab3:
    st.subheader("🚨 Agricultural risk synthesis")

    alerts = []
    # use variables calculated in previous tabs (if present)
    if (spei_val is not None and ndvi is not None) and (spei_val < -1 and ndvi < 0.3):
        alerts.append("🌵 Drought + vegetative stress")
    if pd.notna(ph) and float(ph) < 5.5:
        alerts.append("🧪 Acidic soil")
    if pd.notna(clay) and float(clay) > 40:
        alerts.append("🧱 Very clayey soil")
    if pd.notna(carbon) and float(carbon) < 1.0:
        alerts.append("🌱 Low fertility")

    if alerts:
        st.error("🚨 Risks detected :")
        for a in alerts:
            st.markdown(f"- {a}")
    else:
        st.success("✅ No major risk detected")

    st.markdown("### 📱 Producer summary")
    diagnostic = "\n".join(alerts) if alerts else "✅ Favorable conditions"
    message_final = f"📍 Region : {region}\n{diagnostic}"
    st.markdown(message_final)

    encoded_final = urllib.parse.quote(message_final)
    whatsapp_final = f"https://wa.me/?text={encoded_final}"
    st.markdown(f"[📤 Send this diagnosis on WhatsApp]({whatsapp_final})", unsafe_allow_html=True)
