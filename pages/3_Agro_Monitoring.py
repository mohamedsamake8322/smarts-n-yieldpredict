import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import urllib.parse
from utils import AdvancedAI

# Configuration de la page
st.set_page_config(page_title="Agro Monitoring", page_icon="🛰️", layout="wide")
st.title("🛰️ Suivi Agro-Climatique Intégré")
st.markdown("### Visualisation du climat, de la végétation et du sol par région")

# 📁 Chargement des fichiers
data_path = "C:\\smarts-n-yieldpredict.git\\data"
spei_file = os.path.join(data_path, "SPEI_Mali_ADM2_20250821_1546.csv")
modis_file = os.path.join(data_path, "MODIS_VI_Mali_2020_2025_mali_20250821_1503.csv")
soil_file = os.path.join(data_path, "fusion_completesoil.csv")
wapor_file = os.path.join(data_path, "WAPOR_fusion_long_clean_clean.csv")  # optionnel si tu l'as

# Chargement des données (safe)
def safe_read(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Impossible de lire {os.path.basename(csv_path)} : {e}")
        return pd.DataFrame()

spei_df = safe_read(spei_file)
modis_df = safe_read(modis_file)
soil_df = safe_read(soil_file)
wapor_df = safe_read(wapor_file) if os.path.exists(wapor_file) else pd.DataFrame()

# Normaliser noms de colonnes en minuscules pour éviter les erreurs de casse
for df in (spei_df, modis_df, soil_df, wapor_df):
    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()

# Création de la colonne 'date' dans SPEI si possible
if {"year", "month"}.issubset(spei_df.columns):
    try:
        spei_df["date"] = pd.to_datetime(spei_df["year"].astype(str) + "-" + spei_df["month"].astype(str) + "-01")
    except Exception:
        spei_df["date"] = pd.NaT
else:
    spei_df["date"] = pd.NaT  # fallback

# --- Construire champ adm2_name si absent dans spei_df en mappant avec modis/wapor sur adm2_id
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
            # fallback : utiliser adm2_id comme texte si pas d'info de nom
            spei_df["adm2_name"] = spei_df["adm2_id"].astype(str)
    else:
        # aucune info admin2 dans ce df
        spei_df["adm2_name"] = pd.NA

# Construire liste des régions possibles (unique, non-NA)
regions = []
if "adm2_name" in spei_df.columns:
    regions = pd.Series(spei_df["adm2_name"].dropna().unique()).astype(str).tolist()

# si toujours vide, regarder modis/wapor
if not regions:
    if (not modis_df.empty) and "adm2_name" in modis_df.columns:
        regions = pd.Series(modis_df["adm2_name"].dropna().unique()).astype(str).tolist()
    elif (not wapor_df.empty) and "adm2_name" in wapor_df.columns:
        regions = pd.Series(wapor_df["adm2_name"].dropna().unique()).astype(str).tolist()

# si toujours vide -> avertissement et sortie (sélectbox vide)
if not regions:
    st.error("Aucune région ADM2 trouvée dans les fichiers SPEI/MODIS/WAPOR. Vérifie tes fichiers sources.")
    st.stop()

regions = sorted(regions)
# 🔍 Sélection de la région
region = st.selectbox("📍 Sélectionner une région", regions)

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

# 🧭 Onglets
tab1, tab2, tab3 = st.tabs(["🌦 Climat & Végétation", "🧪 Profil de Sol", "🚨 Alerte Agricole"])

# Initialize diagnostics variables so tab3 can read them even if tab1/2 are vides
ndvi = None
spei_val = None
ph = None
clay = None
carbon = None

# 🌦 Onglet 1 : Climat & Végétation
with tab1:
    st.subheader("🌦 Tendances climatiques et stress végétatif")

    region_spei = filter_by_region(spei_df, region)
    region_modis = filter_by_region(modis_df, region)

    # normaliser noms de colonnes pour les valeurs utilisées
    # spei value peut être 'value' ou 'value' lowercase après normalisation
    spei_value_col = "value" if "value" in region_spei.columns else ("value" if "value" in spei_df.columns else None)

    # fusion sur adm2_name + year si possible (après normalisation, 'adm2_name' existe)
    if not region_spei.empty and not region_modis.empty and "adm2_name" in region_spei.columns and "adm2_name" in region_modis.columns:
        # assure presence de 'year' colonnes
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

    # Graphique SPEI
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
        st.info("Pas assez de données temporelles SPEI pour dessiner le graphique.")

    # Graphique NDVI vs SPEI
    if not fusion_df.empty and "ndvi_mean" in fusion_df.columns and "value" in fusion_df.columns:
        # assure tri par année si possible
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

        # Interprétation simplifiée — prendre la dernière ligne qui a ndvi & value
        try:
            latest = fusion_df.dropna(subset=["ndvi_mean", "value"]).sort_values("year").iloc[-1]
            ndvi = float(latest["ndvi_mean"])
            spei_val = float(latest["value"])
        except Exception:
            ndvi = None
            # spei_val peut être déjà défini par le graphique SPEI
    else:
        st.info("Pas assez de données fusionnées (NDVI + SPEI) pour comparer.")

    # Interprétation & résumé WhatsApp (ne s'affiche que si on a au moins une valeur)
    st.markdown("### 🧠 Interprétation")
    conseil = "Données insuffisantes pour diagnostic."
    if (spei_val is not None) or (ndvi is not None):
        # apply rules defensively
        try:
            if (spei_val is not None and spei_val < -1) and (ndvi is not None and ndvi < 0.3):
                st.error("🚨 Stress végétatif confirmé : sécheresse + faible NDVI")
                conseil = "🚨 Stress confirmé : sol sec + plantes faibles"
            elif (spei_val is not None and spei_val < -1):
                st.warning("⚠️ Sécheresse détectée, surveiller la végétation")
                conseil = "⚠️ Sol sec : surveiller les cultures"
            elif (ndvi is not None and ndvi < 0.3):
                st.warning("🌿 NDVI faible - stress végétatif possible")
                conseil = "🌿 Plantes faibles : vérifier l'irrigation"
            else:
                st.success("✅ Conditions normales")
                conseil = "✅ Tout va bien : bonne humidité et végétation"
        except Exception:
            st.info("Impossible d'interpréter les valeurs — vérifie les données.")
    else:
        st.info("Pas de SPEI/NDVI disponibles pour donner une interprétation.")

    st.markdown("### 🔊 Résumé simplifié pour producteurs")
    st.markdown(conseil)
    # build whatsapp message using available numeric values (format safely)
    msg_parts = [f"🌿 Région : {region}"]
    if ndvi is not None:
        msg_parts.append(f"NDVI : {ndvi:.2f}")
    if spei_val is not None:
        msg_parts.append(f"SPEI : {spei_val:.2f}")
    msg_parts.append(conseil)
    message = "\n".join(msg_parts)
    encoded_msg = urllib.parse.quote(message)
    whatsapp_url = f"https://wa.me/?text={encoded_msg}"
    st.markdown(f"[📱 Envoyer sur WhatsApp]({whatsapp_url})", unsafe_allow_html=True)

# 🧪 Onglet 2 : Profil de Sol
with tab2:
    st.subheader("🧪 Propriétés du sol")

    # soil_df a des colonnes comme ph_adm2_name, claycontent_adm2_name, carbonorganic_adm2_name, etc.
    # On va filtrer en essayant ph_adm2_name puis d'autres colonnes communes
    soil_region = pd.DataFrame()
    if not soil_df.empty:
        # trouver une colonne adm2_name dans soil_df
        adm2_name_cols = [c for c in soil_df.columns if "adm2_name" in c]
        if adm2_name_cols:
            # prendre la colonne la plus simple si existe 'ph_adm2_name' preferée
            col_to_use = None
            for prefer in ["ph_adm2_name", "sand_adm2_name", "claycontent_adm2_name", "carbonorganic_adm2_name"]:
                if prefer in soil_df.columns:
                    col_to_use = prefer
                    break
            if not col_to_use:
                col_to_use = adm2_name_cols[0]
            soil_region = soil_df[soil_df[col_to_use].astype(str) == str(region)]
        else:
            # essayer mapping via adm2_id (si possible)
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
        st.info("Aucun profil de sol trouvé pour cette région dans fusion_completesoil.csv.")
    else:
        # prendre la première ligne utile
        row = soil_region.iloc[0]
        ph = row.get("ph_mean_0_20_mean", None)
        clay = row.get("claycontent_mean_0_20_mean", None)
        carbon = row.get("carbonorganic_mean_0_20_mean", None)

        col1, col2, col3 = st.columns(3)
        col1.metric("pH (0–20cm)", f"{float(ph):.2f}" if pd.notna(ph) else "N/A")
        col2.metric("Argile (%)", f"{float(clay):.1f}" if pd.notna(clay) else "N/A")
        col3.metric("Carbone organique", f"{float(carbon):.2f}" if pd.notna(carbon) else "N/A")

        st.markdown("### ⚠️ Alertes sol")
        if pd.notna(ph) and float(ph) < 5.5:
            st.warning("🧪 Sol acide - risque pour certaines cultures")
        if pd.notna(clay) and float(clay) > 40:
            st.warning("🧱 Sol très argileux - drainage lent")
        if pd.notna(carbon) and float(carbon) < 1.0:
            st.warning("🌱 Faible matière organique - fertilité limitée")
        if (
            pd.notna(ph) and float(ph) >= 5.5 and
            pd.notna(clay) and float(clay) <= 40 and
            pd.notna(carbon) and float(carbon) >= 1.0
        ):
            st.success("✅ Sol favorable à la culture")

# 🚨 Onglet 3 : Alerte Agricole
with tab3:
    st.subheader("🚨 Synthèse des risques agricoles")

    alertes = []
    # utiliser les variables calculées dans les onglets précédents (si présentes)
    if (spei_val is not None and ndvi is not None) and (spei_val < -1 and ndvi < 0.3):
        alertes.append("🌵 Sécheresse + stress végétatif")
    if pd.notna(ph) and float(ph) < 5.5:
        alertes.append("🧪 Sol acide")
    if pd.notna(clay) and float(clay) > 40:
        alertes.append("🧱 Sol très argileux")
    if pd.notna(carbon) and float(carbon) < 1.0:
        alertes.append("🌱 Faible fertilité")

    if alertes:
        st.error("🚨 Risques détectés :")
        for a in alertes:
            st.markdown(f"- {a}")
    else:
        st.success("✅ Aucun risque majeur détecté")

    st.markdown("### 📱 Résumé producteur")
    diagnostic = "\n".join(alertes) if alertes else "✅ Conditions favorables"
    message_final = f"📍 Région : {region}\n{diagnostic}"
    st.markdown(message_final)

    encoded_final = urllib.parse.quote(message_final)
    whatsapp_final = f"https://wa.me/?text={encoded_final}"
    st.markdown(f"[📤 Envoyer ce diagnostic sur WhatsApp]({whatsapp_final})", unsafe_allow_html=True)
