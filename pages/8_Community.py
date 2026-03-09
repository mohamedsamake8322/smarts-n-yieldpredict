"""
Page de la communauté agricole
"""

import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="Communauté - Agro-Scan",
    page_icon="💬",
    layout="wide"
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.styles import load_custom_css, load_mobile_css

load_custom_css()
load_mobile_css()

st.title("💬 Communauté Agricole")

tab1, tab2, tab3 = st.tabs(["📢 Discussions", "❓ Questions", "💡 Conseils"])

with tab1:
    st.subheader("📢 Discussions de la communauté")
    
    # Recherche
    search = st.text_input("🔍 Rechercher dans les discussions")
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        filter_crop = st.selectbox("Filtrer par culture", ["Toutes", "Tomate", "Maïs", "Riz"])
    with col2:
        sort_by = st.selectbox("Trier par", ["Récent", "Populaire", "Pertinence"])
    
    # Discussions simulées
    discussions = [
        {
            "title": "Comment traiter le mildiou sur mes tomates ?",
            "author": "Amadou D.",
            "crop": "Tomate",
            "replies": 12,
            "views": 145,
            "date": "Il y a 2 heures"
        },
        {
            "title": "Meilleur moment pour planter le maïs ?",
            "author": "Fatou K.",
            "crop": "Maïs",
            "replies": 8,
            "views": 98,
            "date": "Il y a 5 heures"
        },
        {
            "title": "Engrais organique vs minéral - Avis ?",
            "author": "Moussa T.",
            "crop": "Général",
            "replies": 25,
            "views": 312,
            "date": "Il y a 1 jour"
        }
    ]
    
    for disc in discussions:
        with st.expander(f"💬 {disc['title']}", expanded=False):
            st.write(f"**Auteur:** {disc['author']} | **Culture:** {disc['crop']}")
            st.write(f"💬 {disc['replies']} réponses | 👁️ {disc['views']} vues | ⏰ {disc['date']}")
            if st.button("Voir la discussion", key=f"view_{disc['title']}"):
                st.info("Fonctionnalité à venir - Ouverture de la discussion complète")

with tab2:
    st.subheader("❓ Posez votre question")
    
    question = st.text_area(
        "Votre question",
        placeholder="Ex: Comment prévenir le mildiou sur mes tomates ?",
        height=100
    )
    
    crop_question = st.selectbox(
        "Culture concernée",
        ["Général", "Tomate", "Maïs", "Riz", "Manioc", "Banane", "Cacao", "Café"]
    )
    
    if st.button("Publier la question", type="primary"):
        if question:
            st.success("✅ Votre question a été publiée ! La communauté va vous répondre.")
        else:
            st.warning("⚠️ Veuillez saisir une question")

with tab3:
    st.subheader("💡 Conseils de la communauté")
    
    st.markdown("""
    **Conseils les plus utiles partagés par la communauté:**
    
    1. **Rotation des cultures** - Changez l'emplacement de vos cultures chaque année
    2. **Paillage** - Utilisez du paillage pour conserver l'humidité et réduire les mauvaises herbes
    3. **Compost** - Faites votre propre compost avec les déchets organiques
    4. **Surveillance** - Inspectez régulièrement vos plants pour détecter les problèmes tôt
    5. **Eau** - Arrosez tôt le matin ou en fin de journée pour éviter l'évaporation
    """)
    
    if st.button("Partager un conseil", type="primary"):
        st.info("Fonctionnalité à venir - Partagez vos meilleurs conseils avec la communauté")

