"""
Conversational assistant page
"""

import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Assistant - Agro-Scan",
    page_icon="💬",
    layout="wide"
)

from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import get_user_id, format_date
from utils.styles import load_custom_css
from services.chatbot_service import ChatbotService
from services.database_service import DatabaseService
from utils.service_adapters import SyncChatbotService, SyncDatabaseService

load_custom_css()

# Title
st.title("💬 Agricultural Assistant")
st.markdown("Ask your questions about plants, diseases, treatments and agricultural practices")

# Initialiser les services
@st.cache_resource
def get_chatbot_service():
    async_service = ChatbotService()
    return SyncChatbotService(async_service)

@st.cache_resource
def get_database_service():
    async_service = DatabaseService()
    return SyncDatabaseService(async_service)

chatbot_service = get_chatbot_service()
database_service = get_database_service()

# Initialiser l'historique de chat
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Sidebar
st.sidebar.title("💬 Assistant")
st.sidebar.markdown("### Suggestions de questions")
suggestions = [
    "Comment traiter le mildiou ?",
    "Quand dois-je arroser mes plantes ?",
    "Quel engrais utiliser pour les tomates ?",
    "Comment prévenir les maladies ?",
    "Quels sont les signes de carence en azote ?"
]

for suggestion in suggestions:
    if st.sidebar.button(suggestion, use_container_width=True, key=f"sugg_{suggestion}"):
        st.session_state.user_input = suggestion
        st.rerun()

# Zone de chat
chat_container = st.container()

with chat_container:
    # Afficher l'historique
    if not st.session_state.chat_messages:
        st.info("👋 Bonjour ! Je suis votre assistant agricole. Comment puis-je vous aider ?")
        st.markdown("💡 **Exemples de questions:**")
        st.markdown("- Comment traiter le mildiou ?")
        st.markdown("- Quand arroser mes plantes ?")
        st.markdown("- Quel engrais utiliser ?")
    else:
        for message in st.session_state.chat_messages:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    if 'suggestions' in message and message['suggestions']:
                        st.write("**Suggestions:**")
                        for sugg in message['suggestions']:
                            if st.button(sugg, key=f"chat_sugg_{sugg}_{len(st.session_state.chat_messages)}"):
                                st.session_state.user_input = sugg
                                st.rerun()

# Zone de saisie
st.markdown("---")
user_input = st.chat_input("Posez votre question...")

# Si l'utilisateur a saisi quelque chose ou utilisé une suggestion
if 'user_input' in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input:
    # Ajouter le message de l'utilisateur
    st.session_state.chat_messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })
    
    # Générer la réponse
    with st.spinner("🤔 Réflexion en cours..."):
        try:
            # Récupérer l'historique utilisateur
            user_history = None
            try:
                user_history = database_service.get_user_chat_history(get_user_id(), 10)
            except:
                pass
            
            # Générer la réponse
            response = chatbot_service.generate_response(
                message=user_input,
                context=None,
                user_history=user_history
            )
            
            # Ajouter la réponse à l'historique
            st.session_state.chat_messages.append({
                'role': 'assistant',
                'content': response.response,
                'suggestions': response.suggestions,
                'timestamp': response.timestamp
            })
            
            # Sauvegarder dans la base de données
            try:
                database_service.save_chat_message(
                    user_id=get_user_id(),
                    message=user_input,
                    response=response.response,
                    context=None
                )
            except Exception as e:
                st.warning(f"⚠️ Le message n'a pas pu être sauvegardé: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
            st.session_state.chat_messages.append({
                'role': 'assistant',
                'content': "Je rencontre une difficulté technique. Pouvez-vous reformuler votre question ?",
                'timestamp': datetime.now().isoformat()
            })
    
    st.rerun()

# Bouton pour effacer l'historique
if st.session_state.chat_messages:
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

