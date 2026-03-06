"""
Service de chatbot conversationnel pour l'assistant agricole
Intègre un modèle local GGUF (phi-3-mini) ou OpenAI si configuré
"""

import logging
import os
from typing import Optional, List, Dict
from datetime import datetime
import requests

from models.chatbot_models import ChatMessage, ChatResponse

logger = logging.getLogger(__name__)

class ChatbotService:
    """Service de chatbot agricole intelligent"""
    
    def __init__(self, chat_model=None):
        """
        Initialise le service chatbot
        
        Args:
            chat_model: Instance de ChatModel (modèle Phi-3) - si None, sera chargé automatiquement
        """
        self.model_loaded = False
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("CHATBOT_MODEL", "gpt-3.5-turbo")
        self.use_openai = bool(self.api_key)
        
        # Modèle local (direct ou via API)
        self.chat_model = chat_model
        self.local_model_url = os.getenv("LOCAL_MODEL_URL", "http://127.0.0.1:5000/completion")

        self._initialize_model()
    
    def _initialize_model(self):
        """Initialise le modèle de chatbot"""
        try:
            if self.use_openai:
                logger.info("Service chatbot initialisé avec OpenAI")
            else:
                logger.info(f"Service chatbot initialisé en mode local (URL={self.local_model_url})")
            self.model_loaded = True
        except Exception as e:
            logger.warning(f"Erreur lors de l'initialisation du chatbot: {str(e)}")
            self.model_loaded = False
    
    def is_ready(self) -> bool:
        """Vérifie si le service est prêt"""
        return self.model_loaded
    
    async def generate_response(
        self,
        message: str,
        context: Optional[Dict] = None,
        user_history: Optional[List[ChatMessage]] = None
    ) -> ChatResponse:
        """
        Génère une réponse à partir du message de l'utilisateur
        """
        try:
            enriched_context = self._build_context(context, user_history)

            if self.use_openai:
                response_text = await self._generate_with_openai(message, enriched_context)
            else:
                response_text = await self._generate_with_local_model(message, enriched_context)

            response = ChatResponse(
                response=response_text,
                suggestions=self._generate_suggestions(message, context),
                context_used=bool(context),
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Réponse générée pour: {message[:50]}...")
            return response
        except Exception as e:
            logger.exception("Erreur lors de la génération de la réponse")
            return ChatResponse(
                response="Je rencontre une difficulté technique. Pouvez-vous reformuler votre question ?",
                suggestions=["Comment traiter une maladie ?", "Quand arroser ?", "Quel engrais utiliser ?"],
                context_used=False,
                timestamp=datetime.now().isoformat()
            )
    
    def _build_context(self, context: Optional[Dict], user_history: Optional[List[ChatMessage]]) -> Dict:
        enriched = {
            "domain": "agriculture",
            "target_audience": "producteurs agricoles",
            "language": "français",
            "tone": "pédagogique et accessible"
        }
        if context:
            enriched.update(context)
        if user_history:
            recent_history = user_history[-5:] if len(user_history) > 5 else user_history
            enriched["recent_history"] = [{"role": msg.role, "content": msg.content} for msg in recent_history]
        return enriched
    
    async def _generate_with_openai(self, message: str, context: Dict) -> str:
        # TODO: Implémenter OpenAI si nécessaire
        return await self._generate_with_local_model(message, context)
    
    async def _generate_with_local_model(self, message: str, context: Dict) -> str:
        """
        Utilise le modèle local directement ou via API
        """
        # Essayer d'abord le service local direct
        try:
            from services.local_model_service import get_local_model_service
            local_service = get_local_model_service()
            
            if local_service.is_ready():
                prompt = self._build_system_prompt(context) + f"\n\nUtilisateur: {message}\nAssistant:"
                response = local_service.generate(prompt, temperature=0.2, max_tokens=300)
                if response:
                    logger.info("✅ Réponse générée par le modèle local direct")
                    return response
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Service local direct non disponible: {str(e)}")
        
        # Fallback: Essayer l'API locale
        try:
            prompt = self._build_system_prompt(context) + f"\n\nUtilisateur: {message}\nAssistant:"
            payload = {"prompt": prompt, "temperature": 0.2, "max_tokens": 200}
            resp = requests.post(self.local_model_url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("content", "")
                if content:
                    return content.strip()
            logger.warning(f"Modèle local API retourné un code {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning("Modèle local API non disponible.")
        except requests.exceptions.Timeout:
            logger.warning("Timeout lors de l'appel au modèle local API.")
        except Exception as e:
            logger.error(f"Erreur appel modèle local API: {str(e)}")
        
        # Mode simulation si le modèle local n'est pas disponible
        return self._generate_simulation_response(message, context)
    
    def _generate_simulation_response(self, message: str, context: Dict) -> str:
        """Génère une réponse de simulation basée sur des règles"""
        message_lower = message.lower()
        
        # Réponses basées sur des mots-clés
        if any(word in message_lower for word in ["mildiou", "maladie", "traitement"]):
            return """Pour traiter le mildiou, voici quelques recommandations :

1. **Traitement préventif** : Appliquez de la bouillie bordelaise tous les 10-15 jours
2. **Traitement curatif** : Retirez les feuilles atteintes et pulvérisez avec un fongicide systémique
3. **Prévention** : Évitez l'humidité excessive sur les feuilles, espacez bien les plants
4. **Alternative biologique** : Utilisez une décoction de prêle ou du bicarbonate de soude

⚠️ Note : Pour des conseils plus précis, configurez le modèle IA local ou utilisez OpenAI."""
        
        elif any(word in message_lower for word in ["arroser", "eau", "irrigation"]):
            return """Pour l'arrosage des plantes :

1. **Fréquence** : Arrosez tôt le matin ou en fin de journée pour éviter l'évaporation
2. **Quantité** : Arrosez abondamment mais moins fréquemment pour encourager les racines profondes
3. **Méthode** : Préférez l'arrosage au pied plutôt que sur les feuilles
4. **Paillage** : Utilisez du paillage pour conserver l'humidité du sol

⚠️ Note : Les besoins varient selon la plante et le climat. Configurez le modèle IA pour des conseils personnalisés."""
        
        elif any(word in message_lower for word in ["engrais", "fertilisant", "nutriments"]):
            return """Pour la fertilisation :

1. **Engrais organiques** : Compost, fumier bien décomposé, purin d'ortie
2. **Engrais minéraux** : NPK équilibré selon les besoins de la plante
3. **Application** : Respectez les doses recommandées, mieux vaut moins que trop
4. **Période** : Appliquez pendant la période de croissance active

⚠️ Note : Pour des recommandations spécifiques à votre culture, configurez le modèle IA."""
        
        else:
            return f"""Je comprends votre question : "{message}"

Pour obtenir des réponses détaillées et personnalisées, vous pouvez :
1. Configurer le modèle IA local (GGUF) dans local_model/
2. Utiliser OpenAI en configurant OPENAI_API_KEY dans .env
3. Consulter la documentation agricole spécialisée

En mode simulation, je peux répondre aux questions sur :
- Traitements des maladies (mildiou, oïdium, etc.)
- Arrosage et irrigation
- Engrais et fertilisation
- Soins généraux des plantes"""
    
    def _build_system_prompt(self, context: Dict) -> str:
        prompt = """Tu es un assistant agricole intelligent et bienveillant.
Ton rôle :
- Répondre aux questions sur les plantes, maladies, soins et pratiques agricoles
- Donner des conseils pratiques, accessibles et adaptés
- Proposer des solutions biologiques quand possible
- Être pédagogique et encourager les bonnes pratiques
"""
        if context.get("recent_detection"):
            prompt += f"\nDétection récente : {context['recent_detection']}"
        return prompt
    
    def _generate_suggestions(self, message: str, context: Optional[Dict]) -> List[str]:
        suggestions = [
            "Comment traiter cette maladie ?",
            "Quand dois-je arroser ?",
            "Quel engrais utiliser ?"
        ]
        if context and context.get("plant_detected"):
            suggestions.insert(0, f"En savoir plus sur {context['plant_detected']}")
        return suggestions
