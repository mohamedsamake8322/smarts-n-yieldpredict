"""
Service unifié pour utiliser le modèle Phi-3 local directement
Remplace les appels HTTP par des appels directs au modèle GGUF
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class Phi3LocalService:
    """Service pour utiliser le modèle Phi-3 local directement"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise le service avec le modèle Phi-3 local
        
        Args:
            model_path: Chemin vers le modèle GGUF. Si None, utilise le chemin par défaut.
        """
        if model_path is None:
            # Chemin par défaut
            base_dir = Path(__file__).parent.parent
            model_path = base_dir / "local_model" / "Phi-3-mini-4k-instruct-q4.gguf"
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable: {self.model_path}")
        
        # Charger le modèle
        logger.info(f"🔄 Chargement du modèle: {self.model_path}")
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=4096,  # Contexte de 4k tokens
                n_threads=4,  # Ajustez selon votre CPU
                verbose=False
            )
            logger.info("✅ Modèle Phi-3 chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Génère une réponse à partir d'un prompt ou de messages
        
        Args:
            prompt: Prompt simple (si messages est None)
            system_prompt: Prompt système
            messages: Liste de messages au format [{"role": "user", "content": "..."}]
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération
            top_p: Top-p sampling
            stop: Liste de séquences d'arrêt
        
        Returns:
            Texte généré
        """
        try:
            # Construire le prompt complet
            if messages:
                # Format conversationnel
                full_prompt = self._messages_to_prompt(messages, system_prompt)
            else:
                # Prompt simple
                if system_prompt:
                    full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
                else:
                    full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            
            # Générer
            response = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["<|end|>", "<|user|>", "<|system|>"],
                echo=False
            )
            
            # Extraire le texte généré
            generated_text = response['choices'][0]['text'].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération: {e}")
            return f"Erreur: {str(e)}"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Convertit une liste de messages en prompt formaté pour Phi-3
        
        Args:
            messages: Liste de messages [{"role": "user/assistant/system", "content": "..."}]
            system_prompt: Prompt système optionnel
        
        Returns:
            Prompt formaté
        """
        prompt_parts = []
        
        # Ajouter le prompt système si fourni
        if system_prompt:
            prompt_parts.append(f"<|system|>\n{system_prompt}<|end|>\n")
        
        # Traiter les messages
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}<|end|>\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}<|end|>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}<|end|>\n")
        
        # Ajouter le début de la réponse de l'assistant
        prompt_parts.append("<|assistant|>\n")
        
        return "".join(prompt_parts)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        """
        Interface compatible avec l'API OpenAI pour les scripts existants
        
        Args:
            messages: Liste de messages
            max_tokens: Nombre maximum de tokens
            temperature: Température
            top_p: Top-p sampling
        
        Returns:
            Dict au format OpenAI API
        """
        # Extraire le prompt système si présent
        system_prompt = None
        user_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                user_messages.append(msg)
        
        # Générer la réponse
        response_text = self.generate(
            prompt="",  # Pas de prompt simple, on utilise messages
            system_prompt=system_prompt,
            messages=user_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Formater la réponse au format OpenAI
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(str(messages).split()),  # Approximation
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(str(messages).split()) + len(response_text.split())
            }
        }








