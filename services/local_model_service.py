"""
Service pour utiliser directement le modèle local Phi-3
Sans passer par FastAPI
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class LocalModelService:
    """Service pour charger et utiliser le modèle Phi-3 local"""
    
    def __init__(self):
        self.model = None
        self.model_path = Path(__file__).parent.parent / "local_model" / "Phi-3-mini-4k-instruct-q4.gguf"
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle GGUF si disponible"""
        try:
            if self.model_path.exists():
                try:
                    from llama_cpp import Llama
                    
                    logger.info(f"Chargement du modèle depuis {self.model_path}")
                    self.model = Llama(
                        model_path=str(self.model_path),
                        n_ctx=2048,
                        n_threads=min(8, os.cpu_count() or 4),
                        n_batch=512,
                        verbose=False
                    )
                    self.model_loaded = True
                    logger.info("✅ Modèle Phi-3 chargé avec succès")
                except ImportError:
                    logger.warning("⚠️ llama-cpp-python non installé. Installez-le avec: pip install llama-cpp-python")
                    self.model_loaded = False
                except Exception as e:
                    logger.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
                    self.model_loaded = False
            else:
                logger.warning(f"⚠️ Modèle non trouvé à {self.model_path}")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
            self.model_loaded = False
    
    def is_ready(self) -> bool:
        """Vérifie si le modèle est prêt"""
        return self.model_loaded and self.model is not None
    
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 200) -> Optional[str]:
        """
        Génère une réponse à partir du prompt
        
        Args:
            prompt: Le prompt d'entrée
            temperature: Température pour la génération (0.0-1.0)
            max_tokens: Nombre maximum de tokens à générer
        
        Returns:
            La réponse générée ou None en cas d'erreur
        """
        if not self.is_ready():
            return None
        
        try:
            output = self.model(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["</s>", "\n\n\n"],
                echo=False
            )
            
            if output and "choices" in output and len(output["choices"]) > 0:
                return output["choices"][0]["text"].strip()
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération: {str(e)}")
            return None

# Instance globale
_local_model_service = None

def get_local_model_service() -> LocalModelService:
    """Retourne l'instance du service de modèle local (singleton)"""
    global _local_model_service
    if _local_model_service is None:
        _local_model_service = LocalModelService()
    return _local_model_service








