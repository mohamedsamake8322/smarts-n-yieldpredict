"""
Robust Error Handler Module

Provides centralized error handling and recovery mechanisms for the Smart Agriculture system.
"""

import traceback
import logging
from typing import Optional, Dict, Any, Callable
import time

class RobustErrorHandler:
    """Gestionnaire d'erreurs robuste pour l'application."""

    def __init__(self):
        self.error_counts = {}
        self.max_retries = 3
        self.recovery_actions = {}

        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('app_errors.log', mode='a')
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Actions de récupération prédéfinies
        self._setup_recovery_actions()

    def _setup_recovery_actions(self):
        """Configurer les actions de récupération automatique."""
        self.recovery_actions = {
            'FileNotFoundError': self._recover_file_error,
            'ConnectionError': self._recover_connection_error,
            'ValueError': self._recover_value_error,
            'RuntimeError': self._recover_runtime_error,
            'torch.cuda.OutOfMemoryError': self._recover_cuda_oom
        }

    def handle_error(self, error: Exception, context: str = "", retry_func: Callable = None, **kwargs) -> Optional[Any]:
        """
        Gestion centralisée des erreurs avec récupération automatique.

        Args:
            error: L'exception capturée
            context: Contexte de l'erreur pour les logs
            retry_func: Fonction à réessayer en cas d'échec
            **kwargs: Arguments supplémentaires pour la récupération

        Returns:
            Résultat de la fonction de retry ou valeur de fallback
        """
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        self.logger.error(f"Erreur dans {context}: {error_type}: {str(error)}")

        # Tentative de récupération automatique
        if error_type in self.recovery_actions:
            try:
                recovery_result = self.recovery_actions[error_type](error, context, retry_func, **kwargs)
                if recovery_result is not None:
                    self.logger.info(f"Récupération automatique réussie pour {error_type} dans {context}")
                    return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"Échec de la récupération automatique: {recovery_error}")

        # Gestion spécifique selon le type d'erreur
        if isinstance(error, FileNotFoundError):
            return self._handle_file_error(error, context)

        elif isinstance(error, ConnectionError):
            return self._handle_connection_error(error, context, retry_func)

        elif isinstance(error, ValueError):
            return self._handle_value_error(error, context)

        elif isinstance(error, RuntimeError):
            return self._handle_runtime_error(error, context)

        else:
            self.logger.error(f"Erreur non gérée: {error_type}")
            return self._fallback_response(error, context)

    def _recover_file_error(self, error, context, retry_func, **kwargs):
        """Récupération pour erreurs de fichiers."""
        filename = getattr(error, 'filename', 'unknown')
        self.logger.warning(f"Tentative de récupération pour fichier manquant: {filename}")

        # Essayer de créer des répertoires manquants
        if filename:
            try:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                self.logger.info(f"Répertoire créé: {os.path.dirname(filename)}")
                if retry_func:
                    return retry_func()
            except Exception as e:
                self.logger.error(f"Impossible de créer le répertoire: {e}")

        return None

    def _recover_connection_error(self, error, context, retry_func, **kwargs):
        """Récupération pour erreurs de connexion."""
        if retry_func and self.error_counts.get('ConnectionError', 0) <= self.max_retries:
            self.logger.info(f"Tentative de reconnexion ({self.error_counts['ConnectionError']})...")
            time.sleep(2 ** self.error_counts['ConnectionError'])  # Backoff exponentiel
            try:
                return retry_func()
            except Exception as retry_error:
                self.logger.error(f"Échec du retry: {retry_error}")
        return None

    def _recover_value_error(self, error, context, retry_func, **kwargs):
        """Récupération pour erreurs de validation."""
        # Pour les erreurs de validation, essayer avec des paramètres par défaut
        if 'default_value' in kwargs:
            self.logger.warning(f"Utilisation de la valeur par défaut pour {context}")
            return kwargs['default_value']
        return None

    def _recover_runtime_error(self, error, context, retry_func, **kwargs):
        """Récupération pour erreurs runtime."""
        # Réduire la taille du batch en cas d'erreur mémoire
        if 'batch_size' in kwargs:
            new_batch_size = max(1, kwargs['batch_size'] // 2)
            self.logger.warning(f"Réduction du batch size à {new_batch_size}")
            kwargs['batch_size'] = new_batch_size
            if retry_func:
                try:
                    return retry_func(**kwargs)
                except Exception as e:
                    self.logger.error(f"Échec avec batch size réduit: {e}")
        return None

    def _recover_cuda_oom(self, error, context, retry_func, **kwargs):
        """Récupération spécifique pour CUDA out of memory."""
        self.logger.warning("CUDA OOM détecté - tentative de récupération")

        # Nettoyer le cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("Cache CUDA nettoyé")

        # Réduire drastiquement la taille du batch
        if 'batch_size' in kwargs:
            kwargs['batch_size'] = max(1, kwargs['batch_size'] // 4)

        if retry_func:
            try:
                return retry_func(**kwargs)
            except Exception as e:
                self.logger.error(f"Échec de récupération CUDA OOM: {e}")

        return None

    def _handle_file_error(self, error: FileNotFoundError, context: str) -> None:
        """Gestion des erreurs de fichiers."""
        print(f"❌ Fichier manquant dans {context}: {error.filename}")
        print("💡 Vérifiez que tous les fichiers requis sont présents")
        return None

    def _handle_connection_error(self, error: ConnectionError, context: str, retry_func) -> Optional[Any]:
        """Gestion des erreurs de connexion avec retry."""
        if retry_func and self.error_counts.get('ConnectionError', 0) <= self.max_retries:
            print(f"🔄 Tentative de reconnexion ({self.error_counts['ConnectionError']})...")
            time.sleep(2)
            try:
                return retry_func()
            except Exception as retry_error:
                self.logger.error(f"Échec du retry: {retry_error}")
        return None

    def _handle_value_error(self, error: ValueError, context: str) -> None:
        """Gestion des erreurs de validation."""
        print(f"⚠️ Données invalides dans {context}: {str(error)}")
        print("💡 Vérifiez le format des données d'entrée")
        return None

    def _handle_runtime_error(self, error: RuntimeError, context: str) -> None:
        """Gestion des erreurs runtime."""
        print(f"🚨 Erreur système dans {context}: {str(error)}")
        print("💡 Vérifiez la disponibilité des ressources système")
        return None

    def _fallback_response(self, error, context):
        """Réponse de fallback pour erreurs non gérées."""
        print(f"⚠️ Erreur non critique dans {context}: {type(error).__name__}")
        return None

    def validate_image_input(self, image, context="validation") -> bool:
        """Validation robuste des images d'entrée."""
        try:
            if image is None:
                raise ValueError("Image est None")

            # Vérification du type
            if not hasattr(image, 'shape') and not hasattr(image, 'size'):
                raise ValueError("Format d'image non reconnu")

            # Vérification des dimensions minimales
            if hasattr(image, 'shape'):
                height, width = image.shape[:2]
            else:
                width, height = image.size

            if width < 32 or height < 32:
                raise ValueError(f"Image trop petite: {width}x{height}")

            if width > 4096 or height > 4096:
                raise ValueError(f"Image trop grande: {width}x{height}")

            return True

        except Exception as e:
            self.logger.error(f"Validation d'image échouée dans {context}: {e}")
            return False

    def safe_execute(self, func: Callable, *args, context="", **kwargs) -> Optional[Any]:
        """
        Exécution sécurisée d'une fonction avec gestion d'erreur.

        Args:
            func: Fonction à exécuter
            context: Contexte pour les logs
            *args, **kwargs: Arguments de la fonction

        Returns:
            Résultat ou None en cas d'erreur
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self.handle_error(e, context, lambda: func(*args, **kwargs), **kwargs)

    def get_error_summary(self) -> Dict[str, int]:
        """Résumé des erreurs rencontrées."""
        return self.error_counts.copy()

    def reset_error_counts(self):
        """Réinitialiser les compteurs d'erreurs."""
        self.error_counts.clear()

# Instance globale du gestionnaire d'erreurs
error_handler = RobustErrorHandler()

def safe_execute(func, *args, context="", **kwargs):
    """
    Fonction utilitaire pour exécution sécurisée.

    Args:
        func: Fonction à exécuter
        context: Contexte pour les logs
        *args, **kwargs: Arguments de la fonction

    Returns:
        Résultat ou None en cas d'erreur
    """
    return error_handler.safe_execute(func, *args, context=context, **kwargs)