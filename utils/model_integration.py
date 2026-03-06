"""
Intégration du modèle Phi-3-mini pour Agro-Scan
"""

import os
import logging
from pathlib import Path
import subprocess
import time
import requests

logger = logging.getLogger(__name__)

class ModelManager:
    """Gestionnaire du modèle local Phi-3-mini"""
    
    def __init__(self):
        self.model_path = Path("local_model/Phi-3-mini-4k-instruct-q4.gguf")
        self.server_script = Path("local_model/model_server.py")
        self.server_url = "http://127.0.0.1:5000"
        self.server_process = None
        self.is_running = False
    
    def check_model_exists(self):
        """Vérifie si le modèle existe"""
        return self.model_path.exists()
    
    def start_server(self):
        """Démarre le serveur du modèle"""
        if not self.check_model_exists():
            logger.warning(f"Modèle non trouvé: {self.model_path}")
            return False
        
        if self.is_server_running():
            logger.info("Le serveur du modèle est déjà en cours d'exécution")
            return True
        
        try:
            # Démarrer le serveur en arrière-plan
            self.server_process = subprocess.Popen(
                ["python", str(self.server_script)],
                cwd=str(self.server_script.parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Attendre que le serveur démarre
            for _ in range(10):
                time.sleep(1)
                if self.is_server_running():
                    self.is_running = True
                    logger.info("Serveur du modèle démarré avec succès")
                    return True
            
            logger.error("Le serveur n'a pas démarré dans les temps")
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage du serveur: {str(e)}")
            return False
    
    def stop_server(self):
        """Arrête le serveur du modèle"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.is_running = False
                logger.info("Serveur du modèle arrêté")
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt du serveur: {str(e)}")
    
    def is_server_running(self):
        """Vérifie si le serveur est en cours d'exécution"""
        try:
            response = requests.get(f"{self.server_url}/docs", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self):
        """Retourne les informations sur le modèle"""
        info = {
            "model_path": str(self.model_path),
            "model_exists": self.check_model_exists(),
            "server_running": self.is_server_running(),
            "server_url": self.server_url
        }
        
        if self.model_path.exists():
            info["model_size_mb"] = round(self.model_path.stat().st_size / (1024 * 1024), 2)
        
        return info

# Instance globale
model_manager = ModelManager()

