import os
import uuid
import tempfile
from werkzeug.utils import secure_filename
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FileManager:
    """Service de gestion des fichiers"""

    def __init__(self, upload_folder: str, allowed_extensions: set, max_size: int):
        self.upload_folder = upload_folder
        self.allowed_extensions = allowed_extensions
        self.max_size = max_size
        self._ensure_upload_folder()

    def _ensure_upload_folder(self):
        """Créer le dossier d'upload s'il n'existe pas"""
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder, exist_ok=True)

    def is_allowed_file(self, filename: str) -> bool:
        """Vérifier si le type de fichier est autorisé"""
        return ('.' in filename and
                filename.rsplit('.', 1)[1].lower() in self.allowed_extensions)

    def save_file(self, file, user_id: int) -> Dict[str, Any]:
        """Sauvegarder un fichier de manière sécurisée"""
        try:
            if not self.is_allowed_file(file.filename):
                return {
                    'success': False,
                    'error': 'Type de fichier non autorisé'
                }

            # Générer un nom de fichier unique
            original_filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            file_path = os.path.join(self.upload_folder, unique_filename)

            # Sauvegarder le fichier
            file.save(file_path)

            # Vérifier la taille
            file_size = os.path.getsize(file_path)
            if file_size > self.max_size:
                os.remove(file_path)
                return {
                    'success': False,
                    'error': f'Fichier trop volumineux. Taille maximale: {self.max_size} bytes'
                }

            return {
                'success': True,
                'filename': unique_filename,
                'original_filename': original_filename,
                'file_path': file_path,
                'file_size': file_size,
                'file_type': os.path.splitext(original_filename)[1].lower()
            }

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du fichier: {str(e)}")
            return {
                'success': False,
                'error': f'Erreur lors de la sauvegarde: {str(e)}'
            }

    def cleanup_file(self, file_path: str):
        """Nettoyer un fichier temporaire"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du fichier {file_path}: {str(e)}")

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Nettoyer les anciens fichiers temporaires"""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            for filename in os.listdir(self.upload_folder):
                file_path = os.path.join(self.upload_folder, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        logger.info(f"Fichier temporaire supprimé: {filename}")
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des anciens fichiers: {str(e)}")
