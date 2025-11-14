import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from models import Document, DocumentVector, DocumentSummary, EntityExtraction
from services.file_manager import FileManager
from utils.api_responses import APIResponse

logger = logging.getLogger(__name__)

class DocumentService:
    """Service de gestion des documents"""

    def __init__(self, file_manager: FileManager, advanced_processor,
                 language_detector, vector_store, advanced_ai):
        self.file_manager = file_manager
        self.advanced_processor = advanced_processor
        self.language_detector = language_detector
        self.vector_store = vector_store
        self.advanced_ai = advanced_ai

    def upload_document(self, file, user_id: int) -> tuple:
        """Traiter l'upload d'un document"""
        try:
            # Sauvegarder le fichier
            file_result = self.file_manager.save_file(file, user_id)
            if not file_result['success']:
                return APIResponse.error(file_result['error'], 400)

            file_path = file_result['file_path']

            try:
                # Traitement du fichier
                result = self.advanced_processor.process_file(file_path, extract_images=True)
                text_content = result['text_content']

                if not text_content.strip():
                    self.file_manager.cleanup_file(file_path)
                    return APIResponse.error('Aucun contenu textuel extrait du fichier', 400)

                # Détection de la langue
                detected_language = self.language_detector.detect_language(text_content)

                # Création du document
                document = Document(
                    filename=file_result['filename'],
                    original_filename=file_result['original_filename'],
                    file_type=file_result['file_type'],
                    file_size=file_result['file_size'],
                    content_text=text_content,
                    detected_language=detected_language,
                    word_count=result.get('word_count', len(text_content.split())),
                    user_id=user_id
                )

                from extensions import db
                db.session.add(document)
                db.session.flush()  # Obtenir l'ID

                # Création du vecteur
                vector_data = self.vector_store._create_vector(text_content)
                doc_vector = DocumentVector(
                    document_id=document.id,
                    vector_method='tfidf'
                )
                doc_vector.set_vector(vector_data)
                db.session.add(doc_vector)

                # Génération du résumé automatique
                if self.advanced_ai.is_available():
                    summary_result = self.advanced_ai.generate_summary(text_content, detected_language)
                    if 'summary' in summary_result:
                        summary = DocumentSummary(
                            document_id=document.id,
                            summary_type='auto',
                            content=summary_result['summary'],
                            language=detected_language,
                            confidence_score=summary_result.get('confidence_score', 0.0)
                        )
                        db.session.add(summary)

                # Extraction des entités
                if self.advanced_ai.is_available():
                    entities_result = self.advanced_ai.extract_entities(text_content, detected_language)
                    if 'entities' in entities_result:
                        for entity in entities_result['entities']:
                            entity_record = EntityExtraction(
                                document_id=document.id,
                                entity_type=entity.get('type', 'unknown'),
                                entity_value=entity.get('value', ''),
                                confidence_score=entity.get('confidence', 0.0),
                                context=entity.get('context', '')
                            )
                            db.session.add(entity_record)

                db.session.commit()

                # Nettoyage du fichier temporaire
                self.file_manager.cleanup_file(file_path)

                return APIResponse.success(
                    data={
                        'document': document.to_dict(),
                        'metadata': result.get('metadata', {}),
                        'tables_found': len(result.get('tables', [])),
                        'images_found': len(result.get('images', []))
                    },
                    message=f'Fichier "{file_result["original_filename"]}" traité avec succès'
                )

            except Exception as e:
                from extensions import db
                db.session.rollback()
                self.file_manager.cleanup_file(file_path)
                logger.error(f"Erreur lors du traitement du fichier: {str(e)}")
                return APIResponse.server_error(f'Erreur lors du traitement: {str(e)}')

        except Exception as e:
            logger.error(f"Erreur d'upload: {str(e)}")
            return APIResponse.server_error(f'Échec du téléchargement: {str(e)}')

    def list_documents(self, user_id: int, workspace_id: Optional[int] = None) -> tuple:
        """Lister les documents d'un utilisateur"""
        try:
            query = Document.query.filter_by(user_id=user_id)

            if workspace_id:
                query = query.filter_by(workspace_id=workspace_id)

            documents = query.order_by(Document.created_at.desc()).all()

            return APIResponse.success({
                'documents': [doc.to_dict() for doc in documents],
                'total': len(documents)
            })

        except Exception as e:
            logger.error(f"Erreur lors du listage des documents: {str(e)}")
            return APIResponse.server_error('Échec du listage des documents')

    def delete_document(self, doc_id: int, user_id: int) -> tuple:
        """Supprimer un document"""
        try:
            from extensions import db
            from auth_utils import check_document_permission

            document = Document.query.get_or_404(doc_id)

            if not check_document_permission(document, 'admin'):
                return APIResponse.forbidden('Permission refusée')

            from extensions import db
            db.session.delete(document)
            db.session.commit()

            return APIResponse.success(message='Document supprimé avec succès')

        except Exception as e:
            from extensions import db
            db.session.rollback()
            logger.error(f"Erreur de suppression: {str(e)}")
            return APIResponse.server_error('Erreur de suppression')
