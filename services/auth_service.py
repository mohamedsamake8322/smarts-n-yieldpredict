from flask_login import login_user, logout_user
from flask import session, request
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

from models import User
from utils.validators import validate_email, validate_password_strength
from utils.api_responses import APIResponse

logger = logging.getLogger(__name__)

class AuthService:
    """Service de gestion de l'authentification"""

    @staticmethod
    def register_user(data: Dict[str, Any]) -> tuple:
        """Inscrire un nouvel utilisateur"""
        try:
            # Validation des champs requis
            required_fields = ['username', 'email', 'password']
            missing = [field for field in required_fields if not data.get(field)]
            if missing:
                return APIResponse.validation_error(
                    {field: f"{field} est requis" for field in missing}
                )

            # Extraction des données
            username = data.get('username', '').strip()
            email = data.get('email', '').strip()
            password = data.get('password', '')
            first_name = data.get('first_name', '').strip()
            last_name = data.get('last_name', '').strip()
            language = data.get('language', 'fr')

            # Validation de l'email
            if not validate_email(email):
                return APIResponse.validation_error(
                    {'email': 'Format d\'email invalide'}
                )

            # Validation du mot de passe
            password_validation = validate_password_strength(password)
            if not password_validation['is_valid']:
                return APIResponse.validation_error(
                    {'password': password_validation['errors']}
                )

            # Vérification de l'unicité
            if User.query.filter_by(username=username).first():
                return APIResponse.validation_error(
                    {'username': 'Ce nom d\'utilisateur existe déjà'}
                )

            if User.query.filter_by(email=email).first():
                return APIResponse.validation_error(
                    {'email': 'Cette adresse email est déjà utilisée'}
                )

            # Création de l'utilisateur
            user = User(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name,
                preferred_language=language
            )
            user.set_password(password)

            from extensions import db
            db.session.add(user)
            db.session.commit()

            # Connexion automatique
            login_user(user)

            return APIResponse.success(
                data=user.to_dict(),
                message='Compte créé avec succès'
            )

        except Exception as e:
            from extensions import db
            db.session.rollback()
            logger.error(f"Erreur lors de l'inscription: {str(e)}")
            return APIResponse.server_error(f'Erreur lors de la création du compte: {str(e)}')

    @staticmethod
    def login_user(data: Dict[str, Any]) -> tuple:
        """Connecter un utilisateur"""
        try:
            username_or_email = data.get('username', '').strip()
            password = data.get('password', '')

            if not username_or_email or not password:
                return APIResponse.validation_error(
                    {'username': 'Nom d\'utilisateur ou email requis',
                     'password': 'Mot de passe requis'}
                )

            # Recherche de l'utilisateur
            user = User.query.filter(
                (User.username == username_or_email) | (User.email == username_or_email)
            ).first()

            if not user or not user.check_password(password):
                return APIResponse.unauthorized('Identifiants incorrects')

            if not user.active:
                return APIResponse.forbidden('Compte désactivé')

            # Mise à jour de la dernière connexion
            user.last_login = datetime.now(timezone.utc)
            from extensions import db
            db.session.commit()

            # Connexion
            login_user(user, remember=True)

            return APIResponse.success(
                data=user.to_dict(),
                message='Connexion réussie'
            )

        except Exception as e:
            logger.error(f"Erreur lors de la connexion: {str(e)}")
            return APIResponse.server_error(f'Erreur de connexion: {str(e)}')

    @staticmethod
    def logout_user() -> tuple:
        """Déconnecter l'utilisateur"""
        try:
            logout_user()
            return APIResponse.success(message='Déconnexion réussie')
        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion: {str(e)}")
            return APIResponse.server_error('Erreur lors de la déconnexion')
