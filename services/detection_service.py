"""
Service de détection des plantes et maladies par IA
Version avancée avec support multi-modèles, Top-K, explications, audit, etc.
"""

# Appliquer le workaround PyTorch/Streamlit en premier
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.pytorch_fix import apply_pytorch_fix
    apply_pytorch_fix()
except ImportError:
    pass

import numpy as np
from PIL import Image
import io
import logging
from typing import Optional, List, Dict, Tuple
import os
import json
import asyncio
from datetime import datetime
import hashlib
import time

from models.detection_models import (
    DetectionResult, PlantInfo, DiseaseInfo, Recommendation,
    TopPrediction, SymptomExplanation, DetectionAudit
)

logger = logging.getLogger(__name__)

class DetectionService:
    """Service de détection intelligente des plantes et maladies - Version Avancée"""
    
    def __init__(self, vision_model=None):
        """
        Initialise le service de détection
        
        Args:
            vision_model: Instance de VisionModel - si None, sera chargé automatiquement
        """
        self.model_loaded = False
        self.model_path = os.getenv("MODEL_PATH", "models/plant_detection_model")
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
        self.top_k = int(os.getenv("TOP_K", "3"))  # Top-K prédictions
        
        # Modèle de vision (direct)
        self.vision_model_instance = vision_model
        
        # Multi-modèles
        self.models = {
            'vision_cnn': None,
            'vision_vit': None,
            'deficiency': None,
            'water_stress': None
        }
        
        # Dossier pour les logs d'audit
        self.audit_dir = Path("data/audit_logs")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger les modèles
        self._load_models()
    
    def _load_models(self):
        """Charge tous les modèles disponibles"""
        # Si modèle de vision fourni directement, l'utiliser
        if self.vision_model_instance and self.vision_model_instance.is_ready():
            self.models['vision_cnn'] = self.vision_model_instance
            logger.info("✅ Modèle de vision chargé (instance fournie)")
            self.model_loaded = True
            return
        
        # Essayer d'abord VisionModel (cherche dans local_model/vision/)
        try:
            from local_model.vision.vision_model import VisionModel
            vision_model = VisionModel()
            if vision_model.is_ready() and vision_model.model is not None:
                # Déterminer si c'est ViT ou CNN
                if hasattr(vision_model, 'arch'):
                    arch = vision_model.arch.lower()
                    if 'vit' in arch or 'transformer' in arch:
                        self.models['vision_vit'] = vision_model
                    else:
                        self.models['vision_cnn'] = vision_model
                else:
                    self.models['vision_cnn'] = vision_model
                logger.info("✅ Modèle de vision chargé depuis local_model/vision/")
                self.model_loaded = True
                return
        except Exception as e:
            logger.debug(f"VisionModel non disponible: {str(e)}")
        
        # Sinon, essayer de charger depuis VisionModelService (ancien système)
        try:
            from services.vision_model_service import VisionModelService
            self.vision_model = VisionModelService()
            if self.vision_model.is_ready():
                model_type = self.vision_model.model_type
                if 'vit' in str(model_type).lower() or 'transformer' in str(model_type).lower():
                    self.models['vision_vit'] = self.vision_model
                else:
                    self.models['vision_cnn'] = self.vision_model
                logger.info("✅ Modèle de vision chargé depuis VisionModelService")
                self.model_loaded = True
            else:
                logger.warning("⚠️ Modèle de vision non disponible")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du chargement du modèle de vision: {str(e)}")
        
        # Modèle de carences nutritionnelles
        try:
            self.models['deficiency'] = self._load_deficiency_model()
        except Exception as e:
            logger.warning(f"⚠️ Modèle de carences non disponible: {str(e)}")
        
        # Modèle de stress hydrique
        try:
            self.models['water_stress'] = self._load_water_stress_model()
        except Exception as e:
            logger.warning(f"⚠️ Modèle de stress hydrique non disponible: {str(e)}")
        
        if not self.model_loaded:
            logger.info("Mode simulation activé")
            self.model_loaded = True
    
    def _load_deficiency_model(self):
        """Charge le modèle de détection de carences nutritionnelles"""
        # TODO: Charger le vrai modèle de carences
        # Pour l'instant, retourne None (simulation)
        return None
    
    def _load_water_stress_model(self):
        """Charge le modèle de détection de stress hydrique"""
        # TODO: Charger le vrai modèle de stress hydrique
        # Pour l'instant, retourne None (simulation)
        return None
    
    def is_ready(self) -> bool:
        """Vérifie si le service est prêt"""
        return self.model_loaded
    
    async def detect(
        self,
        image_data: bytes,
        filename: str,
        text_description: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> DetectionResult:
        """
        Détecte la plante et les maladies dans l'image (avec support texte optionnel)
        
        Args:
            image_data: Données binaires de l'image
            filename: Nom du fichier
            text_description: Description textuelle optionnelle de l'utilisateur
            user_id: ID de l'utilisateur pour l'audit
        
        Returns:
            DetectionResult: Résultat de la détection enrichi
        """
        start_time = time.time()
        detection_id = self._generate_detection_id(image_data)
        
        try:
            # Charger l'image - gérer différents formats
            image_bytes = None
            
            if isinstance(image_data, bytes):
                # Bytes bruts - utiliser directement
                image_bytes = image_data
            elif hasattr(image_data, 'getvalue'):
                # Objet BytesIO ou UploadedFile de Streamlit
                image_bytes = image_data.getvalue()
            elif hasattr(image_data, 'read'):
                # Objet file-like
                if hasattr(image_data, 'seek'):
                    image_data.seek(0)
                image_bytes = image_data.read()
            else:
                # Essayer de convertir en bytes
                image_bytes = bytes(image_data)
            
            # Vérifier que nous avons des bytes valides
            if not image_bytes or len(image_bytes) == 0:
                raise ValueError("Les données d'image sont vides")
            
            # Ouvrir l'image depuis les bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            
            # Prédictions multi-modèles
            all_predictions = await self._predict_multi_models(image_data, image_array)
            
            # Fusionner les résultats des différents modèles
            merged_predictions = self._merge_model_predictions(all_predictions)
            
            # Si description textuelle fournie, combiner avec l'analyse image
            if text_description:
                merged_predictions = await self._combine_image_text(
                    merged_predictions, text_description, image_data
                )
            
            # Top-K prédictions
            top_k_diseases = self._get_top_k_diseases(merged_predictions, k=self.top_k)
            
            # Générer les explications pédagogiques
            symptom_explanations = self._generate_symptom_explanations(
                merged_predictions, top_k_diseases
            )
            
            # Construire le résultat
            result = self._build_result(
                merged_predictions,
                image_array.shape,
                top_k_diseases,
                symptom_explanations,
                text_description
            )
            
            # Calculer le temps de traitement
            processing_time = (time.time() - start_time) * 1000  # en ms
            
            # Créer l'audit
            result.audit = self._create_audit(
                detection_id=detection_id,
                user_id=user_id,
                image_data=image_data,
                processing_time_ms=processing_time,
                predictions=merged_predictions,
                models_used=list(self.models.keys())
            )
            
            # Sauvegarder l'audit
            self._save_audit(result.audit)
            
            logger.info(f"✅ Détection terminée: {result.plant_info.name} (ID: {detection_id})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la détection: {str(e)}")
            raise
    
    async def _predict_multi_models(
        self,
        image_data: bytes,
        image_array: np.ndarray
    ) -> Dict[str, dict]:
        """
        Prédictions avec tous les modèles disponibles
        
        Returns:
            Dict avec les prédictions de chaque modèle
        """
        predictions = {}
        
        # S'assurer que image_data est bien des bytes pour les modèles
        if isinstance(image_data, bytes):
            image_bytes_for_model = image_data
        elif hasattr(image_data, 'getvalue'):
            image_bytes_for_model = image_data.getvalue()
        elif hasattr(image_data, 'read'):
            if hasattr(image_data, 'seek'):
                image_data.seek(0)
            image_bytes_for_model = image_data.read()
        else:
            image_bytes_for_model = bytes(image_data)
        
        # Modèle de vision principal (CNN ou ViT)
        if self.models['vision_cnn']:
            try:
                pred = self.models['vision_cnn'].predict(image_bytes_for_model)
                predictions['vision_cnn'] = pred
            except Exception as e:
                logger.warning(f"Erreur modèle CNN: {e}")
        
        if self.models['vision_vit']:
            try:
                pred = self.models['vision_vit'].predict(image_bytes_for_model)
                predictions['vision_vit'] = pred
            except Exception as e:
                logger.warning(f"Erreur modèle ViT: {e}")
        
        # Modèle de carences
        if self.models['deficiency']:
            try:
                pred = await self._predict_deficiencies(image_array)
                predictions['deficiency'] = pred
            except Exception as e:
                logger.warning(f"Erreur modèle carences: {e}")
        else:
            # Simulation
            predictions['deficiency'] = await self._simulate_deficiencies(image_array)
        
        # Modèle de stress hydrique
        if self.models['water_stress']:
            try:
                pred = await self._predict_water_stress(image_array)
                predictions['water_stress'] = pred
            except Exception as e:
                logger.warning(f"Erreur modèle stress: {e}")
        else:
            # Simulation
            predictions['water_stress'] = await self._simulate_water_stress(image_array)
        
        # Fallback si aucun modèle de vision
        if 'vision_cnn' not in predictions and 'vision_vit' not in predictions:
            predictions['fallback'] = await self._predict_fallback(image_array)
        
        return predictions
    
    def _merge_model_predictions(self, all_predictions: Dict[str, dict]) -> dict:
        """
        Fusionne les prédictions de tous les modèles
        
        Args:
            all_predictions: Dict avec prédictions de chaque modèle
        
        Returns:
            dict: Prédictions fusionnées
        """
        merged = {
            "plant": {"name": "Unknown", "scientific_name": "", "confidence": 0.0},
            "diseases": [],
            "deficiencies": [],
            "stress": {"water": 0.0, "temperature": 0.0}
        }
        
        # Fusionner les prédictions de vision
        vision_preds = []
        for key in ['vision_cnn', 'vision_vit', 'fallback']:
            if key in all_predictions:
                vision_preds.append(all_predictions[key])
        
        if vision_preds:
            # Prendre la meilleure prédiction de plante
            best_plant = max(
                [p.get("plant", {}) for p in vision_preds],
                key=lambda x: x.get("confidence", 0)
            )
            merged["plant"] = best_plant
            
            # Fusionner les maladies (dédupliquer et moyenner les confidences)
            disease_dict = {}
            for pred in vision_preds:
                for disease in pred.get("diseases", []):
                    name = disease["name"]
                    if name not in disease_dict:
                        disease_dict[name] = {
                            "name": name,
                            "confidence": [],
                            "severity": disease.get("severity", "moderate"),
                            "description": disease.get("description", "")
                        }
                    disease_dict[name]["confidence"].append(disease["confidence"])
            
            # Moyenner les confidences
            for name, data in disease_dict.items():
                avg_conf = np.mean(data["confidence"])
                merged["diseases"].append({
                    "name": name,
                    "confidence": float(avg_conf),
                    "severity": data["severity"],
                    "description": data["description"]
                })
            
            # Trier par confidence décroissante
            merged["diseases"].sort(key=lambda x: x["confidence"], reverse=True)
        
        # Fusionner les carences
        if 'deficiency' in all_predictions:
            merged["deficiencies"] = all_predictions['deficiency'].get("deficiencies", [])
        
        # Fusionner le stress
        if 'water_stress' in all_predictions:
            merged["stress"] = all_predictions['water_stress'].get("stress", merged["stress"])
        
        return merged
    
    def _get_top_k_diseases(self, predictions: dict, k: int = 3) -> List[TopPrediction]:
        """
        Récupère les Top-K meilleures prédictions de maladies
        
        Args:
            predictions: Prédictions fusionnées
            k: Nombre de prédictions à retourner
        
        Returns:
            List[TopPrediction]: Top-K prédictions
        """
        diseases = predictions.get("diseases", [])
        
        # Trier par confidence et prendre les k meilleures
        sorted_diseases = sorted(diseases, key=lambda x: x["confidence"], reverse=True)
        top_k = sorted_diseases[:k]
        
        return [
            TopPrediction(
                name=d["name"],
                confidence=d["confidence"],
                severity=d.get("severity", "moderate"),
                description=d.get("description", "")
            )
            for d in top_k
        ]
    
    def _generate_symptom_explanations(
        self,
        predictions: dict,
        top_k_diseases: List[TopPrediction]
    ) -> List[SymptomExplanation]:
        """
        Génère des explications pédagogiques des symptômes détectés
        
        Args:
            predictions: Prédictions fusionnées
            top_k_diseases: Top-K maladies
        
        Returns:
            List[SymptomExplanation]: Explications pédagogiques
        """
        explanations = []
        
        for disease in top_k_diseases:
            symptom = disease.name
            explanation = self._get_disease_explanation(symptom)
            visual_desc = self._get_visual_description(symptom)
            causes = self._get_common_causes(symptom)
            
            explanations.append(SymptomExplanation(
                symptom=symptom,
                explanation=explanation,
                visual_description=visual_desc,
                common_causes=causes
            ))
        
        # Ajouter des explications pour les carences
        for deficiency in predictions.get("deficiencies", []):
            def_type = deficiency.get("type", "")
            explanations.append(SymptomExplanation(
                symptom=f"Carence en {def_type}",
                explanation=self._get_deficiency_explanation(def_type),
                visual_description=self._get_deficiency_visual(def_type),
                common_causes=self._get_deficiency_causes(def_type)
            ))
        
        return explanations
    
    def _get_disease_explanation(self, disease_name: str) -> str:
        """Retourne une explication pédagogique d'une maladie"""
        explanations = {
            "Mildiou": (
                "Le mildiou est une maladie fongique qui se développe dans des conditions "
                "humides. Les spores du champignon se propagent par l'eau et l'air, "
                "infectant les feuilles, tiges et fruits. Les symptômes apparaissent "
                "généralement après des périodes de pluie ou d'humidité élevée."
            ),
            "Oïdium": (
                "L'oïdium est un champignon qui forme un feutrage blanc poudreux sur les "
                "feuilles. Il se développe par temps sec et chaud, contrairement au mildiou. "
                "Il bloque la photosynthèse et affaiblit la plante progressivement."
            ),
            "Anthracnose": (
                "L'anthracnose est causée par des champignons qui créent des taches nécrotiques "
                "sur les feuilles, tiges et fruits. Elle se propage par les éclaboussures d'eau "
                "et peut causer la chute prématurée des fruits."
            )
        }
        return explanations.get(disease_name, f"La maladie '{disease_name}' affecte la santé de la plante.")
    
    def _get_visual_description(self, disease_name: str) -> str:
        """Retourne une description visuelle des symptômes"""
        descriptions = {
            "Mildiou": (
                "Taches brunes ou noires irrégulières sur les feuilles, souvent avec un "
                "duvet blanc ou grisâtre sur la face inférieure. Les feuilles peuvent "
                "se dessécher et tomber."
            ),
            "Oïdium": (
                "Feutrage blanc poudreux qui recouvre les feuilles, d'abord en petites "
                "taches puis s'étendant. Les feuilles peuvent se déformer et jaunir."
            )
        }
        return descriptions.get(disease_name, "Symptômes visuels caractéristiques de cette maladie.")
    
    def _get_common_causes(self, disease_name: str) -> List[str]:
        """Retourne les causes communes d'une maladie"""
        causes = {
            "Mildiou": [
                "Humidité élevée (>80%)",
                "Températures modérées (15-25°C)",
                "Manque de circulation d'air",
                "Feuillage dense"
            ],
            "Oïdium": [
                "Températures chaudes (20-30°C)",
                "Humidité modérée (40-70%)",
                "Variations importantes de température jour/nuit",
                "Stress hydrique"
            ]
        }
        return causes.get(disease_name, ["Conditions environnementales favorables"])
    
    def _get_deficiency_explanation(self, deficiency_type: str) -> str:
        """Explication d'une carence nutritionnelle"""
        explanations = {
            "Azote": "La carence en azote se manifeste par un jaunissement des feuilles, commençant par les plus anciennes.",
            "Phosphore": "Le manque de phosphore cause un retard de croissance et des feuilles violacées.",
            "Potassium": "La carence en potassium provoque des bords de feuilles brûlés et une faible résistance aux maladies."
        }
        return explanations.get(deficiency_type, f"Carence en {deficiency_type} détectée.")
    
    def _get_deficiency_visual(self, deficiency_type: str) -> str:
        """Description visuelle d'une carence"""
        visuals = {
            "Azote": "Feuilles jaunissantes, croissance ralentie, tiges fines.",
            "Phosphore": "Feuilles vert foncé ou violacées, croissance réduite.",
            "Potassium": "Bords de feuilles brûlés, taches nécrotiques."
        }
        return visuals.get(deficiency_type, "Symptômes visuels de carence.")
    
    def _get_deficiency_causes(self, deficiency_type: str) -> List[str]:
        """Causes d'une carence"""
        return [
            "Sol pauvre en nutriments",
            "pH du sol inadapté",
            "Drainage excessif",
            "Manque de fertilisation"
        ]
    
    async def _combine_image_text(
        self,
        predictions: dict,
        text_description: str,
        image_data: bytes
    ) -> dict:
        """
        Combine l'analyse d'image avec la description textuelle
        
        Args:
            predictions: Prédictions de l'image
            text_description: Description textuelle de l'utilisateur
            image_data: Données de l'image
        
        Returns:
            dict: Prédictions enrichies
        """
        # Analyser le texte pour extraire des indices
        text_keywords = self._extract_keywords_from_text(text_description)
        
        # Enrichir les prédictions avec les indices textuels
        enriched = predictions.copy()
        
        # Si le texte mentionne des symptômes spécifiques, ajuster les confidences
        for disease in enriched.get("diseases", []):
            disease_name_lower = disease["name"].lower()
            if any(kw in text_description.lower() for kw in text_keywords.get("symptoms", [])):
                # Augmenter légèrement la confidence si le texte confirme
                disease["confidence"] = min(1.0, disease["confidence"] * 1.1)
        
        # Ajouter une analyse combinée
        enriched["combined_analysis"] = (
            f"Analyse combinée: L'image montre {predictions['plant']['name']}. "
            f"La description textuelle mentionne: {text_description[:100]}..."
        )
        
        return enriched
    
    def _extract_keywords_from_text(self, text: str) -> dict:
        """Extrait des mots-clés pertinents du texte"""
        keywords = {
            "symptoms": ["tache", "jaune", "brun", "noir", "flétri", "tombé", "déformé"],
            "severity": ["grave", "sévère", "modéré", "léger", "début"],
            "location": ["feuille", "tige", "fruit", "racine", "fleur"]
        }
        
        found = {k: [] for k in keywords.keys()}
        text_lower = text.lower()
        
        for category, words in keywords.items():
            for word in words:
                if word in text_lower:
                    found[category].append(word)
        
        return found
    
    def _generate_recommendations(self, predictions: dict) -> List[Recommendation]:
        """
        Génère des recommandations dynamiques adaptées
        
        Args:
            predictions: Prédictions fusionnées
        
        Returns:
            List[Recommendation]: Recommandations adaptatives
        """
        recommendations = []
        plant_name = predictions["plant"].get("name", "plante")
        crop_type = self._infer_crop_type(plant_name)
        
        # Recommandations pour les maladies (adaptées au niveau de gravité)
        for disease in predictions.get("diseases", []):
            disease_name = disease["name"]
            severity = disease.get("severity", "moderate")
            confidence = disease.get("confidence", 0.5)
            
            # Adapter selon la gravité et le type de culture
            rec = self._create_disease_recommendation(
                disease_name, severity, confidence, crop_type
            )
            if rec:
                recommendations.append(rec)
        
        # Recommandations pour les carences
        for deficiency in predictions.get("deficiencies", []):
            def_type = deficiency.get("type", "")
            severity = deficiency.get("severity", "light")
            rec = self._create_deficiency_recommendation(def_type, severity, crop_type)
            if rec:
                recommendations.append(rec)
        
        # Recommandations pour le stress hydrique
        water_stress = predictions.get("stress", {}).get("water", 0)
        if water_stress > 0.6:
            rec = self._create_water_stress_recommendation(water_stress, crop_type)
            if rec:
                recommendations.append(rec)
        
        # Si aucune maladie détectée
        if not predictions.get("diseases") and not predictions.get("deficiencies"):
            recommendations.append(Recommendation(
                type="prevention",
                priority="low",
                title="Plante en bonne santé",
                description="Aucun problème détecté. Continuez les bonnes pratiques.",
                steps=[
                    "Maintenir une surveillance régulière",
                    "Continuer les pratiques préventives",
                    "Observer l'évolution"
                ],
                products=[],
                organic_alternatives=[]
            ))
        
        # Trier par priorité
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 4))
        
        return recommendations
    
    def _infer_crop_type(self, plant_name: str) -> str:
        """Infère le type de culture à partir du nom de la plante"""
        plant_lower = plant_name.lower()
        if any(x in plant_lower for x in ["tomate", "tomato", "piment", "poivron"]):
            return "solanaceae"
        elif any(x in plant_lower for x in ["maïs", "corn", "riz", "rice"]):
            return "cereal"
        elif any(x in plant_lower for x in ["haricot", "bean", "pois", "pea"]):
            return "legume"
        else:
            return "general"
    
    def _create_disease_recommendation(
        self,
        disease_name: str,
        severity: str,
        confidence: float,
        crop_type: str
    ) -> Optional[Recommendation]:
        """Crée une recommandation adaptée pour une maladie"""
        # Déterminer la priorité
        if severity in ["severe", "critical"] or confidence > 0.8:
            priority = "urgent"
        elif severity == "moderate" or confidence > 0.6:
            priority = "high"
        else:
            priority = "medium"
        
        # Recommandations spécifiques par maladie
        if disease_name == "Mildiou":
            return Recommendation(
                type="treatment",
                priority=priority,
                title="Traitement du mildiou",
                description=(
                    "Traitement urgent requis" if priority == "urgent" else
                    "Traitement préventif et curatif recommandé"
                ),
                steps=self._get_mildew_steps(severity, crop_type),
                products=["Bouillie bordelaise", "Fongicide systémique"],
                organic_alternatives=["Décoction de prêle", "Bicarbonate de soude"]
            )
        elif disease_name == "Oïdium":
            return Recommendation(
                type="treatment",
                priority=priority,
                title="Traitement de l'oïdium",
                description="Traitement préventif et curatif nécessaire",
                steps=self._get_powdery_mildew_steps(severity),
                products=["Soufre mouillable", "Fongicide anti-oïdium"],
                organic_alternatives=["Lait (10% dans l'eau)", "Bicarbonate de soude"]
            )
        
        return None
    
    def _get_mildew_steps(self, severity: str, crop_type: str) -> List[str]:
        """Étapes de traitement du mildiou adaptées"""
        base_steps = [
            "Retirer les feuilles fortement atteintes",
            "Pulvériser avec bouillie bordelaise (20g/L)",
            "Renouveler tous les 10-15 jours",
            "Éviter l'humidité excessive sur les feuilles"
        ]
        
        if severity in ["severe", "critical"]:
            base_steps.insert(1, "⚠️ Traitement d'urgence: Appliquer un fongicide systémique immédiatement")
        
        if crop_type == "solanaceae":
            base_steps.append("Pour les solanacées: Traiter tôt le matin pour éviter la brûlure")
        
        return base_steps
    
    def _get_powdery_mildew_steps(self, severity: str) -> List[str]:
        """Étapes de traitement de l'oïdium"""
        return [
            "Améliorer la circulation d'air",
            "Pulvériser avec soufre mouillable",
            "Traiter tôt le matin ou en fin de journée",
            "Éviter les traitements en plein soleil"
        ]
    
    def _create_deficiency_recommendation(
        self,
        deficiency_type: str,
        severity: str,
        crop_type: str
    ) -> Optional[Recommendation]:
        """Crée une recommandation pour une carence"""
        priority = "high" if severity in ["moderate", "severe"] else "medium"
        
        if deficiency_type == "Azote":
            return Recommendation(
                type="nutrition",
                priority=priority,
                title="Correction de la carence en azote",
                description="Apport d'azote nécessaire pour la croissance",
                steps=[
                    "Appliquer un engrais azoté (urée, nitrate d'ammonium)",
                    "Ou utiliser du compost bien décomposé",
                    "Arroser après l'application",
                    "Surveiller l'amélioration dans les 7-10 jours"
                ],
                products=["Engrais azoté NPK", "Compost"],
                organic_alternatives=["Fumier composté", "Purin d'ortie"]
            )
        
        return None
    
    def _create_water_stress_recommendation(
        self,
        water_stress_level: float,
        crop_type: str
    ) -> Optional[Recommendation]:
        """Crée une recommandation pour le stress hydrique"""
        priority = "urgent" if water_stress_level > 0.8 else "high"
        
        return Recommendation(
            type="irrigation",
            priority=priority,
            title="Stress hydrique détecté",
            description=(
                "Stress hydrique sévère" if water_stress_level > 0.8 else
                "La plante montre des signes de manque d'eau"
            ),
            steps=[
                "Augmenter la fréquence d'arrosage immédiatement" if water_stress_level > 0.8 else "Augmenter la fréquence d'arrosage",
                "Pailler le sol pour conserver l'humidité",
                "Arroser tôt le matin ou en fin de journée",
                "Vérifier le drainage du sol"
            ],
            products=[],
            organic_alternatives=["Paillage organique", "Système de goutte-à-goutte"]
        )
    
    def _create_audit(
        self,
        detection_id: str,
        user_id: Optional[str],
        image_data: bytes,
        processing_time_ms: float,
        predictions: dict,
        models_used: List[str]
    ) -> DetectionAudit:
        """Crée un enregistrement d'audit"""
        # Hash de l'image pour l'identification
        image_hash = hashlib.md5(image_data).hexdigest()
        
        # Scores de confiance par modèle (doit être Dict[str, float])
        diseases_list = predictions.get("diseases", [])
        diseases_max_confidence = max([d.get("confidence", 0.0) for d in diseases_list], default=0.0)
        
        confidence_scores = {
            "overall": predictions.get("plant", {}).get("confidence", 0.0),
            "diseases_max": diseases_max_confidence,  # Maximum des confidences des maladies
            "num_diseases": float(len(diseases_list))  # Nombre de maladies détectées
        }
        
        return DetectionAudit(
            detection_id=detection_id,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            image_hash=image_hash,
            processing_time_ms=processing_time_ms,
            models_used=models_used,
            confidence_scores=confidence_scores,
            metadata={
                "plant": predictions.get("plant", {}).get("name", "Unknown"),
                "num_diseases": len(predictions.get("diseases", [])),
                "num_deficiencies": len(predictions.get("deficiencies", []))
            }
        )
    
    def _save_audit(self, audit: DetectionAudit):
        """Sauvegarde l'audit dans un fichier JSON"""
        try:
            audit_file = self.audit_dir / f"audit_{audit.detection_id}.json"
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(audit.dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde de l'audit: {e}")
    
    def _generate_detection_id(self, image_data: bytes) -> str:
        """Génère un ID unique pour la détection"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_hash = hashlib.md5(image_data).hexdigest()[:8]
        return f"det_{timestamp}_{image_hash}"
    
    def _build_result(
        self,
        predictions: dict,
        image_shape: tuple,
        top_k_diseases: List[TopPrediction],
        symptom_explanations: List[SymptomExplanation],
        text_description: Optional[str] = None
    ) -> DetectionResult:
        """Construit l'objet DetectionResult enrichi"""
        plant_info = PlantInfo(
            name=predictions["plant"]["name"],
            scientific_name=predictions["plant"].get("scientific_name", ""),
            confidence=predictions["plant"]["confidence"]
        )
        
        diseases = [
            DiseaseInfo(
                name=d["name"],
                confidence=d["confidence"],
                severity=d.get("severity", "moderate"),
                description=d.get("description", "")
            )
            for d in predictions.get("diseases", [])
        ]
        
        recommendations = self._generate_recommendations(predictions)
        overall_severity = self._calculate_severity(predictions)
        
        return DetectionResult(
            plant_info=plant_info,
            diseases=diseases,
            top_k_diseases=top_k_diseases,
            deficiencies=predictions.get("deficiencies", []),
            stress_indicators=predictions.get("stress", {}),
            recommendations=recommendations,
            symptom_explanations=symptom_explanations,
            overall_severity=overall_severity,
            confidence_score=predictions["plant"]["confidence"],
            image_dimensions={"width": image_shape[1], "height": image_shape[0]},
            timestamp=datetime.now().isoformat(),
            text_description=text_description,
            combined_analysis=predictions.get("combined_analysis")
        )
    
    def _calculate_severity(self, predictions: dict) -> str:
        """Calcule le niveau de gravité global"""
        diseases = predictions.get("diseases", [])
        if not diseases:
            return "low"
        
        severities = [d.get("severity", "low") for d in diseases]
        
        if "critical" in severities:
            return "critical"
        elif "severe" in severities:
            return "severe"
        elif "moderate" in severities:
            return "moderate"
        else:
            return "low"
    
    # Méthodes de simulation pour les modèles non implémentés
    async def _simulate_deficiencies(self, image_array: np.ndarray) -> dict:
        """Simule la détection de carences"""
        await asyncio.sleep(0.05)
        return {
            "deficiencies": [
                {"type": "Azote", "confidence": 0.65, "severity": "light"}
            ]
        }
    
    async def _predict_deficiencies(self, image_array: np.ndarray) -> dict:
        """Prédit les carences (à implémenter avec le vrai modèle)"""
        # TODO: Implémenter avec le modèle de carences
        return await self._simulate_deficiencies(image_array)
    
    async def _simulate_water_stress(self, image_array: np.ndarray) -> dict:
        """Simule la détection de stress hydrique"""
        await asyncio.sleep(0.05)
        return {
            "stress": {
                "water": 0.55,
                "temperature": 0.30
            }
        }
    
    async def _predict_water_stress(self, image_array: np.ndarray) -> dict:
        """Prédit le stress hydrique (à implémenter avec le vrai modèle)"""
        # TODO: Implémenter avec le modèle de stress hydrique
        return await self._simulate_water_stress(image_array)
    
    async def _predict_fallback(self, image_array: np.ndarray) -> dict:
        """Prédiction de fallback (simulation)"""
        await asyncio.sleep(0.1)
        return {
            "plant": {
                "name": "Tomate",
                "scientific_name": "Solanum lycopersicum",
                "confidence": 0.85
            },
            "diseases": [
                {
                    "name": "Mildiou",
                    "confidence": 0.75,
                    "severity": "moderate",
                    "description": "Taches brunes sur les feuilles avec duvet blanc"
                }
            ]
        }
