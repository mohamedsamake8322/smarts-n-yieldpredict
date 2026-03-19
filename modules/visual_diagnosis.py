"""
Visual Diagnosis Module

Handles image analysis using Swin Transformer for disease classification,
followed by explanation generation using BLIP-2 and normalized BLIP2 JSON data.
"""

import os
import re
import json
import uuid
from PIL import Image
import torch
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from config import BLIP2_NORMALIZED_DIR, BLIP2_I18N_DIR, SWIN_MODEL_PATH
from models.swin_classifier import SwinDiseaseClassifier
from models.blip2_explainer import BLIP2Explainer
from models.prediction_logger import PredictionLogger

# Import du gestionnaire d'erreurs
from models.error_handler import RobustErrorHandler

class VisualDiagnosis:
    def __init__(
        self,
        blip2_json_dir=None,
        strict_swin_model: bool = True,
        language_code: str = "en",
    ):
        """Initialize the visual diagnosis module.

        Args:
            blip2_json_dir: Directory containing normalized BLIP2 JSON files
                           (defaults to config.BLIP2_NORMALIZED_DIR)
            strict_swin_model: If True, require the trained Swin model + FAISS index
                               to be present (no mock fallback).
        """
        self.language_code = (language_code or "en").lower()

        if blip2_json_dir is not None:
            self.blip2_json_dir = blip2_json_dir
        else:
            if self.language_code not in {"en", ""}:
                candidate_dir = os.path.join(BLIP2_I18N_DIR, self.language_code)
                self.blip2_json_dir = candidate_dir if os.path.exists(candidate_dir) else BLIP2_NORMALIZED_DIR
            else:
                self.blip2_json_dir = BLIP2_NORMALIZED_DIR
        self.strict_swin_model = strict_swin_model

        # Initialize error handler
        self.error_handler = RobustErrorHandler()

        # Initialize Swin classifier with error handling
        self.swin_classifier = self.error_handler.safe_execute(
            lambda: SwinDiseaseClassifier(strict=self.strict_swin_model),
            context="initialization_swin"
        )
        if self.swin_classifier:
            print("✅ Swin classifier initialized")
        else:
            print("❌ Error initializing Swin classifier")

        # Initialize BLIP-2 explainer with error handling
        self.blip2_explainer = self.error_handler.safe_execute(
            lambda: BLIP2Explainer(load_model=False),
            context="initialization_blip2"
        )
        if self.blip2_explainer:
            print("✅ BLIP-2 explainer initialized (deferred loading)")
        else:
            print("❌ Error initializing BLIP-2 explainer")

        # Initialize prediction logger
        self.logger = PredictionLogger()

        # Initialize Agricultural Assistant for RAG (improvement #2)
        self.agricultural_assistant = self.error_handler.safe_execute(
            lambda: __import__('modules.agricultural_assistant', fromlist=['AgriculturalAssistant']).AgriculturalAssistant(),
            context="initialization_agricultural_assistant"
        )
        if self.agricultural_assistant:
            print("✅ Agricultural Assistant initialized for RAG")
        else:
            print("❌ Error initializing Agricultural Assistant")

        # Load BLIP2 data mapping with error handling
        self.blip2_data = self.error_handler.safe_execute(
            self._load_blip2_data,
            context="loading_blip2_data"
        ) or {}

    def _normalize_disease_key(self, name: str) -> str:
        """Normalize disease names to a canonical key for lookups."""
        if not name:
            return ""
        key = name.strip().lower()
        # Normalize delimiters and remove double spaces
        key = key.replace("_", " ").replace("-", " ")
        key = re.sub(r"\s+", " ", key)
        return key

    def _load_blip2_data(self):
        """Load normalized BLIP2 JSON data into a dictionary."""
        data = {}
        if not os.path.exists(self.blip2_json_dir):
            print(f"Warning: BLIP2 directory not found at {self.blip2_json_dir}")
            return data

        for filename in os.listdir(self.blip2_json_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.blip2_json_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        # Use filename without extension as key (normalized for lookup)
                        key = self._normalize_disease_key(filename.replace('.json', ''))
                        data[key] = content
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        return data

    def _add_knowledge_grounding(self, disease_info, disease_name):
        """
        Add knowledge grounding references using FAISS search over Plantwise database.

        Args:
            disease_info: Disease information dictionary (from 109 JSON)
            disease_name: Name of the disease (used as query for FAISS)

        Returns:
            str: Grounding text to append to explanation
        """
        grounding = "\n\n**Recommandations agricoles (Plantwise):**"

        try:
            # Use FAISS to search Plantwise knowledge base
            from modules.agricultural_assistant import AgriculturalAssistant
            assistant = AgriculturalAssistant()

            # Search using disease name as query
            results = assistant.search(f"{disease_name} treatment prevention management", top_k=3)

            if results:
                for i, result in enumerate(results[:2]):  # Show top 2 results
                    # Get detailed content from the JSON file
                    detailed_info = assistant.get_detailed_info(result['filename'])

                    if detailed_info:
                        title = result.get('title', f'Entry {i+1}')
                        content = detailed_info.get('content', '')
                        sections = detailed_info.get('sections', {})

                        # Extract relevant agricultural information
                        if sections.get('treatment') or sections.get('management'):
                            treatment = sections.get('treatment') or sections.get('management', '')
                            grounding += f"\n• **Traitement:** {treatment[:150]}..."
                        elif sections.get('prevention'):
                            prevention = sections.get('prevention', '')
                            grounding += f"\n• **Prévention:** {prevention[:150]}..."
                        elif sections.get('symptoms'):
                            symptoms = sections.get('symptoms', '')
                            grounding += f"\n• **Symptômes:** {symptoms[:150]}..."
                        else:
                            # Use general content
                            grounding += f"\n• **Conseils:** {content[:150]}..."

                        # Add source/reference if available
                        if detailed_info.get('source') or detailed_info.get('reference'):
                            source = detailed_info.get('source') or detailed_info.get('reference')
                            grounding += f" (Source: {source})"

            else:
                # Fallback to basic disease info if FAISS search fails
                if disease_info.get('management'):
                    grounding += f"\n• Gestion: {disease_info['management'][:200]}..."
                if disease_info.get('sources'):
                    grounding += f"\n• Références: {', '.join(disease_info['sources'][:2])}"

        except Exception as e:
            # Fallback if FAISS search fails
            grounding += f"\n• Note: Recherche avancée indisponible ({str(e)[:50]})"
            if disease_info.get('management'):
                grounding += f"\n• Gestion de base: {disease_info['management'][:150]}..."

        grounding += "\n\n*Ces recommandations sont basées sur la base de connaissances Plantwise.*"

        return grounding

    def generate_attention_map(self, image_path, disease_name):
        """
        Generate a simple attention map visualization.

        Args:
            image_path: Path to the image
            disease_name: Predicted disease name

        Returns:
            str: Description of attention areas (simplified version)
        """
        # For now, return a descriptive text about typical attention areas
        # In a full implementation, this would generate actual Grad-CAM visualizations

        attention_areas = {
            'leaf_spots': 'zones tachées ou nécrosées sur les feuilles',
            'leaf_edges': 'bordures des feuilles avec jaunissement',
            'stems': 'tiges avec des lésions ou déformations',
            'fruit': 'fruits avec pourriture ou malformations',
            'whole_plant': 'plante entière montrant un dépérissement général'
        }

        # Simple mapping based on disease type
        disease_lower = disease_name.lower()

        if 'spot' in disease_lower or 'blight' in disease_lower:
            focus_areas = ['leaf_spots', 'leaf_edges']
        elif 'mold' in disease_lower or 'rot' in disease_lower:
            focus_areas = ['fruit', 'stems']
        elif 'virus' in disease_lower:
            focus_areas = ['leaf_edges', 'whole_plant']
        else:
            focus_areas = ['leaf_spots', 'stems']

        attention_text = "Le modèle a focalisé son analyse sur: " + ", ".join([attention_areas[area] for area in focus_areas])

        return attention_text + ". Cette visualisation simplifiée indique les zones de l'image utilisées pour le diagnostic."

    def log_user_feedback(self, prediction_id, user_feedback, correct_disease=None, additional_notes=None):
        """
        Log user feedback on a prediction.

        Args:
            prediction_id: ID of the prediction
            user_feedback: "correct", "incorrect", or "unsure"
            correct_disease: If incorrect, the actual disease
            additional_notes: Additional user comments
        """
        self.logger.log_feedback(prediction_id, user_feedback, correct_disease, additional_notes)
        print(f"✅ Feedback enregistré pour la prédiction {prediction_id}")

    def get_prediction_statistics(self):
        """
        Get statistics about logged predictions.

        Returns:
            dict: Prediction statistics
        """
        return self.logger.get_statistics()

    def export_training_data(self):
        """
        Export logged predictions as training data.

        Returns:
            list: Training data entries
        """
        return self.logger.export_training_data()

    def classify_image(self, image_path):
        """
        Classify disease in image using Swin Transformer.

        Args:
            image_path: Path to the image file

        Returns:
            dict: Classification results with disease name and confidence
        """
        if not self.swin_classifier:
            # Fallback mock result
            return {
                'disease': 'Unknown (model not loaded)',
                'confidence': 0.0,
                'scientific_name': 'Unknown'
            }

        predictions = self.swin_classifier.classify_image(image_path, top_k=1)
        if predictions:
            pred = predictions[0]
            return {
                'disease': pred['disease'],
                'confidence': pred['confidence'],
                'scientific_name': self._get_scientific_name(pred['disease'])
            }
        else:
            return {
                'disease': 'Unknown',
                'confidence': 0.0,
                'scientific_name': 'Unknown'
            }

    def get_top_predictions(self, image_path, top_k=3, confidence_threshold=0.3):
        """
        Get top-k disease predictions with confidence scores and warnings.

        Args:
            image_path: Path to the image
            top_k: Number of top predictions
            confidence_threshold: Minimum confidence to consider prediction reliable

        Returns:
            list: List of predictions with disease names, confidences, and warnings
        """
        if not self.swin_classifier:
            # Fallback mock predictions
            return [
                {'disease': 'Unknown (model not loaded)', 'confidence': 0.0, 'warning': 'Model not available'}
            ] * min(top_k, 3)

        predictions = self.swin_classifier.classify_image(image_path, top_k=top_k)

        # Add warnings for low confidence predictions
        for pred in predictions:
            if pred['confidence'] < confidence_threshold:
                pred['warning'] = f"Low confidence ({pred['confidence']:.1%}). This prediction may be uncertain."
            else:
                pred['warning'] = None

        return [
            {'disease': pred['disease'], 'confidence': pred['confidence'], 'warning': pred.get('warning')}
            for pred in predictions
        ]

    def get_disease_info(self, disease_name):
        """
        Get disease information from BLIP2 data.

        Args:
            disease_name: Name of the disease

        Returns:
            dict: Disease information
        """
        key = self._normalize_disease_key(disease_name)
        return self.blip2_data.get(key, {})

    def get_basic_info(self, disease_info):
        """
        Extract basic disease information for initial display.

        Args:
            disease_info: Full disease information dict

        Returns:
            dict: Simplified info with name, causal agent, and main symptoms
        """
        return {
            'name': disease_info.get('name', 'Unknown disease'),
            'causal_agent': disease_info.get('causal_agent', 'Unknown'),
            'symptoms': disease_info.get('symptoms', 'No symptom info available')[:500]  # First 500 chars
        }

    def get_detailed_info(self, disease_info):
        """
        Extract detailed disease information.

        Args:
            disease_info: Full disease information dict

        Returns:
            dict: Complete information including management and prevention
        """
        return {
            'name': disease_info.get('name', ''),
            'scientific_name': disease_info.get('scientific_name', ''),
            'causal_agent': disease_info.get('causal_agent', ''),
            'hosts': disease_info.get('hosts', []),
            'description': disease_info.get('description', ''),
            'symptoms': disease_info.get('symptoms', ''),
            'management': disease_info.get('management', ''),
            'prevention': disease_info.get('prevention', ''),
            'sources': disease_info.get('sources', [])
        }

    def generate_explanation(self, image_path, disease_info):
        """
        Generate explanation using BLIP-2.

        Args:
            image_path: Path to the image
            disease_info: Disease information dict

        Returns:
            str: Generated explanation
        """
        if self.blip2_explainer:
            # Enforce constrained generation to prevent hallucinations
            return self.blip2_explainer.generate_explanation(
                image_path,
                disease_info,
                use_constrained=True,
                language_code=self.language_code,
            )
        else:
            # Fallback explanation
            name = disease_info.get('name', 'Unknown disease')
            description = disease_info.get('description', 'No description available')
            return f"Based on the image analysis, this appears to be {name}. {description}"

    def _generate_rag_explanation(self, image_path, disease_info, disease_name):
        """
        Generate RAG-enhanced explanation with Plantwise knowledge grounding.

        Args:
            image_path: Path to the image
            disease_info: Disease information from knowledge base
            disease_name: Name of the disease

        Returns:
            str: Enhanced explanation with sources
        """
        try:
            # Get BLIP-2 explanation
            base_explanation = self.generate_explanation(image_path, disease_info)

            # Add RAG context from Agricultural Assistant
            if hasattr(self, 'agricultural_assistant') and self.agricultural_assistant:
                rag_context = self.agricultural_assistant.search_relevant_knowledge(disease_name, top_k=2)

                if rag_context:
                    # Combine explanations
                    enhanced_explanation = f"{base_explanation}\n\n📚 Scientific Context:\n"
                    for i, context in enumerate(rag_context, 1):
                        enhanced_explanation += f"{i}. {context['content']}\n"

                    # Add source attribution
                    sources = [ctx.get('source', 'Plantwise Knowledge Base') for ctx in rag_context]
                    enhanced_explanation += f"\n🔍 Sources: {', '.join(set(sources))}"

                    return enhanced_explanation

            # Fallback to base explanation with knowledge grounding
            return base_explanation + self._add_knowledge_grounding(disease_info, disease_name)

        except Exception as e:
            self.logger.log_error(f"RAG explanation failed: {str(e)}", "visual_diagnosis")
            return self.generate_explanation(image_path, disease_info)

    def diagnose(self, image_path, confidence_threshold=0.3, user_id=None, enable_unknown_detection=True):
        """
        Complete diagnosis pipeline with top-3 explanations, logging, unknown detection, and FAISS validation.

        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence for reliable predictions
            user_id: Optional user identifier for logging
            enable_unknown_detection: Whether to enable out-of-distribution detection

        Returns:
            dict: Diagnosis results with top-3 predictions, explanations, and prediction ID
        """
        # Step 1: Get top predictions with warnings
        predictions = self.get_top_predictions(image_path, top_k=3, confidence_threshold=confidence_threshold)

        # Step 2: Unknown disease detection
        if enable_unknown_detection:
            unknown_result = self._detect_unknown_disease_dynamic(image_path, predictions)
            if unknown_result['is_unknown']:
                # Log unknown detection
                prediction_id = str(uuid.uuid4())
                diagnosis_result = {
                    'unknown_disease': True,
                    'reason': unknown_result['reason'],
                    'confidence': unknown_result['confidence'],
                    'prediction_id': prediction_id
                }
                self.logger.log_prediction(image_path, [], diagnosis_result, user_id)

                return {
                    'predictions': [],
                    'unknown_disease': True,
                    'reason': unknown_result['reason'],
                    'confidence': unknown_result['confidence'],
                    'prediction_id': prediction_id,
                    'explanations': {},
                    'top_disease': 'Unknown',
                    'disease_info': {},
                    'faiss_validation': {},
                    'similar_images': []
                }

        # Step 3: Get info for top prediction
        top_disease = predictions[0]['disease'] if predictions else None
        disease_info = self.get_disease_info(top_disease) if top_disease else {}

        # Step 4: Generate explanations for top-3 predictions
        explanations = {}
        sources = {}

        for i, pred in enumerate(predictions[:3]):  # Top 3
            disease_name = pred['disease']
            if disease_name and disease_name != 'Unknown (model not loaded)':
                # Get disease info
                info = self.get_disease_info(disease_name)

                # Generate BLIP-2 explanation with RAG (improvement #2)
                explanation = self._generate_rag_explanation(image_path, info, disease_name)

                explanations[f'prediction_{i+1}'] = {
                    'disease': disease_name,
                    'confidence': pred['confidence'],
                    'warning': pred.get('warning'),
                    'explanation': explanation
                }

                # Store sources
                if info.get('sources'):
                    sources[disease_name] = info['sources']

        # Step 5: FAISS validation (improvement #3)
        faiss_validation = self._validate_with_faiss(image_path, predictions)

        # Step 6: Apply FAISS decision override if needed
        if faiss_validation.get('decision_override'):
            override_info = faiss_validation['decision_override']
            print(f"🚨 FAISS Override: {override_info['reason']}")

            # Trouver la maladie override dans les prédictions ou l'ajouter
            override_disease = override_info['override_disease']
            override_found = False

            for pred in predictions:
                if pred['disease'] == override_disease:
                    # Remonter cette prédiction en tête
                    pred['confidence'] *= faiss_validation.get('confidence_adjustment', 1.0)
                    pred['faiss_override'] = True
                    predictions.remove(pred)
                    predictions.insert(0, pred)
                    override_found = True
                    break

            if not override_found:
                # Ajouter la maladie FAISS comme nouvelle prédiction
                override_pred = {
                    'disease': override_disease,
                    'confidence': 0.8 * faiss_validation.get('confidence_adjustment', 1.0),  # Confiance élevée mais ajustée
                    'faiss_override': True,
                    'reason': f'Override FAISS: {override_info["reason"]}'
                }
                predictions.insert(0, override_pred)

            # Mettre à jour la maladie top
            top_disease = predictions[0]['disease']

        # Step 7: Get similar images for visual comparison (improvement #4)
        similar_images = self.get_similar_images(image_path, top_k=3)

        # Step 8: Log the prediction
        prediction_id = str(uuid.uuid4())
        diagnosis_result = {
            'predictions': predictions,
            'top_disease': top_disease,
            'explanations': explanations,
            'sources': sources,
            'confidence_threshold': confidence_threshold,
            'faiss_validation': faiss_validation,
            'similar_images': similar_images,
            'prediction_id': prediction_id,
            'unknown_disease': False
        }

        self.logger.log_prediction(image_path, predictions, diagnosis_result, user_id)

        return diagnosis_result

        # Add prediction ID to result
        diagnosis_result['prediction_id'] = prediction_id

        return diagnosis_result

    def _detect_unknown_disease_dynamic(self, image_path, predictions, historical_predictions=None):
        """
        Détection dynamique des maladies inconnues basée sur la distribution des prédictions.

        Args:
            image_path: Path to the image
            predictions: List of predictions from the model
            historical_predictions: Optional historical predictions for baseline

        Returns:
            dict: {'is_unknown': bool, 'reason': str, 'confidence': float, 'score': float}
        """
        if not predictions:
            return {'is_unknown': True, 'reason': 'Aucune prédiction disponible', 'confidence': 0.0, 'score': 1.0}

        # Extraire les confiances
        confidences = [p['confidence'] for p in predictions]

        # Méthode 1: Analyse statistique des prédictions actuelles
        if len(confidences) >= 3:
            # Calculer les métriques statistiques
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            max_conf = max(confidences)
            min_conf = min(confidences)

            # Score d'incertitude basé sur la distribution
            # Plus l'écart-type est élevé et la confiance max faible, plus c'est suspect
            uncertainty_score = (1 - max_conf) * (std_conf / (mean_conf + 1e-6))

            # Seuil dynamique basé sur percentile (85ème percentile comme seuil)
            sorted_confidences = sorted(confidences, reverse=True)
            dynamic_threshold = np.percentile(sorted_confidences, 85)

            # Si la meilleure prédiction est en dessous du seuil dynamique
            if max_conf < dynamic_threshold * 0.7:  # 30% en dessous du seuil dynamique
                return {
                    'is_unknown': True,
                    'reason': f'Confiance maximale ({max_conf:.3f}) en dessous du seuil dynamique ({dynamic_threshold:.3f})',
                    'confidence': max_conf,
                    'score': uncertainty_score
                }

        # Méthode 2: Comparaison avec données historiques (si disponibles)
        if historical_predictions and len(historical_predictions) > 10:
            # Calculer la moyenne historique des meilleures prédictions
            historical_best = [max([p['confidence'] for p in pred_set]) for pred_set in historical_predictions[-50:]]  # Dernières 50
            historical_mean = np.mean(historical_best)
            historical_std = np.std(historical_best)

            # Si la prédiction actuelle est 2 écarts-types en dessous de la moyenne historique
            if max_conf < (historical_mean - 2 * historical_std):
                return {
                    'is_unknown': True,
                    'reason': f'Confiance ({max_conf:.3f}) anormalement basse vs historique (μ={historical_mean:.3f}, σ={historical_std:.3f})',
                    'confidence': max_conf,
                    'score': abs(max_conf - historical_mean) / (historical_std + 1e-6)
                }

        # Méthode 3: Analyse de l'entropie normalisée
        if len(confidences) >= 3:
            # Calculer l'entropie de Shannon normalisée
            probs = np.array(confidences) / sum(confidences)
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            max_entropy = np.log(len(confidences))  # Entropie maximale possible
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Si l'entropie est très élevée (> 0.8), les prédictions sont très incertaines
            if normalized_entropy > 0.8:
                return {
                    'is_unknown': True,
                    'reason': f'Entropie normalisée élevée ({normalized_entropy:.3f} > 0.8) - prédictions très incertaines',
                    'confidence': max_conf,
                    'score': normalized_entropy
                }

        # Méthode 4: Distance FAISS avec seuil adaptatif
        try:
            if self.swin_classifier and hasattr(self.swin_classifier, 'faiss_index'):
                features = self.swin_classifier.extract_features(image_path)
                if features is not None:
                    distances, _ = self.swin_classifier.faiss_index.search(features.reshape(1, -1), k=5)
                    min_distance = distances[0][0]

                    # Seuil adaptatif basé sur la distribution des distances dans l'index
                    # Pour l'instant, seuil fixe mais pourrait être calculé dynamiquement
                    adaptive_threshold = 1.5  # À améliorer avec calcul dynamique

                    if min_distance > adaptive_threshold:
                        return {
                            'is_unknown': True,
                            'reason': f'Distance FAISS élevée ({min_distance:.3f} > {adaptive_threshold:.3f}) - hors distribution',
                            'confidence': max_conf,
                            'score': min_distance / adaptive_threshold
                        }
        except Exception as e:
            # Ne pas échouer si FAISS n'est pas disponible
            pass

        return {'is_unknown': False, 'reason': 'Maladie reconnue', 'confidence': max_conf, 'score': 0.0}

    def _validate_with_faiss(self, image_path, predictions):
        """
        Validate predictions using FAISS similarity search and impact final decision.

        Args:
            image_path: Path to the image
            predictions: List of predictions

        Returns:
            dict: Validation results with potential decision override
        """
        validation = {
            'validated': False,
            'warnings': [],
            'similar_diseases': [],
            'decision_override': None,
            'confidence_adjustment': 1.0
        }

        try:
            if not self.swin_classifier or not hasattr(self.swin_classifier, 'faiss_index'):
                validation['warnings'].append('FAISS index non disponible')
                return validation

            # Get image features
            features = self.swin_classifier.extract_features(image_path)
            if features is None:
                validation['warnings'].append('Extraction de features échouée')
                return validation

            # Search similar images in FAISS
            distances, indices = self.swin_classifier.faiss_index.search(
                features.reshape(1, -1), k=10  # Augmenter pour meilleure validation
            )

            # Get metadata for similar images
            similar_diseases = []
            disease_counts = {}

            for i, idx in enumerate(indices[0]):
                if idx < len(self.swin_classifier.metadata):
                    meta = self.swin_classifier.metadata[idx]
                    disease = meta.get('disease', 'Unknown')
                    distance = float(distances[0][i])

                    similar_diseases.append({
                        'disease': disease,
                        'distance': distance,
                        'index': i
                    })

                    # Compter les occurrences de chaque maladie
                    if disease not in disease_counts:
                        disease_counts[disease] = []
                    disease_counts[disease].append(distance)

            validation['similar_diseases'] = similar_diseases
            validation['validated'] = True

            # Analyse de cohérence avancée
            if predictions and similar_diseases:
                top_pred = predictions[0]['disease']
                top_pred_confidence = predictions[0]['confidence']

                # Calculer le consensus FAISS
                faiss_consensus = {}
                for disease, distances in disease_counts.items():
                    avg_distance = np.mean(distances)
                    count = len(distances)
                    faiss_consensus[disease] = {
                        'avg_distance': avg_distance,
                        'count': count,
                        'score': count / (avg_distance + 1e-6)  # Score basé sur fréquence et proximité
                    }

                # Trier par score FAISS
                sorted_faiss = sorted(faiss_consensus.items(), key=lambda x: x[1]['score'], reverse=True)
                faiss_top_disease = sorted_faiss[0][0] if sorted_faiss else None
                faiss_top_score = sorted_faiss[0][1]['score'] if sorted_faiss else 0

                # Détection d'incohérence majeure
                if faiss_top_disease and faiss_top_disease != top_pred:
                    faiss_distance = faiss_consensus[faiss_top_disease]['avg_distance']

                    # Condition d'override: FAISS très confiant ET Swin peu confiant
                    if faiss_distance < 1.0 and top_pred_confidence < 0.7:
                        validation['decision_override'] = {
                            'original_disease': top_pred,
                            'override_disease': faiss_top_disease,
                            'reason': f'Incohérence détectée: FAISS suggère {faiss_top_disease} '
                                     f'(distance: {faiss_distance:.3f}) vs Swin {top_pred} '
                                     f'(confiance: {top_pred_confidence:.3f})',
                            'faiss_evidence': faiss_consensus[faiss_top_disease]
                        }
                        validation['warnings'].append(
                            f'🚨 OVERRIDE: Prédiction corrigée de {top_pred} vers {faiss_top_disease} '
                            f'basé sur similarité FAISS'
                        )

                    elif faiss_distance < 1.5:
                        # Avertissement sans override
                        validation['warnings'].append(
                            f'Incohérence détectée: FAISS suggère {faiss_top_disease} '
                            f'(distance: {faiss_distance:.3f}) au lieu de {top_pred}'
                        )

                        # Ajustement de confiance
                        confidence_penalty = min(0.3, faiss_distance / 5.0)
                        validation['confidence_adjustment'] = max(0.1, 1.0 - confidence_penalty)

        except Exception as e:
            validation['warnings'].append(f'Erreur validation FAISS: {str(e)}')

        return validation

    def get_similar_images(self, image_path, top_k=3):
        """
        Get similar images from the training dataset using FAISS.

        Args:
            image_path: Path to the query image
            top_k: Number of similar images to return

        Returns:
            list: List of similar images with metadata
        """
        try:
            if not self.swin_classifier or not hasattr(self.swin_classifier, 'faiss_index'):
                return []

            # Get image features
            features = self.swin_classifier.extract_features(image_path)
            if features is None:
                return []

            # Search similar images
            distances, indices = self.swin_classifier.faiss_index.search(
                features.reshape(1, -1), k=top_k
            )

            similar_images = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.swin_classifier.metadata):
                    meta = self.swin_classifier.metadata[idx]
                    similar_images.append({
                        'disease': meta.get('disease', 'Unknown'),
                        'confidence': meta.get('confidence', 0.0),
                        'distance': float(distances[0][i]),
                        'image_path': meta.get('image_path', ''),
                        'metadata': meta
                    })

            return similar_images

        except Exception as e:
            print(f"Error getting similar images: {e}")
            return []