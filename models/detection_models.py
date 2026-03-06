"""
Modèles de données pour la détection des plantes et maladies
"""

from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class PlantInfo(BaseModel):
    """Informations sur la plante détectée"""
    name: str
    scientific_name: str
    confidence: float

class DiseaseInfo(BaseModel):
    """Informations sur une maladie détectée"""
    name: str
    confidence: float
    severity: str  # low, moderate, severe, critical
    description: Optional[str] = None

class Recommendation(BaseModel):
    """Recommandation de traitement ou d'action"""
    type: str  # treatment, prevention, nutrition, irrigation
    priority: str  # low, medium, high, urgent
    title: str
    description: str
    steps: List[str]
    products: List[str]
    organic_alternatives: List[str]

class TopPrediction(BaseModel):
    """Prédiction avec score pour Top-K"""
    name: str
    confidence: float
    severity: str
    description: Optional[str] = None

class SymptomExplanation(BaseModel):
    """Explication pédagogique des symptômes"""
    symptom: str
    explanation: str
    visual_description: str
    common_causes: List[str]

class DetectionAudit(BaseModel):
    """Informations d'audit pour une détection"""
    detection_id: str
    timestamp: str
    user_id: Optional[str] = None
    image_hash: Optional[str] = None
    processing_time_ms: float
    models_used: List[str]
    confidence_scores: Dict[str, float]
    metadata: Dict

class DetectionResult(BaseModel):
    """Résultat complet d'une détection"""
    plant_info: PlantInfo
    diseases: List[DiseaseInfo]
    top_k_diseases: List[TopPrediction]  # Top 3 prédictions
    deficiencies: List[Dict]
    stress_indicators: Dict
    recommendations: List[Recommendation]
    symptom_explanations: List[SymptomExplanation]  # Explications pédagogiques
    overall_severity: str  # low, moderate, severe, critical
    confidence_score: float
    image_dimensions: Dict[str, int]
    timestamp: str
    text_description: Optional[str] = None  # Description textuelle fournie par l'utilisateur
    combined_analysis: Optional[str] = None  # Analyse combinée image + texte
    audit: Optional[DetectionAudit] = None





