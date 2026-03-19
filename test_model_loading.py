#!/usr/bin/env python
"""Test si le modèle peut être chargé correctement en local"""

import os
import sys

# Vérifier le répertoire courant
print(f"📂 Répertoire courant: {os.getcwd()}")
print(f"📂 Fichiers config.py existe: {os.path.exists('config.py')}")

# Importer config
from config import BASE_PATH, SWIN_MODEL_PATH, SWIN_FAISS_INDEX, SWIN_METADATA

print(f"\n✅ Configuration chargée:")
print(f"   BASE_PATH: {BASE_PATH}")
print(f"   SWIN_MODEL_PATH: {SWIN_MODEL_PATH}")
print(f"   SWIN_FAISS_INDEX: {SWIN_FAISS_INDEX}")
print(f"   SWIN_METADATA: {SWIN_METADATA}")

# Vérifier les fichiers
print(f"\n🔍 Vérification des fichiers:")
print(f"   Modèle existe: {os.path.exists(SWIN_MODEL_PATH)}")
print(f"   FAISS index existe: {os.path.exists(SWIN_FAISS_INDEX)}")
print(f"   Métadonnées existent: {os.path.exists(SWIN_METADATA)}")

# Essayer charger le classifier
print(f"\n🤖 Tentative de chargement du classifier...")
try:
    from models.swin_classifier import SwinDiseaseClassifier
    
    classifier = SwinDiseaseClassifier()
    
    if classifier.model is not None:
        print(f"✅ SUCCÈS! Modèle chargé correctement")
        print(f"   Modèle: {classifier.model.__class__.__name__}")
        print(f"   Classes: {len(classifier.class_names)} maladies détectées")
    else:
        print(f"❌ Modèle non chargé")
        
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    import traceback
    traceback.print_exc()
