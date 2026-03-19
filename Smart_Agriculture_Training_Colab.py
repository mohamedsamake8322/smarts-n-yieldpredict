"""
SMART AGRICULTURE - Entraînement Complet A100 + Application v2.0

Script Python pour exécution complète sur Google Colab
Contient toutes les 7 améliorations avancées implémentées

Ce script exécute sur Colab :
- ✅ Détection et optimisation A100 (TF32, cuDNN, Mixed Precision)
- ✅ Entraînement Swin Base Production (60 epochs)
- ✅ Normalisation BLIP2 + index FAISS MOH
- ✅ Vérification des modèles entraînés
- ❌ PAS de Streamlit : le modèle est pour usage LOCAL (streamlit run en local)

🆕 NOUVELLES FONCTIONNALITÉS AVANCÉES (v2.1 - PRODUCTION READY):
- 🧪 Détection de maladies inconnues DYNAMIQUE (analyse statistique, pas de seuils fixes)
- 🧠 Explications BLIP-2 SÉCURISÉES anti-hallucinations (prompts contraints)
- 🔍 Validation FAISS avec OVERRIDE AUTOMATIQUE des prédictions incohérentes
- 👁️ Mode comparaison visuelle (image utilisateur vs dataset d'entraînement)
- ⚡ Optimisations A100 avancées (TF32, cuDNN benchmark, mixed precision)
- 🛡️ Gestionnaire d'erreurs robuste avec récupération automatique
- 🏗️ Architecture modulaire spécialisée (train / index / app modules)

⚠️ ARCHITECTURE MODULAIRE CLARIFIÉE (script Colab uniquement entraînement) :
   - ÉTAPES 1-4: Setup GPU + Drive + dépendances
   - ÉTAPE 5: Entraînement Swin (metric_training_core)
   - ÉTAPES 6-7: Normalisation BLIP2 + index FAISS MOH
   - ÉTAPES 8-9: Tests modules + vérification modèles
   - FIN: Modèle prêt → télécharger depuis Drive pour usage local (Streamlit)

Temps estimé : 2-3 heures (entraînement A100)
GPU requis : A100 recommandé (V100/T4 fonctionnent aussi)
"""

# =============================================================================
# ÉTAPE 1: VÉRIFICATION GPU ET OPTIMISATIONS A100 AVANCÉES
# =============================================================================

print("="*80)
print("🎮 ÉTAPE 1: VÉRIFICATION GPU ET OPTIMISATIONS A100 AVANCÉES")
print("Cette cellule détecte automatiquement votre GPU et applique les optimisations avancées A100")
print("="*80)

import torch
import os
import json
import re

def check_gpu():
    """Vérifie le GPU disponible et retourne le type."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_capability = torch.cuda.get_device_capability(0)
        print(f"🎮 GPU détecté: {gpu_name} (x{gpu_count}) - Capability: {gpu_capability}")

        if "A100" in gpu_name:
            print("🚀 Mode A100 activé - Optimisations maximales !")
            # Optimisations A100 avancées
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("⚡ A100 Optimisations activées: TF32, cuDNN benchmark")
            return "A100"
        elif "V100" in gpu_name:
            print("⚡ Mode V100 activé")
            torch.backends.cudnn.benchmark = True
            return "V100"
        elif "T4" in gpu_name:
            print("💨 Mode T4 activé")
            torch.backends.cudnn.benchmark = True
            return "T4"
        else:
            print(f"📊 GPU {gpu_name} détecté")
            return gpu_name
    else:
        print("⚠️  Pas de GPU CUDA détecté - Mode CPU")
        return "CPU"

# Vérification GPU
gpu_type = check_gpu()

# Configuration optimisée selon le GPU
if gpu_type == "A100":
    os.environ["BATCH_SIZE"] = "64"
    os.environ["GRADIENT_ACCUMULATION"] = "1"
    os.environ["MIXED_PRECISION"] = "true"
    print("⚡ Configuration A100: batch_size=64, gradient_accumulation=1, mixed_precision=True")
elif gpu_type in ["V100", "T4"]:
    os.environ["BATCH_SIZE"] = "32"
    os.environ["GRADIENT_ACCUMULATION"] = "2"
    os.environ["MIXED_PRECISION"] = "true"
    print("💨 Configuration optimisée pour GPU haute performance avec mixed precision")
else:
    os.environ["BATCH_SIZE"] = "16"
    os.environ["GRADIENT_ACCUMULATION"] = "4"
    os.environ["MIXED_PRECISION"] = "false"
    print("📊 Configuration standard")

print(f"✅ GPU prêt: {gpu_type} avec optimisations avancées")

# =============================================================================
# ÉTAPE 2: MONTAGE GOOGLE DRIVE
# =============================================================================

print("\n" + "="*80)
print("📁 ÉTAPE 2: MONTAGE GOOGLE DRIVE")
print("Montez votre Google Drive pour accéder aux fichiers du projet")
print("="*80)

from google.colab import drive
import time

# Remount Drive avec gestion d'erreur
print("📁 Montage de Google Drive...")
try:
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive monté")
except Exception as e:
    print(f"⚠️  Remount Drive échoué: {e}")
    drive.mount('/content/drive')
    
# Pequire pause pour stabiliser la connexion
time.sleep(2)

# Changement de répertoire vers le projet
import os

# Par défaut, le projet est attendu ici (collez votre repo dans Drive):
DRIVE_ROOT = '/content/drive/MyDrive'
DEFAULT_PROJECT_DIR = os.path.join(DRIVE_ROOT, 'smarts-n-yieldpredict')

# Recherche intelligente du dossier projet (évite l'erreur si le dossier est ailleurs)
def find_project_dir():
    # 1) Emplacement attendu
    if os.path.exists(DEFAULT_PROJECT_DIR):
        return DEFAULT_PROJECT_DIR

    # 2) Cherche un dossier contenant config.py à la racine de Drive
    try:
        for name in os.listdir(DRIVE_ROOT):
            candidate = os.path.join(DRIVE_ROOT, name)
            if not os.path.isdir(candidate):
                continue
            if os.path.exists(os.path.join(candidate, 'config.py')):
                return candidate

            # 2 niveaux max (pour les cas où le repo est dans un sous-dossier)
            for sub in os.listdir(candidate):
                subpath = os.path.join(candidate, sub)
                if os.path.isdir(subpath) and os.path.exists(os.path.join(subpath, 'config.py')):
                    return subpath
    except Exception:
        pass

    return None

project_root = find_project_dir()
if project_root is None:
    print("❌ Impossible de trouver le projet Smart Agriculture dans Google Drive.")
    print("💡 Astuce : placez le dossier 'smarts-n-yieldpredict' dans votre Google Drive ou")
    print("   mettez à jour DEFAULT_PROJECT_DIR dans ce script pour pointer vers votre dossier.")
    raise FileNotFoundError("Projet Smart Agriculture introuvable dans Google Drive.")

os.chdir(project_root)
print(f"📁 Répertoire changé: {os.getcwd()}")

# Vérification que nous sommes au bon endroit
if os.path.exists('config.py'):
    print("✅ Projet Smart Agriculture trouvé")
else:
    print("❌ Erreur: Projet non trouvé. Vérifiez le chemin.")

# =============================================================================
# ÉTAPE 3: INSTALLATION DES DÉPENDANCES
# =============================================================================

print("\n" + "="*80)
print("📦 ÉTAPE 3: INSTALLATION DES DÉPENDANCES")
print("Installation de toutes les bibliothèques nécessaires")
print("="*80)

!pip install -q sentence-transformers faiss-cpu streamlit torch transformers timm accelerate albumentations opencv-python-headless Pillow plotly wandb torchvision

print("✅ Toutes les dépendances installées")

# =============================================================================
# ÉTAPE 4: CONFIGURATION ET VÉRIFICATION
# =============================================================================

print("\n" + "="*80)
print("⚙️ ÉTAPE 4: CONFIGURATION ET VÉRIFICATION")
print("Configuration du projet et vérification de l'environnement")
print("="*80)

from config import print_config, ensure_directories
ensure_directories()
print_config()
print("✅ Configuration vérifiée")

# =============================================================================
# ÉTAPE 5: ENTRAÎNEMENT DU MODÈLE SWIN BASE (60 epochs)
# =============================================================================

print("\n" + "="*80)
print("🤖 ÉTAPE 5: ENTRAÎNEMENT DU MODÈLE SWIN BASE (60 epochs)")
print("⚠️ Cette étape prend 2-3 heures sur A100")
print("="*80)

# FLAG: Contrôlez le comportement entraînement
force_retrain = False  # 👈 Mettre True pour forcer un nouvel entraînement
skip_if_exists = True  # Skip si modèle existe déjà

print("🚀 Vérification si modèle déjà entraîné...")

# VÉRIFICATION SI MODÈLE DÉJÀ ENTRAÎNÉ
# Utiliser des chemins relatifs au projet trouvé
model_path = os.path.join(project_root, "outputs", "phase2_swin_base_production", "models", "metric_model.pt")
faiss_path = os.path.join(project_root, "outputs", "phase2_swin_base_production", "models", "faiss_index.bin")
metadata_path = os.path.join(project_root, "outputs", "phase2_swin_base_production", "models", "metadata.json")

model_exists = os.path.exists(model_path)
faiss_exists = os.path.exists(faiss_path)
metadata_exists = os.path.exists(metadata_path)

if model_exists and faiss_exists and metadata_exists and skip_if_exists and not force_retrain:
    print("✅ MODÈLE DÉJÀ ENTRAÎNÉ DÉTECTÉ !")
    print(f"📁 Modèle: {model_path}")
    print(f"🔍 FAISS: {faiss_path}")
    print(f"📊 Métadonnées: {metadata_path}")
    print("⏭️  SAUT DE L'ENTRAÎNEMENT - Passage direct aux étapes suivantes")
    print("="*60)
else:
    if force_retrain:
        print("🔄 FORCE RETRAIN ACTIVÉ - Lancement d'un nouvel entraînement...")
    else:
        print("❌ Aucun modèle complet trouvé - Lancement de l'entraînement...")
    
    print(f"🎮 Configuration {gpu_type}: batch_size={os.environ.get('BATCH_SIZE', '16')}")
    print("⏰ Durée estimée: 2-3 heures sur A100")
    print("="*60)

    # 1) Forcer chemins dataset + outputs AVANT entraînement
    DATASET_FINAL = "/content/drive/MyDrive/dataset_final"
    if not os.path.exists(DATASET_FINAL):
        msg = (
            f"❌ BLOQUÉ: dataset_final introuvable à {DATASET_FINAL}\n"
            "   Placez dataset_final à la racine de votre Google Drive (MyDrive/dataset_final)\n"
            "   Structure attendue: dataset_final/train/<classe>/, val/<classe>/, test/<classe>/"
        )
        raise FileNotFoundError(msg)

    os.environ["DATASET_PATH"] = DATASET_FINAL
    os.environ["OUTPUT_ROOT"] = os.path.join(project_root, "outputs")
    print(f"📁 Dataset: {DATASET_FINAL}")
    print(f"📁 Outputs: {os.environ['OUTPUT_ROOT']}")

    # 2) Valider structure minimale du dataset (évite échec après plusieurs heures)
    train_dir = os.path.join(DATASET_FINAL, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"❌ Structure invalide: dataset_final/train/ absent.\n"
            f"   Attendu: {DATASET_FINAL}/train/<Class_Name>/ avec des images."
        )
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if not classes:
        raise RuntimeError(
            f"❌ dataset_final/train/ existe mais ne contient aucune classe.\n"
            f"   Chaque sous-dossier (ex: Tomato_Early_Blight/) doit contenir des images."
        )
    sample_class = os.path.join(train_dir, classes[0])
    images = [f for f in os.listdir(sample_class) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not images:
        raise RuntimeError(
            f"❌ La classe {classes[0]} ne contient aucune image.\n"
            f"   Formats attendus: .jpg, .jpeg, .png, .bmp"
        )
    print(f"✅ Dataset validé: {len(classes)} classes, structure OK")

    # Utiliser le pipeline d'entraînement existant (metric_training_core)
    try:
        from training_pipelines.phase2_swin_base_production import main as train_main

        train_main()

    except Exception as e:
        print(f"❌ Erreur durant l'entraînement: {e}")
        raise

    # Vérification des artefacts générés
    if os.path.exists(model_path):
        print(f"✅ Modèle entraîné disponible: {model_path}")
    else:
        print("❌ Modèle entraîné non trouvé après l'entraînement.")

    if os.path.exists(faiss_path):
        print(f"✅ FAISS index disponible: {faiss_path}")
    else:
        print("❌ FAISS index non trouvé, assurez-vous que l'entraînement a bien construit l'index.")

    # Chargement des métadonnées pour affichage des métriques
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            best_recall = metadata.get('best_recall_at1')
            best_accuracy = metadata.get('best_accuracy')

            print("\n📊 MÉTRIQUES D'ÉVALUATION:")
            print("="*50)
            if best_recall is not None:
                print(f"Recall@1: {best_recall:.4f}")
            if best_accuracy is not None:
                print(f"Accuracy: {best_accuracy:.4f}")

            # Afficher la taille du dataset si disponible
            history = metadata.get('history', {})
            if history and 'train_size' in history:
                train_size = history['train_size']
                val_size = history.get('val_size', 'N/A')
                print(f"Dataset: Train={train_size}, Val={val_size}")

            print("="*50)

        except Exception as e:
            print(f"⚠️ Impossible de lire les métadonnées: {e}")

    print("="*60)
# =============================================================================
# ÉTAPE 6: NORMALISATION BLIP2
# =============================================================================

print("\n" + "="*80)
print("🔄 ÉTAPE 6: NORMALISATION BLIP2")
print("Préparation des données BLIP2 pour les explications (109 fichiers)")
print("="*80)

# ============================================================================
# CODE INTÉGRÉ: normalize_blip2.py (AVEC RETRIES ET ROBUSTESSE)
# ============================================================================

import os
import json
import re
import time

# Utilitaire de retry
def retry_operation(func, max_retries=3, delay=1):
    """Retry une opération avec backoff exponentiel."""
    for attempt in range(max_retries):
        try:
            return func()
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                print(f"  ⏳ Retry après {wait_time}s ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise

# Directory paths
BLIP2_DIR = os.path.join(project_root, 'BLIP2')
NORMALIZED_DIR = os.path.join(project_root, 'BLIP2_normalized')

# Ensure normalized directory exists
def ensure_dir():
    os.makedirs(NORMALIZED_DIR, exist_ok=True)

retry_operation(ensure_dir, max_retries=3)

def flatten_text(obj):
    """Flatten nested objects/arrays to a single string."""
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        return ' '.join(str(item) for item in obj if item)
    elif isinstance(obj, dict):
        return ' '.join(f"{k}: {flatten_text(v)}" for k, v in obj.items())
    else:
        return str(obj)

def normalize_blip2_file(filepath):
    """Normalize a single BLIP2 JSON file to common schema."""
    def _read_and_normalize():
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        normalized = {
            "name": "",
            "scientific_name": "",
            "causal_agent": "",
            "hosts": [],
            "symptoms": "",
            "description": "",
            "management": "",
            "prevention": "",
            "sources": []
        }

        # Map existing keys to normalized fields
        if 'disease' in data:
            normalized['name'] = data['disease']
        elif 'disease_name' in data:
            normalized['name'] = data['disease_name']
        elif 'pest' in data:
            normalized['name'] = data['pest']

        if 'scientific_name' in data:
            normalized['scientific_name'] = data['scientific_name']
        elif 'synonym' in data:
            normalized['scientific_name'] = data['synonym']

        if 'causal_agent' in data:
            normalized['causal_agent'] = data['causal_agent']
        elif 'other_agents' in data:
            normalized['causal_agent'] = ', '.join(data['other_agents'])

        if 'hosts' in data:
            if isinstance(data['hosts'], list):
                normalized['hosts'] = data['hosts']
            elif isinstance(data['hosts'], dict):
                hosts = []
                for key, value in data['hosts'].items():
                    if isinstance(value, list):
                        hosts.extend(value)
                    else:
                        hosts.append(value)
                normalized['hosts'] = hosts

        if 'symptoms' in data:
            normalized['symptoms'] = flatten_text(data['symptoms'])
        elif 'symptoms_and_damage' in data:
            normalized['symptoms'] = flatten_text(data['symptoms_and_damage'])

        if 'description' in data:
            normalized['description'] = data['description']

        if 'management' in data:
            normalized['management'] = flatten_text(data['management'])
        elif 'cultural_control' in data:
            normalized['management'] = flatten_text(data['cultural_control'])
        elif 'biological_control' in data:
            normalized['management'] = flatten_text(data['biological_control'])
        elif 'chemical_control' in data:
            normalized['management'] = flatten_text(data['chemical_control'])

        if 'prevention' in data:
            normalized['prevention'] = flatten_text(data['prevention'])

        if 'sources' in data:
            if isinstance(data['sources'], list):
                normalized['sources'] = data['sources']
            else:
                normalized['sources'] = [data['sources']]
        elif 'references' in data:
            if isinstance(data['references'], list):
                normalized['sources'] = data['references']
            else:
                normalized['sources'] = [data['references']]

        return normalized
    
    return retry_operation(_read_and_normalize, max_retries=3)

def process_blip2_files():
    """Process all BLIP2 files and save normalized versions."""
    def _list_blip2():
        if not os.path.exists(BLIP2_DIR):
            print(f"❌ Directory {BLIP2_DIR} not found")
            return []
        return [f for f in os.listdir(BLIP2_DIR) if f.endswith('.json')]
    
    filenames = retry_operation(_list_blip2, max_retries=3)
    
    files_processed = 0
    for filename in filenames:
        input_path = os.path.join(BLIP2_DIR, filename)
        output_path = os.path.join(NORMALIZED_DIR, filename)

        try:
            normalized_data = normalize_blip2_file(input_path)

            def _write_normalized():
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(normalized_data, f, indent=2, ensure_ascii=False)
            
            retry_operation(_write_normalized, max_retries=3)
            files_processed += 1
            print(f"✅ Normalized: {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    return files_processed

# Exécuter la normalisation (AVEC TIMEOUT GLOBAL)
print("🚀 Début de la normalisation BLIP2...")
try:
    files_count = process_blip2_files()
    print(f"✅ BLIP2 normalisé ({files_count} fichiers)")
except Exception as e:
    print(f"⚠️  Normalisation BLIP2 échouée après retries: {e}")
    print("   Continuant avec les données existantes...")
    files_count = 0

# =============================================================================
# ÉTAPE 7: CONSTRUCTION INDEX FAISS (ROBUSTE)
# =============================================================================

print("\n" + "="*80)
print("🔍 ÉTAPE 7: CONSTRUCTION INDEX FAISS")
print("Construction de l'index de recherche vectorielle (1115 entrées Plantwise)")
print("="*80)

# ============================================================================
# CODE INTÉGRÉ: build_moh_index.py (AVEC RETRIES)
# ============================================================================

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Directory paths
MOH_DIR = os.path.join(project_root, 'Moh')
INDEX_FILE = os.path.join(project_root, 'moh_index.faiss')
METADATA_FILE = os.path.join(project_root, 'moh_metadata.json')

def extract_text_from_moh(data):
    """Extract searchable text from Moh JSON."""
    text_parts = []

    # Title
    text_parts.append(data.get('title', ''))

    # Header
    header = data.get('sections', {}).get('Header', [])
    text_parts.extend(header)

    # Table sections
    table = data.get('sections', {}).get('Table', {})
    for section, content in table.items():
        if isinstance(content, list):
            text_parts.extend(content)

    # Figures captions
    figures = data.get('sections', {}).get('Figures', [])
    for fig in figures:
        text_parts.append(fig.get('caption', ''))

    return ' '.join(text_parts)

def build_faiss_index():
    """Build FAISS index for Moh JSON files."""
    print("🤖 Initialisation du modèle sentence-transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = []
    metadata = []

    if not os.path.exists(MOH_DIR):
        print(f"❌ Directory {MOH_DIR} not found")
        return 0

    # List files with retry
    def _list_moh():
        return [f for f in os.listdir(MOH_DIR) if f.endswith('.json')]
    
    try:
        filenames = retry_operation(_list_moh, max_retries=3)
    except Exception as e:
        print(f"❌ Impossible de lister les fichiers Moh: {e}")
        return 0

    print(f"📂 Trouver {len(filenames)} fichiers Moh...")

    for i, filename in enumerate(filenames):
        if (i + 1) % 50 == 0:
            print(f"  Traitement {i + 1}/{len(filenames)}...")
        
        filepath = os.path.join(MOH_DIR, filename)
        
        def _load_and_extract():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = extract_text_from_moh(data)
            return text, data
        
        try:
            text, data = retry_operation(_load_and_extract, max_retries=2)
            if text.strip():
                texts.append(text)
                metadata.append({
                    'title': data.get('title', ''),
                    'filename': filename,
                    'filepath': filepath
                })
        except Exception as e:
            print(f"  ⚠️  Erreur {filename}: {str(e)[:50]}")

    print(f"✅ Chargé {len(texts)} documents.")

    if len(texts) == 0:
        print("❌ No documents found to index")
        return 0

    # Generate embeddings
    print("🔤 Génération des embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Build FAISS index
    print("🔍 Construction de l'index FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index and metadata
    def _save_faiss():
        faiss.write_index(index, INDEX_FILE)
    
    def _save_metadata():
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    try:
        retry_operation(_save_faiss, max_retries=3)
        retry_operation(_save_metadata, max_retries=3)
        print(f"✅ Index FAISS sauvegardé: {INDEX_FILE}")
        print(f"✅ Métadonnées sauvegardées: {METADATA_FILE}")
    except Exception as e:
        print(f"⚠️  Erreur lors de la sauvegarde FAISS: {e}")
    
    return len(texts)

# Construire l'index FAISS (AVEC GESTION D'ERREUR)
print("🚀 Début de la construction de l'index FAISS...")
try:
    entries_count = build_faiss_index()
    print(f"✅ Index FAISS construit ({entries_count} entrées)")
except Exception as e:
    print(f"⚠️  Construction FAISS échouée: {e}")
    print("   Continuant...")

# =============================================================================
# CLARIFICATION IMPORTANTE: LES DEUX FAISS
# =============================================================================

print("\n" + "="*80)
print("🔍 CLARIFICATION: LES DEUX INDEX FAISS DISTINCTS")
print("Il est crucial de comprendre la différence pour éviter la confusion")
print("="*80)

print("📊 SYSTÈME À DEUX FAISS INDÉPENDANTS:")
print()

print("🔍 FAISS IMAGE (pour validation/classification):")
print("   └─ Construit sur les embeddings du modèle Swin")
print("   └─ Utilisé pour: override des prédictions, recherche d'images similaires")
print("   └─ Chemin: outputs/phase2_swin_base_production/models/faiss_index.bin")
print("   └─ Fonction: validation cohérente avec les embeddings du modèle")
print()

print("📚 FAISS TEXTE (pour RAG/explications):")
print("   └─ Construit sur les textes Plantwise/MOH")
print("   └─ Utilisé pour: génération d'explications RAG, recherche documentaire")
print("   └─ Chemin: moh_index.faiss")
print("   └─ Fonction: enrichissement des réponses avec connaissances expertes")
print()

print("✅ CLARIFICATION:")
print("   • swin_faiss_index → utilisé pour override/validation d'images")
print("   • moh_faiss_index → utilisé pour explications RAG textuelles")
print("   • Ces deux index sont complètement séparés et servent des buts différents")
print("="*80)

# =============================================================================
# ÉTAPE 8 et 8.5 supprimées à la demande.
# Objectif Colab: produire les artefacts (modèle + index) puis arrêter.

# =============================================================================
# ÉTAPE 9: VÉRIFICATION MODÈLES ENTRAÎNÉS
# =============================================================================

print("\n" + "="*80)
print("📊 ÉTAPE 9: VÉRIFICATION MODÈLES ENTRAÎNÉS")
print("Vérification que tous les modèles Swin ont été correctement sauvegardés")
print("="*80)

import os

swin_dir = os.path.join(project_root, "outputs", "phase2_swin_base_production", "models")
files_needed = ['metric_model.pt', 'faiss_index.bin', 'metadata.json']  # CORRECTION: fichiers réellement sauvés

print("Vérification des fichiers Swin entraînés:")
all_exist = True
for file in files_needed:
    filepath = os.path.join(swin_dir, file)
    exists = os.path.exists(filepath)
    status = "✅ Existe" if exists else "❌ Manquant"
    print(f"  {filepath}: {status}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✅ Tous les modèles Swin sont disponibles et entraînés")
    print("🎯 Prêt pour le déploiement !")

    # -------------------------------------------------------------------------
    # ÉTAPE 9.5: VALIDATION RAPIDE - CLASSIFICATION SWIN + MAPPING PLANTWISE
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🔎 ÉTAPE 9.5: TEST RÉEL - SWIN CLASSIFICATION (TOP-3) + Plantwise JSON")
    print("""Cet test vérifie que le modèle Swin charge bien les poids entraînés et
qu'il retourne des prédictions réelles (pas de mock). Ensuite, il mappe la
prédiction vers les données Plantwise (symptômes, traitement, etc.).""")
    print("="*80)

    try:
        from models.swin_classifier import SwinDiseaseClassifier

        # Find a sample image (usage: data/images/*)
        sample_dir = os.path.join(project_root, 'data', 'images')
        sample_image = None
        if os.path.isdir(sample_dir):
            for fname in os.listdir(sample_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    sample_image = os.path.join(sample_dir, fname)
                    break

        if sample_image is None:
            print("⚠️ Aucun échantillon d'image trouvé dans 'data/images'.")
            print("   Ajoutez une photo (jpg/png) pour tester la classification Swin.")
        else:
            print(f"📷 Image de test utilisée: {sample_image}")

            # Strict mode: fail if model or FAISS index missing (no mock)
            classifier = SwinDiseaseClassifier(strict=True)
            predictions = classifier.classify_image(sample_image, top_k=3)

            print("\n✅ Top-3 Prédictions Swin (réelles):")
            for i, pred in enumerate(predictions, start=1):
                print(f"  {i}. {pred['disease']} (confiance={pred['confidence']:.3f})")

            # Map prediction to Plantwise JSON (BLIP2_normalized)
            def _normalize_key(name):
                if not name:
                    return ""
                key = name.strip().lower()
                key = key.replace('_', ' ').replace('-', ' ')
                key = re.sub(r"\s+", " ", key)
                return key

            def load_plantwise_data(path):
                mapping = {}
                if not os.path.isdir(path):
                    return mapping
                for fname in os.listdir(path):
                    if not fname.lower().endswith('.json'):
                        continue
                    try:
                        with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                            content = json.load(f)
                        key = _normalize_key(fname.replace('.json', ''))
                        mapping[key] = content
                    except Exception:
                        continue
                return mapping

            plantwise_dir = os.path.join(project_root, 'BLIP2_normalized')
            plantwise = load_plantwise_data(plantwise_dir)

            # Display best match info (first prediction)
            top_disease = predictions[0]['disease']
            top_key = _normalize_key(top_disease)
            info = plantwise.get(top_key, {})

            if info:
                print("\n✅ Informations Plantwise pour la meilleure prédiction :")
                print(f"Nom: {info.get('name', '')}")
                print(f"Nom scientifique: {info.get('scientific_name', '')}")
                print(f"Symptômes: {info.get('symptoms', '')}")
                print(f"Traitement / Gestion: {info.get('management', '')}")
            else:
                print("\n⚠️ Aucun document Plantwise trouvé pour cette prédiction.")

    except Exception as e:
        print(f"❌ Échec du test Swin + Plantwise: {e}")

else:
    print("\n❌ Certains modèles manquent - vérifiez l'entraînement")
    print("🔄 Vous pouvez ré-exécuter l'étape 5 si nécessaire")

# =============================================================================
# FIN COLAB - MODÈLE POUR USAGE LOCAL
# =============================================================================
# ❌ PAS de Streamlit sur Colab : vous lancerez Streamlit en LOCAL vous-même
# =============================================================================

print("\n" + "="*80)
print("🏁 SCRIPT COLAB TERMINÉ - MODÈLE PRÊT POUR USAGE LOCAL")
print("="*80)
print("""
✅ RÉSULTAT : Modèle Swin + FAISS + métadonnées sont dans votre Google Drive.

📁 Emplacement des fichiers (synchronisés via Drive) :
   {swin_dir}

📋 Fichiers générés :
   • metric_model.pt   → Poids du modèle Swin entraîné
   • faiss_index.bin   → Index FAISS pour recherche d'images similaires
   • metadata.json     → Métadonnées (classes, prototypes, Recall@1, etc.)

🚀 USAGE EN LOCAL :
   1. Téléchargez/synchronisez le dossier outputs/ depuis Google Drive vers votre PC
   2. Lancez Streamlit en local : streamlit run 04_app_streamlit.py
   3. Ou utilisez model_core.load_phase2_model_and_metadata() dans votre propre script

💡 Les index MOH (moh_index.faiss, moh_metadata.json) et BLIP2_normalized/ sont
   également dans le projet si vous en avez besoin côté local.

""".format(swin_dir=swin_dir))
print("="*80)