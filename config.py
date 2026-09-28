"""
Configuration for the Smart Agriculture Application
"""

import os

DEBUG = False
ENV = "production"

IN_COLAB = 'COLAB_RELEASE_TAG' in os.environ

def find_project_root():
    if IN_COLAB:
        drive_root = '/content/drive/MyDrive'
        if os.path.exists(drive_root):
            try:
                for name in os.listdir(drive_root):
                    candidate = os.path.join(drive_root, name)
                    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'config.py')):
                        return candidate
                    for sub in os.listdir(candidate):
                        subpath = os.path.join(candidate, sub)
                        if os.path.isdir(subpath) and os.path.exists(os.path.join(subpath, 'config.py')):
                            return subpath
            except Exception:
                pass
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = find_project_root()

BLIP2_NORMALIZED_DIR = os.path.join(BASE_PATH, 'BLIP2_normalized')
BLIP2_I18N_DIR = os.path.join(BASE_PATH, 'BLIP2_i18n')
MOH_DIR = os.path.join(BASE_PATH, 'Moh')
MODELS_DIR = os.path.join(BASE_PATH, 'models')
MOH_INDEX_FILE = os.path.join(BASE_PATH, 'moh_index.faiss')
MOH_METADATA_FILE = os.path.join(BASE_PATH, 'moh_metadata.json')
BLIP2_DIR = os.path.join(BASE_PATH, 'BLIP2')

# Modèle DINOv2 — hébergé sur Hugging Face, pas de chemin local
DINOV2_MODEL_REPO = "mohamedsamake8322/maladie-plantes-dinov2"
DINOV2_CKPT_FILENAME = "dinov2_v3_infer.pt"
DINOV2_IDX_TO_CLASS_FILENAME = "idx_to_class.json"
DINOV2_IMAGE_SIZE = 518

def ensure_directories():
    dirs = [BLIP2_NORMALIZED_DIR, BLIP2_I18N_DIR, MOH_DIR, MODELS_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def print_config():
    print(f"Running in: {'Google Colab' if IN_COLAB else 'Local environment'}")
    print(f"Base path: {BASE_PATH}")
    print(f"DINOv2 repo: {DINOV2_MODEL_REPO}")