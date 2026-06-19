"""
Configuration for the Smart Agriculture Application

Handles environment detection (local vs Google Colab) and path management.
"""

import os

# Existing configuration
DEBUG = False
ENV = "production"

# ============= Path Configuration =============

# Detect environment
IN_COLAB = 'COLAB_RELEASE_TAG' in os.environ

# Base path configuration - DYNAMIC DISCOVERY
def find_project_root():
    """Find the project root by searching for config.py"""
    if IN_COLAB:
        # In Colab, search for project containing config.py
        drive_root = '/content/drive/MyDrive'
        if os.path.exists(drive_root):
            try:
                for name in os.listdir(drive_root):
                    candidate = os.path.join(drive_root, name)
                    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'config.py')):
                        return candidate
                    # Check one level deeper
                    for sub in os.listdir(candidate):
                        subpath = os.path.join(candidate, sub)
                        if os.path.isdir(subpath) and os.path.exists(os.path.join(subpath, 'config.py')):
                            return subpath
            except:
                pass
    
    # Local or fallback: use directory containing config.py
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = find_project_root()

# Directory paths
BLIP2_NORMALIZED_DIR = os.path.join(BASE_PATH, 'BLIP2_normalized')
BLIP2_I18N_DIR = os.path.join(BASE_PATH, 'BLIP2_i18n')
MOH_DIR = os.path.join(BASE_PATH, 'Moh')
MODELS_DIR = os.path.join(BASE_PATH, 'models')
MOH_INDEX_FILE = os.path.join(BASE_PATH, 'moh_index.faiss')
MOH_METADATA_FILE = os.path.join(BASE_PATH, 'moh_metadata.json')

# BLIP2 original directory (for backward compatibility)
BLIP2_DIR = os.path.join(BASE_PATH, 'BLIP2')

# Model paths - Use relative paths for both Colab and local
SWIN_MODEL_DIR = os.path.join(BASE_PATH, 'outputs', 'phase2_swin_base_production', 'models')
SWIN_CHECKPOINT_DIR = os.path.join(BASE_PATH, 'outputs', 'phase2_swin_base_production', 'checkpoints')

SWIN_MODEL_PATH = os.path.join(SWIN_MODEL_DIR, 'senedisease_macro_f1.pt')
SWIN_FAISS_INDEX = os.path.join(SWIN_MODEL_DIR, 'faiss_index.bin')
SWIN_METADATA = os.path.join(SWIN_MODEL_DIR, 'metadata.json')
SWIN_METADATA_FULL = os.path.join(SWIN_MODEL_DIR, 'metadata.pkl')

# Ensure directories exist
def ensure_directories():
    """Ensure all required directories exist."""
    dirs = [BLIP2_NORMALIZED_DIR, BLIP2_I18N_DIR, MOH_DIR, MODELS_DIR, SWIN_MODEL_DIR, SWIN_CHECKPOINT_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# Print configuration info
def print_config():
    """Print the current configuration."""
    print(f"Running in: {'Google Colab' if IN_COLAB else 'Local environment'}")
    print(f"Base path: {BASE_PATH}")
    print(f"BLIP2 normalized: {BLIP2_NORMALIZED_DIR}")
    print(f"Moh directory: {MOH_DIR}")
    print(f"Swin model dir: {SWIN_MODEL_DIR}")
    print(f"Swin model path: {SWIN_MODEL_PATH}")
    print(f"Swin FAISS index: {SWIN_FAISS_INDEX}")
    print(f"FAISS index: {MOH_INDEX_FILE}")

