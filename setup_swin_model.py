#!/usr/bin/env python3
"""
Setup script for Swin Transformer model files.

This script helps download or copy the trained Swin model files
from Google Drive or other locations to the local environment.
"""

import os
import requests
import shutil
from config import SWIN_MODEL_DIR, SWIN_CHECKPOINT_DIR, ensure_directories, IN_COLAB

def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive.

    Args:
        file_id: Google Drive file ID
        destination: Local destination path
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    """Get confirmation token for large files."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save response content to file."""
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def setup_swin_model():
    """Setup Swin model files."""
    print("🔧 Setting up Swin Transformer model files...")

    # Ensure directories exist
    ensure_directories()

    # Model files to download/copy
    model_files = {
        'metric_model.pt': None,  # Will be set based on environment
        'faiss_index.bin': None,
        'metadata.json': None,
        'metadata.pkl': None
    }

    if IN_COLAB:
        print("Running in Google Colab - model files should already be available")
        # In Colab, files are already at the expected paths
        return True

    # For local environment, we need to download or copy files
    print("Running locally - checking for model files...")

    # Check if files already exist
    all_exist = all(os.path.exists(os.path.join(SWIN_MODEL_DIR, filename))
                   for filename in model_files.keys())

    if all_exist:
        print("✅ All model files already exist locally")
        return True

    print("⚠️  Some model files missing. Please:")
    print("1. Download the trained model files from your Google Drive")
    print("2. Copy them to:", SWIN_MODEL_DIR)
    print("Required files:")
    for filename in model_files.keys():
        print(f"  - {filename}")

    print("\nAlternatively, you can manually specify Google Drive file IDs:")
    print("Edit this script and add the file IDs to the model_files dictionary")

    # Example of how to add Google Drive IDs (user needs to fill these)
    # model_files = {
    #     'metric_model.pt': 'YOUR_FILE_ID_HERE',
    #     'faiss_index.bin': 'YOUR_FILE_ID_HERE',
    #     'metadata.json': 'YOUR_FILE_ID_HERE',
    #     'metadata.pkl': 'YOUR_FILE_ID_HERE'
    # }

    return False

def copy_from_colab_to_local():
    """
    Helper function to copy model files from Colab paths to local paths.
    Run this in Colab to prepare files for local use.
    """
    if not IN_COLAB:
        print("This function should be run in Google Colab")
        return

    colab_model_dir = "/content/drive/MyDrive/outputs/phase2_swin_base_production/models"
    local_model_dir = "/content/smarts-n-yieldpredict/models/swin_base_production"

    os.makedirs(local_model_dir, exist_ok=True)

    files_to_copy = [
        'metric_model.pt',
        'faiss_index.bin',
        'metadata.json',
        'metadata.pkl'
    ]

    for filename in files_to_copy:
        src = os.path.join(colab_model_dir, filename)
        dst = os.path.join(local_model_dir, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Copied {filename}")
        else:
            print(f"⚠️  {filename} not found in Colab")

    print(f"\nFiles copied to: {local_model_dir}")
    print("Download the entire 'models' folder and place it in your local project")

if __name__ == "__main__":
    if IN_COLAB:
        copy_from_colab_to_local()
    else:
        setup_swin_model()