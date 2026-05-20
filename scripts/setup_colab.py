#!/usr/bin/env python3
"""
Google Colab Setup Script

Automatically sets up the environment and mounts Google Drive.
Run this in the first cell of your Colab notebook.
"""

import os
import sys
from google.colab import drive

print("🚀 Setting up Smart Agriculture Application for Google Colab...")
print("=" * 60)

# Mount Google Drive
print("\n📁 Mounting Google Drive...")
drive.mount('/content/drive')

# Verify project structure
project_path = "/content/drive/MyDrive/smarts-n-yieldpredict"
print(f"✓ Google Drive mounted at /content/drive")
print(f"✓ Project path: {project_path}")

# Change to project directory
os.chdir(project_path)
sys.path.insert(0, project_path)

print("\n📦 Verifying project structure...")
required_dirs = ['BLIP2', 'BLIP2_normalized', 'Moh', 'modules', 'pages']
for dir_name in required_dirs:
    path = os.path.join(project_path, dir_name)
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"  {exists} {dir_name}/")

print("\n🔧 Installing/upgrading required packages...")
packages_to_install = [
    'sentence-transformers',
    'faiss-cpu',
    'streamlit',
    'torch',
    'torchvision',
    'transformers',
    'pillow',
    'numpy'
]

for package in packages_to_install:
    try:
        __import__(package.replace('-', '_'))
        print(f"  ✓ {package} (already installed)")
    except ImportError:
        print(f"  📥 Installing {package}...")
        os.system(f"pip install {package} -q")
        print(f"  ✓ {package} installed")

print("\n✅ Setup complete!")
print("\n📋 Next steps:")
print("  1. Verify config.py is set to use Colab paths")
print("  2. Run: python normalize_blip2.py")
print("  3. Run: python build_moh_index.py")
print("  4. Run: streamlit run 04_app_streamlit.py")
print("\n" + "=" * 60)
