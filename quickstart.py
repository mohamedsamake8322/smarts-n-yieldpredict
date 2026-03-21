#!/usr/bin/env python3
"""
Quick Start Script for Smart Agriculture Application

Runs all setup steps in sequence: normalization, index building, testing, and app launch.
"""

import os
import sys
import subprocess
from config import BASE_PATH, IN_COLAB, print_config

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main setup and execution function."""
    print(f"\n{'🚀'*30}")
    print("SMART AGRICULTURE APPLICATION - QUICK START")
    print(f"{'🚀'*30}")
    
    # Show configuration
    print_config()
    
    # Step 1: Verify project structure
    print(f"\n{'='*60}")
    print("1️⃣ Verifying project structure...")
    print(f"{'='*60}")
    
    required_dirs = ['BLIP2', 'Moh', 'modules', 'pages']
    all_present = True
    
    for dir_name in required_dirs:
        path = os.path.join(BASE_PATH, dir_name)
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_name}/")
        if not exists and dir_name not in ['models']:
            all_present = False
    
    if not all_present:
        print("\n⚠️  Some required directories are missing. Please verify the project structure.")
        return False
    
    # Step 2: Normalize BLIP2 files
    success = run_command(
        [sys.executable, "normalize_blip2.py"],
        "Normalizing BLIP2 JSON files (109 files)"
    )
    if not success:
        print("\n⚠️  Failed to normalize BLIP2 files. Continuing anyway...")
    
    # Step 3: Build FAISS index
    success = run_command(
        [sys.executable, "build_moh_index.py"],
        "Building FAISS index for Plantwise knowledge base (1115 files)"
    )
    if not success:
        print("\n⚠️  Failed to build FAISS index. The assistant may not work.")
    
    # Step 4: Test modules
    success = run_command(
        [sys.executable, "test_modules.py"],
        "Testing modules"
    )
    if not success:
        print("\n⚠️  Module tests failed. Please check the setup.")
    
    # Step 5: Launch Streamlit
    print(f"\n{'='*60}")
    print("5️⃣ Launching Streamlit Application...")
    print(f"{'='*60}")
    print("\n🌐 Starting web server...")
    print("   → Access the app at: http://localhost:8501")
    print("   → Press Ctrl+C to stop the server")
    
    if IN_COLAB:
        # For Colab, use specific configuration
        run_command(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
             "--logger.level=error", "--client.showErrorDetails=false"],
            "Launching App on Google Colab"
        )
    else:
        # For local machine
        run_command(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
            "Launching App"
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
