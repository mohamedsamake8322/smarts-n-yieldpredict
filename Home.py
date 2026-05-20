"""
Smart Disease Detection - Main Application
Professional Sènè crop disease diagnostic assistant

This file serves as the main entry point for Streamlit Cloud deployment.
It imports and re-exports the actual application from scripts/Home.py
"""

# Import everything from the actual Home application
import sys
from pathlib import Path

# Add scripts folder to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Import and execute the actual Home.py
exec(open(Path(__file__).parent / "scripts" / "Home.py").read())
