"""
Integration of the Phi-3-mini model for Agro-Scan
"""

import os
import logging
from pathlib import Path
import subprocess
import time
import requests

logger = logging.getLogger(__name__)

class ModelManager:
    """Local Phi-3-mini model manager"""
    
    def __init__(self):
        self.model_path = Path("local_model/Phi-3-mini-4k-instruct-q4.gguf")
        self.server_script = Path("local_model/model_server.py")
        self.server_url = "http://127.0.0.1:5000"
        self.server_process = None
        self.is_running = False
    
    def check_model_exists(self):
        """Check if model file exists"""
        return self.model_path.exists()
    
    def start_server(self):
        """Start the model server"""
        if not self.check_model_exists():
            logger.warning(f"Model not found: {self.model_path}")
            return False
        
        if self.is_server_running():
            logger.info("Model server is already running")
            return True
        
        try:
            # Start the server in the background
            self.server_process = subprocess.Popen(
                ["python", str(self.server_script)],
                cwd=str(self.server_script.parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for the server to start
            for _ in range(10):
                time.sleep(1)
                if self.is_server_running():
                    self.is_running = True
                    logger.info("Model server started successfully")
                    return True
            
            logger.error("Server did not start in time")
            return False
            
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            return False
    
    def stop_server(self):
        """Stop the model server"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.is_running = False
                logger.info("Model server stopped")
            except Exception as e:
                logger.error(f"Error stopping server: {str(e)}")
    
    def is_server_running(self):
        """Check if server is running"""
        try:
            response = requests.get(f"{self.server_url}/docs", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self):
        """Return model info"""
        info = {
            "model_path": str(self.model_path),
            "model_exists": self.check_model_exists(),
            "server_running": self.is_server_running(),
            "server_url": self.server_url
        }
        
        if self.model_path.exists():
            info["model_size_mb"] = round(self.model_path.stat().st_size / (1024 * 1024), 2)
        
        return info

# Instance globale
model_manager = ModelManager()

