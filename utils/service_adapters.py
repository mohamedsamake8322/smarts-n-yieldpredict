"""
Adaptateurs pour rendre les services compatibles avec Streamlit (synchrone)
"""

import asyncio
from functools import wraps

def sync_to_async(func):
    """Convertit une fonction async en fonction synchrone"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # Si une boucle est déjà en cours, utiliser run_until_complete
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, func(*args, **kwargs))
                return future.result()
        else:
            return loop.run_until_complete(func(*args, **kwargs))
    
    return wrapper

# Wrapper pour DetectionService
class SyncDetectionService:
    """Version synchrone de DetectionService"""
    
    def __init__(self, async_service):
        self.async_service = async_service
    
    def is_ready(self):
        return self.async_service.is_ready()
    
    def detect(self, image_data: bytes, filename: str):
        """Version synchrone de detect"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.detect(image_data, filename)
            )
        finally:
            loop.close()

# Wrapper pour ChatbotService
class SyncChatbotService:
    """Version synchrone de ChatbotService"""
    
    def __init__(self, async_service):
        self.async_service = async_service
    
    def is_ready(self):
        return self.async_service.is_ready()
    
    def generate_response(self, message: str, context=None, user_history=None):
        """Version synchrone de generate_response"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.generate_response(message, context, user_history)
            )
        finally:
            loop.close()

# Wrapper pour DatabaseService
class SyncDatabaseService:
    """Version synchrone de DatabaseService"""
    
    def __init__(self, async_service):
        self.async_service = async_service
    
    def is_ready(self):
        return self.async_service.is_ready()
    
    def save_detection(self, user_id, image_data, filename, result, location=None):
        """Version synchrone de save_detection"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.save_detection(user_id, image_data, filename, result, location)
            )
        finally:
            loop.close()
    
    def get_user_detections(self, user_id, limit=20):
        """Version synchrone de get_user_detections"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.get_user_detections(user_id, limit)
            )
        finally:
            loop.close()
    
    def get_detection(self, detection_id):
        """Version synchrone de get_detection"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.get_detection(detection_id)
            )
        finally:
            loop.close()
    
    def delete_detection(self, detection_id, user_id):
        """Version synchrone de delete_detection"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.delete_detection(detection_id, user_id)
            )
        finally:
            loop.close()
    
    def save_chat_message(self, user_id, message, response, context=None):
        """Version synchrone de save_chat_message"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.save_chat_message(user_id, message, response, context)
            )
        finally:
            loop.close()
    
    def get_user_chat_history(self, user_id, limit=20):
        """Version synchrone de get_user_chat_history"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.get_user_chat_history(user_id, limit)
            )
        finally:
            loop.close()
    
    def get_user_stats(self, user_id):
        """Version synchrone de get_user_stats"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.get_user_stats(user_id)
            )
        finally:
            loop.close()










