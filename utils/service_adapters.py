"""
Adapters to make services Streamlit-compatible (synchronous)
"""

import asyncio
from functools import wraps

def sync_to_async(func):
    """Convert an async function to a synchronous one"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # If an event loop is already running, use run_until_complete
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, func(*args, **kwargs))
                return future.result()
        else:
            return loop.run_until_complete(func(*args, **kwargs))
    
    return wrapper

# Wrapper pour DetectionService
class SyncDetectionService:
    """Synchronous version of DetectionService"""
    
    def __init__(self, async_service):
        self.async_service = async_service
    
    def is_ready(self):
        return self.async_service.is_ready()
    
    def detect(self, image_data: bytes, filename: str):
        """Synchronous version of detect"""
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
    """Synchronous version of ChatbotService"""
    
    def __init__(self, async_service):
        self.async_service = async_service
    
    def is_ready(self):
        return self.async_service.is_ready()
    
    def generate_response(self, message: str, context=None, user_history=None):
        """Synchronous version of generate_response"""
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
    """Synchronous version of DatabaseService"""
    
    def __init__(self, async_service):
        self.async_service = async_service
    
    def is_ready(self):
        return self.async_service.is_ready()
    
    def save_detection(self, user_id, image_data, filename, result, location=None):
        """Synchronous version of save_detection"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.save_detection(user_id, image_data, filename, result, location)
            )
        finally:
            loop.close()
    
    def get_user_detections(self, user_id, limit=20):
        """Synchronous version of get_user_detections"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.get_user_detections(user_id, limit)
            )
        finally:
            loop.close()
    
    def get_detection(self, detection_id):
        """Synchronous version of get_detection"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_service.get_detection(detection_id)
            )
        finally:
            loop.close()
    
    def delete_detection(self, detection_id, user_id):
        """Synchronous version of delete_detection"""
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










