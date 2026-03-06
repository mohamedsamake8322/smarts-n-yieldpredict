"""
Modèles de données pour le chatbot conversationnel
"""

from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class ChatMessage(BaseModel):
    """Message de conversation"""
    role: str  # user, assistant
    content: str
    timestamp: Optional[str] = None

class ChatResponse(BaseModel):
    """Réponse du chatbot"""
    response: str
    suggestions: List[str]
    context_used: bool
    timestamp: str





