"""
Main Agro-Scan application (FastAPI)
Single entry point that loads both models and starts the API
"""

import os
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Agro-Scan")

# ================= CHARGEMENT DES MODÈLES =================

# Modèle 1 : Vision (pour les images)
vision_model = None
try:
    from local_model.vision.vision_model import VisionModel
    vision_model = VisionModel()
    if vision_model.is_ready():
        logger.info("✅ Vision model loaded")
    else:
        logger.warning("⚠️ Vision model running in emulation mode")
except Exception as e:
    logger.warning(f"⚠️ Error loading vision model: {e}")

# Modèle 2 : Langage (Phi-3 pour le chatbot)
chat_model = None
try:
    from local_model.chat.chat_model import ChatModel
    chat_model = ChatModel()
    if chat_model.is_ready():
        logger.info("✅ Language model (Phi-3) loaded")
    else:
        logger.warning("⚠️ Language model running in emulation mode")
except Exception as e:
    logger.warning(f"⚠️ Error loading language model: {e}")

# ================= INITIALISATION FASTAPI =================

app = FastAPI(
    title="Agro-Scan API",
    description="API for agricultural plant and disease detection", 
    version="2.0.0"
)

# CORS
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fichiers statiques
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ================= INITIALISATION DES SERVICES =================

from services.detection_service import DetectionService
from services.chatbot_service import ChatbotService
from services.database_service import DatabaseService
from models.detection_models import DetectionResult
from models.chatbot_models import ChatResponse

# Services avec modèles intégrés
detection_service = DetectionService(vision_model=vision_model)
chatbot_service = ChatbotService(chat_model=chat_model)
database_service = DatabaseService()

# ================= MODÈLES PYDANTIC =================

class DetectionRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL de l'image")
    user_id: Optional[str] = Field(None, description="Identifiant utilisateur")
    location: Optional[dict] = Field(None, description="Localisation géographique")
    text_description: Optional[str] = Field(None, description="Description textuelle optionnelle")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Message utilisateur")
    context: Optional[dict] = Field(None, description="Contexte conversationnel")
    user_id: Optional[str] = Field(None, description="Identifiant utilisateur")

# ================= ROUTES =================

@app.get("/", summary="Welcome")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Welcome to the Agro-Scan API",
        "version": "2.0.0",
        "status": "active",
        "models": {
            "vision": vision_model.is_ready() if vision_model else False,
            "chat": chat_model.is_ready() if chat_model else False
        }
    }

@app.get("/health", summary="System health")
async def health_check():
    """Check API and model status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "detection": detection_service.is_ready(),
            "chatbot": chatbot_service.is_ready(),
            "database": database_service.is_ready()
        },
        "models": {
            "vision": vision_model.is_ready() if vision_model else False,
            "chat": chat_model.is_ready() if chat_model else False
        }
    }

@app.post("/api/detect", response_model=DetectionResult, summary="Disease detection")
async def detect_plant_disease(
    file: UploadFile = File(...),
    user_id: Optional[str] = None,
    location: Optional[str] = None,
    text_description: Optional[str] = None
):
    """
    Endpoint for plant and disease detection
    
    Supporte maintenant l'analyse combinée image + texte
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        image_data = await file.read()
        
        logger.info(f"Détection lancée pour utilisateur={user_id}, fichier={file.filename}")
        
        # Détection avec support texte optionnel
        result = await detection_service.detect(
            image_data=image_data,
            filename=file.filename,
            text_description=text_description,
            user_id=user_id
        )
        
        if user_id:
            await database_service.save_detection(
                user_id=user_id,
                image_data=image_data,
                filename=file.filename,
                result=result,
                location=location
            )
        
        return result
        
    except Exception as e:
        logger.exception("Erreur lors de la détection")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/chat", response_model=ChatResponse, summary="Chat avec l'assistant")
async def chat_with_assistant(request: ChatRequest):
    """
    Endpoint pour l'assistant conversationnel
    
    Utilise le modèle Phi-3 local pour générer les réponses
    """
    try:
        logger.info(f"Message reçu: {request.message[:50]}...")
        
        user_history = None
        if request.user_id:
            user_history = await database_service.get_user_chat_history(request.user_id)
        
        response = await chatbot_service.generate_response(
            message=request.message,
            context=request.context,
            user_history=user_history
        )
        
        if request.user_id:
            await database_service.save_chat_message(
                user_id=request.user_id,
                message=request.message,
                response=response.response,
                context=request.context
            )
        
        return response
        
    except Exception as e:
        logger.exception("Erreur lors du chat")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/history/{user_id}", summary="Historique utilisateur")
async def get_user_history(user_id: str, limit: int = 20):
    """Récupération de l'historique des détections et conversations"""
    try:
        detections = await database_service.get_user_detections(user_id, limit)
        chats = await database_service.get_user_chat_history(user_id, limit)
        
        return {
            "detections": detections,
            "chats": chats,
            "total_detections": len(detections),
            "total_chats": len(chats)
        }
    except Exception as e:
        logger.exception("Erreur lors de la récupération de l'historique")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/detection/{detection_id}", summary="Détails d'une détection")
async def get_detection(detection_id: str):
    """Récupération d'une détection spécifique"""
    try:
        detection = await database_service.get_detection(detection_id)
        if not detection:
            raise HTTPException(status_code=404, detail="Détection non trouvée")
        return detection
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur lors de la récupération de la détection")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/api/detection/{detection_id}", summary="Supprimer une détection")
async def delete_detection(detection_id: str, user_id: str):
    """Suppression d'une détection"""
    try:
        success = await database_service.delete_detection(detection_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Détection non trouvée ou non autorisée")
        return {"message": "Détection supprimée avec succès"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur lors de la suppression")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/plants", summary="Liste des plantes")
async def get_plants_list(search: Optional[str] = None):
    """Liste des plantes supportées avec possibilité de recherche"""
    try:
        plants = await database_service.get_plants_list(search)
        return {"plants": plants, "count": len(plants)}
    except Exception as e:
        logger.exception("Erreur lors de la récupération des plantes")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/stats/{user_id}", summary="User statistics")
async def get_user_stats(user_id: str):
    """User statistics"""
    try:
        stats = await database_service.get_user_stats(user_id)
        return stats
    except Exception as e:
        logger.exception("Erreur lors de la récupération des stats")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ================= MAIN =================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("🚀 Démarrage de Agro-Scan API...")
    logger.info(f"📸 Modèle Vision: {'✅' if vision_model and vision_model.is_ready() else '⚠️ Simulation'}")
    logger.info(f"💬 Modèle Chat: {'✅' if chat_model and chat_model.is_ready() else '⚠️ Simulation'}")
    logger.info(f"🌐 Serveur: http://{host}:{port}")
    logger.info(f"📚 Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "app_fastapi:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )







