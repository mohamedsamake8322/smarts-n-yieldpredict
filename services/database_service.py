"""
Service de gestion de la base de données
Gère le stockage des détections, conversations et données utilisateur
"""

import logging
import os
import json
from typing import Optional, List, Dict
from datetime import datetime
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service de gestion de la base de données"""
    
    def __init__(self):
        self.db_path = os.getenv("DATABASE_PATH", "data/agro_scan.db")
        self.images_dir = os.getenv("IMAGES_DIR", "data/images")
        self._ensure_directories()
        self._initialize_database()
    
    def _ensure_directories(self):
        """Crée les répertoires nécessaires"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.images_dir).mkdir(parents=True, exist_ok=True)
    
    def _initialize_database(self):
        """Initialise la base de données SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table des détections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    filename TEXT,
                    image_path TEXT,
                    plant_name TEXT,
                    plant_scientific_name TEXT,
                    diseases TEXT,
                    deficiencies TEXT,
                    recommendations TEXT,
                    severity TEXT,
                    confidence REAL,
                    location TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table des conversations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    message TEXT,
                    response TEXT,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table des utilisateurs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP
                )
            """)
            
            # Table des plantes (référentiel)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    scientific_name TEXT,
                    category TEXT,
                    description TEXT,
                    common_diseases TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index pour améliorer les performances
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_user ON detections(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_created ON detections(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_messages(created_at)")
            
            conn.commit()
            conn.close()
            
            # Initialisation des données de référence
            self._initialize_reference_data()
            
            logger.info("Base de données initialisée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {str(e)}")
            raise
    
    def _initialize_reference_data(self):
        """Initialise les données de référence (plantes communes)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Vérifier si des données existent déjà
        cursor.execute("SELECT COUNT(*) FROM plants")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
        
        # Plantes communes africaines et tropicales
        plants_data = [
            ("Tomate", "Solanum lycopersicum", "Légume", "Plante potagère très cultivée", "Mildiou, Oïdium, Alternariose"),
            ("Maïs", "Zea mays", "Céréale", "Céréale de base en Afrique", "Rouille, Charbon, Helminthosporiose"),
            ("Riz", "Oryza sativa", "Céréale", "Céréale principale", "Pyriculariose, Helminthosporiose"),
            ("Manioc", "Manihot esculenta", "Racine", "Plante à racines tubéreuses", "Mosaïque, Bactériose"),
            ("Banane", "Musa acuminata", "Fruit", "Fruitier tropical", "Sigatoka, Maladie de Panama"),
            ("Cacao", "Theobroma cacao", "Arbre", "Arbre à cacao", "Pourriture brune, Moniliose"),
            ("Café", "Coffea arabica", "Arbre", "Arbre à café", "Rouille, Anthracnose"),
            ("Arachide", "Arachis hypogaea", "Légumineuse", "Légumineuse à graines", "Taches foliaires, Pourriture"),
            ("Haricot", "Phaseolus vulgaris", "Légumineuse", "Légumineuse comestible", "Anthracnose, Rouille"),
            ("Piment", "Capsicum annuum", "Légume", "Plante à épices", "Mildiou, Virus"),
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO plants (name, scientific_name, category, description, common_diseases)
            VALUES (?, ?, ?, ?, ?)
        """, plants_data)
        
        conn.commit()
        conn.close()
    
    def is_ready(self) -> bool:
        """Vérifie si le service est prêt"""
        return os.path.exists(self.db_path)
    
    async def save_detection(
        self,
        user_id: str,
        image_data: bytes,
        filename: str,
        result,
        location: Optional[str] = None
    ):
        """Sauvegarde une détection dans la base de données"""
        try:
            import uuid
            detection_id = str(uuid.uuid4())
            
            # Sauvegarde de l'image
            image_filename = f"{detection_id}_{filename}"
            image_path = os.path.join(self.images_dir, image_filename)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Préparation des données
            diseases_json = json.dumps([d.dict() for d in result.diseases])
            deficiencies_json = json.dumps(result.deficiencies)
            recommendations_json = json.dumps([r.dict() for r in result.recommendations])
            
            # Insertion dans la base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO detections (
                    id, user_id, filename, image_path, plant_name, plant_scientific_name,
                    diseases, deficiencies, recommendations, severity, confidence, location
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                detection_id,
                user_id,
                filename,
                image_path,
                result.plant_info.name,
                result.plant_info.scientific_name,
                diseases_json,
                deficiencies_json,
                recommendations_json,
                result.overall_severity,
                result.confidence_score,
                location
            ))
            
            # Mise à jour de l'utilisateur
            cursor.execute("""
                INSERT OR REPLACE INTO users (user_id, last_active)
                VALUES (?, CURRENT_TIMESTAMP)
            """, (user_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Détection sauvegardée: {detection_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la détection: {str(e)}")
            raise
    
    async def get_user_detections(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Récupère les détections d'un utilisateur"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM detections
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            detections = [dict(row) for row in rows]
            
            # Désérialisation des JSON
            for detection in detections:
                detection['diseases'] = json.loads(detection['diseases']) if detection['diseases'] else []
                detection['deficiencies'] = json.loads(detection['deficiencies']) if detection['deficiencies'] else []
                detection['recommendations'] = json.loads(detection['recommendations']) if detection['recommendations'] else []
            
            conn.close()
            return detections
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des détections: {str(e)}")
            return []
    
    async def get_detection(self, detection_id: str) -> Optional[Dict]:
        """Récupère une détection spécifique"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            detection = dict(row)
            detection['diseases'] = json.loads(detection['diseases']) if detection['diseases'] else []
            detection['deficiencies'] = json.loads(detection['deficiencies']) if detection['deficiencies'] else []
            detection['recommendations'] = json.loads(detection['recommendations']) if detection['recommendations'] else []
            
            conn.close()
            return detection
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération: {str(e)}")
            return None
    
    async def delete_detection(self, detection_id: str, user_id: str) -> bool:
        """Supprime une détection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Vérifier que la détection appartient à l'utilisateur
            cursor.execute("SELECT image_path FROM detections WHERE id = ? AND user_id = ?", (detection_id, user_id))
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return False
            
            # Supprimer l'image
            image_path = row[0]
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # Supprimer de la base de données
            cursor.execute("DELETE FROM detections WHERE id = ? AND user_id = ?", (detection_id, user_id))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {str(e)}")
            return False
    
    async def save_chat_message(
        self,
        user_id: str,
        message: str,
        response: str,
        context: Optional[Dict] = None
    ):
        """Sauvegarde un message de conversation"""
        try:
            import uuid
            message_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_messages (id, user_id, message, response, context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                message_id,
                user_id,
                message,
                response,
                json.dumps(context) if context else None
            ))
            
            # Mise à jour de l'utilisateur
            cursor.execute("""
                INSERT OR REPLACE INTO users (user_id, last_active)
                VALUES (?, CURRENT_TIMESTAMP)
            """, (user_id,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message: {str(e)}")
    
    async def get_user_chat_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Récupère l'historique des conversations d'un utilisateur"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM chat_messages
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            messages = []
            
            for row in rows:
                msg = dict(row)
                if msg['context']:
                    msg['context'] = json.loads(msg['context'])
                messages.append(msg)
            
            conn.close()
            return messages
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique: {str(e)}")
            return []
    
    async def get_plants_list(self, search: Optional[str] = None) -> List[Dict]:
        """Récupère la liste des plantes avec recherche optionnelle"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if search:
                cursor.execute("""
                    SELECT * FROM plants
                    WHERE name LIKE ? OR scientific_name LIKE ? OR description LIKE ?
                    ORDER BY name
                """, (f"%{search}%", f"%{search}%", f"%{search}%"))
            else:
                cursor.execute("SELECT * FROM plants ORDER BY name")
            
            rows = cursor.fetchall()
            plants = [dict(row) for row in rows]
            
            conn.close()
            return plants
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des plantes: {str(e)}")
            return []
    
    async def get_user_stats(self, user_id: str) -> Dict:
        """Récupère les statistiques d'un utilisateur"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Nombre total de détections
            cursor.execute("SELECT COUNT(*) FROM detections WHERE user_id = ?", (user_id,))
            total_detections = cursor.fetchone()[0]
            
            # Nombre de conversations
            cursor.execute("SELECT COUNT(*) FROM chat_messages WHERE user_id = ?", (user_id,))
            total_chats = cursor.fetchone()[0]
            
            # Maladies les plus fréquentes
            cursor.execute("""
                SELECT diseases FROM detections
                WHERE user_id = ? AND diseases IS NOT NULL
            """, (user_id,))
            
            all_diseases = []
            for row in cursor.fetchall():
                diseases = json.loads(row[0])
                all_diseases.extend([d.get('name') for d in diseases if isinstance(d, dict)])
            
            # Comptage des maladies
            from collections import Counter
            disease_counts = Counter(all_diseases)
            top_diseases = dict(disease_counts.most_common(5))
            
            # Plantes les plus détectées
            cursor.execute("""
                SELECT plant_name, COUNT(*) as count
                FROM detections
                WHERE user_id = ?
                GROUP BY plant_name
                ORDER BY count DESC
                LIMIT 5
            """, (user_id,))
            
            top_plants = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                "total_detections": total_detections,
                "total_chats": total_chats,
                "top_diseases": top_diseases,
                "top_plants": top_plants
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {str(e)}")
            return {
                "total_detections": 0,
                "total_chats": 0,
                "top_diseases": {},
                "top_plants": {}
            }





