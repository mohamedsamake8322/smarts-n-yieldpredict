"""
Storage utilities for managing diagnosis history, feedback, and statistics
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3

# Storage paths
STORAGE_DIR = Path("data/app_storage")
HISTORY_FILE = STORAGE_DIR / "history.json"
FEEDBACK_FILE = STORAGE_DIR / "feedback.json"
DB_FILE = STORAGE_DIR / "app.db"
DATASET_REVIEW_DIR = STORAGE_DIR / "dataset_to_review"

# Ensure directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

def ensure_db():
    """Initialize SQLite database if needed"""
    if not DB_FILE.exists():
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnostic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                image_name TEXT,
                disease TEXT,
                confidence REAL,
                date TEXT,
                user_feedback TEXT,
                timestamp INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_scans INTEGER,
                correct INTEGER,
                incorrect INTEGER,
                unknown INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                country TEXT,
                total_scans INTEGER,
                contributions INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()

def save_diagnosis(image_name: str, disease: str, confidence: float, image_hash: str = None) -> Dict:
    """
    Save a diagnosis to history
    """
    ensure_db()
    
    diagnosis_record = {
        "image_name": image_name,
        "disease": disease,
        "confidence": float(confidence),
        "date": datetime.now().isoformat(),
        "user_feedback": None,
    }
    
    # Save to JSON for quick access
    history = load_history()
    history.append(diagnosis_record)
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    # Also save to DB
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO diagnostic_history 
            (image_name, disease, confidence, date, user_feedback, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_name, disease, confidence, datetime.now().isoformat(), None, int(datetime.now().timestamp())))
        
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Duplicate
    finally:
        conn.close()
    
    return diagnosis_record

def save_feedback(disease: str, feedback: str, confidence: float = None):
    """
    Save user feedback on a diagnosis
    """
    feedback_record = {
        "disease": disease,
        "feedback": feedback,  # "correct", "incorrect", "unsure"
        "confidence": confidence,
        "date": datetime.now().isoformat(),
    }
    
    # Save to JSON
    feedbacks = load_feedback()
    feedbacks.append(feedback_record)
    
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedbacks, f, indent=2, ensure_ascii=False)
    
    # Update DB
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE diagnostic_history 
        SET user_feedback = ?
        WHERE disease = ? AND date IN (
            SELECT date FROM diagnostic_history 
            WHERE disease = ? AND user_feedback IS NULL
            ORDER BY date DESC LIMIT 1
        )
    ''', (feedback, disease, disease))
    
    conn.commit()
    conn.close()

def load_history() -> List[Dict]:
    """Load diagnosis history from JSON"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_feedback() -> List[Dict]:
    """Load feedback from JSON"""
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def get_statistics() -> Dict[str, Any]:
    """
    Calculate statistics from feedback
    """
    feedbacks = load_feedback()
    history = load_history()
    
    total_scans = len(history)
    correct = sum(1 for f in feedbacks if f["feedback"] == "correct")
    incorrect = sum(1 for f in feedbacks if f["feedback"] == "incorrect")
    unsure = sum(1 for f in feedbacks if f["feedback"] == "unsure")
    
    # Count diseases
    disease_counts = {}
    for h in history:
        disease = h["disease"]
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    # Sort by count
    top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    accuracy = (correct / len(feedbacks) * 100) if feedbacks else 0
    
    return {
        "total_scans": total_scans,
        "correct": correct,
        "incorrect": incorrect,
        "unsure": unsure,
        "accuracy": accuracy,
        "top_diseases": top_diseases,
        "disease_counts": disease_counts,
    }

def get_history_dataframe():
    """Get history as pandas dataframe for display"""
    try:
        import pandas as pd
        history = load_history()
        if not history:
            return pd.DataFrame()
        return pd.DataFrame(history)
    except ImportError:
        return None

def save_wrong_image_for_review(image_bytes: bytes, disease: str, confidence: float):
    """
    Save incorrectly diagnosed image for dataset review
    """
    filename = f"{disease}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = DATASET_REVIEW_DIR / filename
    
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    # Log metadata
    metadata = {
        "filename": filename,
        "disease": disease,
        "confidence": confidence,
        "date": datetime.now().isoformat(),
    }
    
    metadata_file = DATASET_REVIEW_DIR / f"{filename}.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def create_or_update_user(username: str, country: str = None):
    """Create or update user profile"""
    ensure_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (username, country, total_scans, contributions)
            VALUES (?, ?, ?, ?)
        ''', (username, country, 0, 0))
    except sqlite3.IntegrityError:
        if country:
            cursor.execute('''
                UPDATE users 
                SET country = ?
                WHERE username = ?
            ''', (country, username))
    
    conn.commit()
    conn.close()

def get_user_stats(username: str) -> Dict:
    """Get user statistics"""
    ensure_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT total_scans, contributions FROM users WHERE username = ?
    ''', (username,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "username": username,
            "total_scans": result[0],
            "contributions": result[1],
        }
    return {"username": username, "total_scans": 0, "contributions": 0}

def clear_all_data():
    """Clear all stored data (for development/testing)"""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
    if FEEDBACK_FILE.exists():
        FEEDBACK_FILE.unlink()
    if DB_FILE.exists():
        DB_FILE.unlink()
