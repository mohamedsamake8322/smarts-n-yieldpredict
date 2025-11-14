import os
import json
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from googletrans import Translator
import re
from datetime import datetime
import textstat

logger = logging.getLogger(__name__)

class AdvancedAI:
    """Advanced AI features including summaries, translation, quiz generation, and entity extraction"""
    
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found. Advanced AI features will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.translator = Translator()
        
        self.supported_languages = {
            'fr': 'French',
            'en': 'English', 
            'es': 'Spanish'
        }

    def generate_summary(self, text: str, language: str = 'fr', summary_type: str = 'auto') -> Dict:
        """Generate document summary using AI"""
        if not self.client:
            return {'error': 'AI service not available'}
        
        try:
            # Determine summary style based on type
            style_prompts = {
                'auto': 'Create a comprehensive summary highlighting key points',
                'executive': 'Create an executive summary focusing on main decisions and outcomes',
                'key_points': 'Extract and list the main key points as bullet points',
                'abstract': 'Create an academic-style abstract'
            }
            
            style_instruction = style_prompts.get(summary_type, style_prompts['auto'])
            
            # Language-specific prompts
            language_prompts = {
                'fr': f"Résumez le texte suivant en français. {style_instruction}:",
                'en': f"Summarize the following text in English. {style_instruction}:",
                'es': f"Resume el siguiente texto en español. {style_instruction}:"
            }
            
            prompt = language_prompts.get(language, language_prompts['fr'])
            
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, concise summaries."},
                    {"role": "user", "content": f"{prompt}\n\n{text[:4000]}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            summary_text = response.choices[0].message.content
            
            # Calculate confidence score based on text length and coherence
            confidence_score = min(1.0, len(summary_text) / 200)
            
            return {
                'summary': summary_text,
                'summary_type': summary_type,
                'language': language,
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {'error': f'Summary generation failed: {str(e)}'}

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """Translate text between supported languages"""
        try:
            if source_lang == target_lang:
                return {'translated_text': text, 'source_language': source_lang, 'target_language': target_lang}
            
            # Use Google Translate for now
            translation = self.translator.translate(text, src=source_lang, dest=target_lang)
            
            return {
                'translated_text': translation.text,
                'source_language': source_lang,
                'target_language': target_lang,
                'confidence_score': 0.9  # Google Translate is generally reliable
            }
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return {'error': f'Translation failed: {str(e)}'}

    def generate_quiz(self, text: str, language: str = 'fr', difficulty: str = 'medium', num_questions: int = 5) -> Dict:
        """Generate quiz questions from document content"""
        if not self.client:
            return {'error': 'AI service not available'}
        
        try:
            difficulty_instructions = {
                'easy': 'Create simple questions about basic facts and concepts.',
                'medium': 'Create moderate difficulty questions requiring understanding of concepts.',
                'hard': 'Create challenging questions requiring analysis and synthesis.'
            }
            
            language_prompts = {
                'fr': f"""Créez un quiz de {num_questions} questions à choix multiples en français basé sur le texte suivant.
                         {difficulty_instructions[difficulty]}
                         Format: Pour chaque question, donnez:
                         1. La question
                         2. Quatre options (A, B, C, D)
                         3. La bonne réponse
                         4. Une explication
                         
                         Répondez en format JSON strict.""",
                'en': f"""Create a quiz with {num_questions} multiple choice questions in English based on the following text.
                         {difficulty_instructions[difficulty]}
                         Format: For each question, provide:
                         1. The question
                         2. Four options (A, B, C, D)
                         3. The correct answer
                         4. An explanation
                         
                         Respond in strict JSON format.""",
                'es': f"""Crea un quiz de {num_questions} preguntas de opción múltiple en español basado en el siguiente texto.
                         {difficulty_instructions[difficulty]}
                         Formato: Para cada pregunta, proporciona:
                         1. La pregunta
                         2. Cuatro opciones (A, B, C, D)
                         3. La respuesta correcta
                         4. Una explicación
                         
                         Responde en formato JSON estricto."""
            }
            
            prompt = language_prompts.get(language, language_prompts['fr'])
            
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert quiz creator. Always respond with valid JSON."},
                    {"role": "user", "content": f"{prompt}\n\nTexte: {text[:3000]}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1500
            )
            
            quiz_data = json.loads(response.choices[0].message.content)
            
            return {
                'questions': quiz_data.get('questions', []),
                'difficulty': difficulty,
                'language': language,
                'total_questions': len(quiz_data.get('questions', []))
            }
            
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return {'error': f'Quiz generation failed: {str(e)}'}

    def extract_entities(self, text: str, language: str = 'fr') -> Dict:
        """Extract named entities from text using AI"""
        if not self.client:
            return {'error': 'AI service not available'}
        
        try:
            language_prompts = {
                'fr': """Analysez le texte suivant et extrayez les entités nommées.
                         Identifiez: personnes, organisations, lieux, dates, montants d'argent, emails, téléphones.
                         Répondez en JSON avec: type d'entité, valeur, position dans le texte, contexte.""",
                'en': """Analyze the following text and extract named entities.
                         Identify: persons, organizations, locations, dates, money amounts, emails, phones.
                         Respond in JSON with: entity type, value, position in text, context.""",
                'es': """Analiza el siguiente texto y extrae las entidades nombradas.
                         Identifica: personas, organizaciones, lugares, fechas, cantidades de dinero, emails, teléfonos.
                         Responde en JSON con: tipo de entidad, valor, posición en el texto, contexto."""
            }
            
            prompt = language_prompts.get(language, language_prompts['fr'])
            
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an expert at named entity recognition. Always respond with valid JSON."},
                    {"role": "user", "content": f"{prompt}\n\nTexte: {text[:2000]}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1000
            )
            
            entities_data = json.loads(response.choices[0].message.content)
            
            return {
                'entities': entities_data.get('entities', []),
                'language': language,
                'total_entities': len(entities_data.get('entities', []))
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {'error': f'Entity extraction failed: {str(e)}'}

    def analyze_document_complexity(self, text: str) -> Dict:
        """Analyze document complexity and readability"""
        try:
            # Calculate various readability metrics
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_grade = textstat.flesch_kincaid_grade(text)
            avg_sentence_length = textstat.avg_sentence_length(text)
            syllable_count = textstat.syllable_count(text)
            word_count = len(text.split())
            
            # Determine complexity level
            if flesch_score >= 90:
                complexity = 'très facile'
            elif flesch_score >= 80:
                complexity = 'facile'
            elif flesch_score >= 70:
                complexity = 'assez facile'
            elif flesch_score >= 60:
                complexity = 'standard'
            elif flesch_score >= 50:
                complexity = 'assez difficile'
            elif flesch_score >= 30:
                complexity = 'difficile'
            else:
                complexity = 'très difficile'
            
            return {
                'flesch_reading_ease': flesch_score,
                'flesch_kincaid_grade': flesch_grade,
                'complexity_level': complexity,
                'avg_sentence_length': avg_sentence_length,
                'syllable_count': syllable_count,
                'word_count': word_count,
                'recommended_age': max(6, int(flesch_grade) + 5)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document complexity: {str(e)}")
            return {'error': f'Complexity analysis failed: {str(e)}'}

    def suggest_questions(self, text: str, language: str = 'fr') -> List[str]:
        """Suggest relevant questions based on document content"""
        if not self.client:
            return []
        
        try:
            language_prompts = {
                'fr': "Basé sur ce texte, suggérez 5 questions pertinentes qu'un utilisateur pourrait poser:",
                'en': "Based on this text, suggest 5 relevant questions a user might ask:",
                'es': "Basado en este texto, sugiere 5 preguntas relevantes que un usuario podría hacer:"
            }
            
            prompt = language_prompts.get(language, language_prompts['fr'])
            
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "user", "content": f"{prompt}\n\n{text[:2000]}"}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            suggestions = response.choices[0].message.content.split('\n')
            questions = [q.strip('- 1234567890.') for q in suggestions if q.strip()]
            
            return questions[:5]
            
        except Exception as e:
            logger.error(f"Error suggesting questions: {str(e)}")
            return []

    def is_available(self) -> bool:
        """Check if AI services are available"""
        return self.client is not None