import os
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class VoiceAssistant:
    """AI-powered voice assistant with multilingual support"""
    
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found. AI features will be disabled.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.language_codes = {
            'french': 'fr',
            'english': 'en', 
            'spanish': 'es',
            'fr': 'fr',
            'en': 'en',
            'es': 'es'
        }
        
        self.language_names = {
            'fr': 'French',
            'en': 'English',
            'es': 'Spanish'
        }
    
    def get_ai_response(self, question, context=None, language='en'):
        """Get AI response to user question with optional document context"""
        if not self.client:
            return "AI assistant is not available. Please check your OpenAI API key configuration."
        
        try:
            # Normalize language code
            lang_code = self.language_codes.get(language.lower(), 'en')
            lang_name = self.language_names.get(lang_code, 'English')
            
            # Build system prompt
            system_prompt = self._build_system_prompt(lang_code, lang_name)
            
            # Build user message with context
            user_message = self._build_user_message(question, context, lang_name)
            
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _build_system_prompt(self, lang_code, lang_name):
        """Build system prompt based on language"""
        prompts = {
            'en': """You are a helpful multilingual voice assistant with access to document content. 
                     You can help users find information in their uploaded documents and answer questions.
                     
                     Guidelines:
                     - Respond in English unless specifically asked otherwise
                     - Be concise but comprehensive in your answers
                     - When referencing document content, cite the source
                     - If you don't find relevant information in the documents, say so clearly
                     - Be friendly and professional in your responses""",
            
            'fr': """Vous êtes un assistant vocal multilingue utile avec accès au contenu des documents.
                     Vous pouvez aider les utilisateurs à trouver des informations dans leurs documents téléchargés et répondre aux questions.
                     
                     Directives:
                     - Répondez en français sauf indication contraire
                     - Soyez concis mais complet dans vos réponses
                     - Lorsque vous référencez le contenu d'un document, citez la source
                     - Si vous ne trouvez pas d'informations pertinentes dans les documents, dites-le clairement
                     - Soyez amical et professionnel dans vos réponses""",
            
            'es': """Eres un asistente de voz multilingüe útil con acceso al contenido de documentos.
                     Puedes ayudar a los usuarios a encontrar información en sus documentos cargados y responder preguntas.
                     
                     Pautas:
                     - Responde en español a menos que se especifique lo contrario
                     - Sé conciso pero completo en tus respuestas
                     - Cuando hagas referencia al contenido del documento, cita la fuente
                     - Si no encuentras información relevante en los documentos, dilo claramente
                     - Sé amigable y profesional en tus respuestas"""
        }
        
        return prompts.get(lang_code, prompts['en'])
    
    def _build_user_message(self, question, context, lang_name):
        """Build user message with context"""
        if not context or len(context) == 0:
            return f"Question: {question}\n\nNote: No relevant document context found."
        
        message = f"Question: {question}\n\nRelevant document context:\n\n"
        
        for i, result in enumerate(context, 1):
            message += f"Document {i} - {result.get('filename', 'Unknown')}:\n"
            message += f"{result.get('content', '')[:500]}...\n\n"
        
        return message
    
    def get_supported_languages(self):
        """Return list of supported languages"""
        return [
            {'code': 'en', 'name': 'English'},
            {'code': 'fr', 'name': 'Français'},
            {'code': 'es', 'name': 'Español'}
        ]
