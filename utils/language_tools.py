import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Simple language detection based on common words and patterns"""
    
    def __init__(self):
        # Common words in each language
        self.language_patterns = {
            'fr': {
                'common_words': {
                    'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 
                    'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 
                    'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'même', 'du',
                    'elle', 'vous', 'sa', 'comme', 'faire', 'leur', 'nous', 'je',
                    'mais', 'ou', 'si', 'très', 'bien', 'peut', 'cette', 'ces',
                    'la', 'les', 'des', 'qui', 'est', 'où', 'été', 'sont'
                },
                'patterns': [r'\bqu[e\']\b', r'\bd[e\']\b', r'\bl[ae]\b']
            },
            'es': {
                'common_words': {
                    'el', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 
                    'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 
                    'está', 'como', 'del', 'los', 'al', 'una', 'las', 'pero',
                    'todo', 'esta', 'fue', 'ser', 'ha', 'más', 'muy', 'puede',
                    'hasta', 'donde', 'hace', 'cuando', 'porque', 'sobre',
                    'la', 'este', 'ese', 'otros', 'tienen', 'han', 'había'
                },
                'patterns': [r'\bel\b', r'\bla\b', r'\blos\b', r'\blas\b', r'ción\b', r'dad\b']
            },
            'en': {
                'common_words': {
                    'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it',
                    'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
                    'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had',
                    'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when',
                    'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do',
                    'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out',
                    'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would'
                },
                'patterns': [r'\bthe\b', r'ing\b', r'ed\b', r'ly\b']
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the given text"""
        try:
            if not text or not text.strip():
                return 'en'  # Default to English
            
            # Clean and tokenize text
            words = self._tokenize(text.lower())
            
            if not words:
                return 'en'
            
            # Score each language
            scores = {}
            for lang_code, lang_data in self.language_patterns.items():
                scores[lang_code] = self._calculate_language_score(words, text, lang_data)
            
            # Return language with highest score
            best_language = max(scores, key=scores.get)
            
            # Only return detected language if confidence is reasonable
            if scores[best_language] > 0.1:
                logger.info(f"Detected language: {best_language} (score: {scores[best_language]:.3f})")
                return best_language
            
            # Default to English if no clear winner
            logger.info(f"Language detection uncertain, defaulting to English. Scores: {scores}")
            return 'en'
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return 'en'  # Default fallback
    
    def _tokenize(self, text: str) -> list:
        """Tokenize text into words"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Filter out very short words and numbers
        words = [word for word in words if len(word) > 1 and word.isalpha()]
        
        return words
    
    def _calculate_language_score(self, words: list, full_text: str, lang_data: dict) -> float:
        """Calculate language score based on common words and patterns"""
        if not words:
            return 0.0
        
        score = 0.0
        total_words = len(words)
        
        # Score based on common words
        common_words = lang_data['common_words']
        word_matches = sum(1 for word in words if word in common_words)
        word_score = word_matches / total_words if total_words > 0 else 0
        
        # Score based on patterns
        pattern_score = 0
        patterns = lang_data.get('patterns', [])
        for pattern in patterns:
            matches = len(re.findall(pattern, full_text.lower()))
            pattern_score += matches / len(full_text.split()) if full_text.split() else 0
        
        # Combine scores
        score = (word_score * 0.7) + (pattern_score * 0.3)
        
        return score
    
    def get_supported_languages(self) -> dict:
        """Return list of supported languages"""
        return {
            'en': 'English',
            'fr': 'Français', 
            'es': 'Español'
        }
