import logging
import numpy as np
from typing import List, Dict, Optional
import re
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class VectorStore:
    """In-memory vector store for document search and retrieval"""
    
    def __init__(self):
        self.documents = {}  # doc_id -> document info
        self.vectors = {}    # doc_id -> vector representation
        self.next_id = 1
        self.word_frequencies = defaultdict(int)
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
    
    def add_document(self, filename: str, content: str, language: str = 'en') -> int:
        """Add a document to the vector store"""
        try:
            doc_id = self.next_id
            self.next_id += 1
            
            # Store document info
            self.documents[doc_id] = {
                'id': doc_id,
                'filename': filename,
                'content': content,
                'language': language,
                'word_count': len(content.split())
            }
            
            # Create vector representation
            vector = self._create_vector(content)
            self.vectors[doc_id] = vector
            
            # Update global statistics
            self.total_documents += 1
            self._update_document_frequencies(content)
            
            logger.info(f"Added document {doc_id}: {filename} ({language})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for documents similar to the query"""
        try:
            if not self.documents:
                return []
            
            # Create query vector
            query_vector = self._create_vector(query)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_vector in self.vectors.items():
                similarity = self._cosine_similarity(query_vector, doc_vector)
                if similarity > 0.1:  # Minimum similarity threshold
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for doc_id, similarity in similarities[:limit]:
                doc = self.documents[doc_id]
                results.append({
                    'id': doc_id,
                    'filename': doc['filename'],
                    'content': self._get_relevant_snippet(doc['content'], query),
                    'language': doc['language'],
                    'similarity': similarity
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the store"""
        try:
            return [
                {
                    'id': doc['id'],
                    'filename': doc['filename'],
                    'language': doc['language'],
                    'word_count': doc['word_count']
                }
                for doc in self.documents.values()
            ]
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete a document from the store"""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                del self.vectors[doc_id]
                self.total_documents -= 1
                logger.info(f"Deleted document {doc_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def _create_vector(self, text: str) -> np.ndarray:
        """Create a TF-IDF vector representation of text"""
        try:
            # Tokenize and clean text
            words = self._tokenize(text)
            
            # Calculate term frequencies
            word_count = len(words)
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
                self.word_frequencies[word] += 1
            
            # Create vocabulary from all words seen so far
            vocabulary = list(self.word_frequencies.keys())
            vector = np.zeros(len(vocabulary))
            
            # Calculate TF-IDF values
            for i, word in enumerate(vocabulary):
                if word in word_freq:
                    tf = word_freq[word] / word_count
                    idf = math.log(max(1, self.total_documents) / max(1, self.document_frequencies[word]))
                    vector[i] = tf * idf
            
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
            
        except Exception as e:
            logger.error(f"Error creating vector: {str(e)}")
            return np.array([])
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words and filter
        words = text.split()
        words = [word for word in words if len(word) > 2 and word.isalpha()]
        
        return words
    
    def _update_document_frequencies(self, text: str):
        """Update document frequency statistics"""
        words = set(self._tokenize(text))
        for word in words:
            self.document_frequencies[word] += 1
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if len(vec1) != len(vec2):
                # Pad shorter vector with zeros
                max_len = max(len(vec1), len(vec2))
                if len(vec1) < max_len:
                    vec1 = np.pad(vec1, (0, max_len - len(vec1)))
                if len(vec2) < max_len:
                    vec2 = np.pad(vec2, (0, max_len - len(vec2)))
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def _get_relevant_snippet(self, content: str, query: str, max_length: int = 300) -> str:
        """Extract relevant snippet from document content"""
        try:
            query_words = set(self._tokenize(query))
            sentences = content.split('.')
            
            # Score sentences by query word overlap
            sentence_scores = []
            for sentence in sentences:
                sentence_words = set(self._tokenize(sentence))
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0:
                    sentence_scores.append((sentence.strip(), overlap))
            
            if sentence_scores:
                # Sort by score and take best sentences
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                snippet = '. '.join([s[0] for s in sentence_scores[:2]])
                
                if len(snippet) > max_length:
                    snippet = snippet[:max_length] + '...'
                
                return snippet
            
            # Fallback: return beginning of content
            return content[:max_length] + ('...' if len(content) > max_length else '')
            
        except Exception as e:
            logger.error(f"Error extracting snippet: {str(e)}")
            return content[:max_length] + ('...' if len(content) > max_length else '')
