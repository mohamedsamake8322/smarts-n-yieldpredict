"""
Agricultural Assistant Module

Provides search functionality over the Plantwise knowledge base using FAISS vector search.
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from config import MOH_DIR, MOH_INDEX_FILE, MOH_METADATA_FILE

class AgriculturalAssistant:
    def __init__(self, index_file=None, metadata_file=None):
        """
        Initialize the agricultural assistant.

        Args:
            index_file: Path to FAISS index file (defaults to config.MOH_INDEX_FILE)
            metadata_file: Path to metadata JSON file (defaults to config.MOH_METADATA_FILE)
        """
        self.index_file = index_file or MOH_INDEX_FILE
        self.metadata_file = metadata_file or MOH_METADATA_FILE
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load index and metadata if they exist
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            print(f"Warning: Index files not found at {self.index_file}")
            self.index = None
            self.metadata = []

    def search(self, query, top_k=5):
        """
        Search the knowledge base for relevant information.

        Args:
            query: User's question
            top_k: Number of top results to return

        Returns:
            list: List of relevant entries with scores
        """
        if not self.index:
            return []

        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search index
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                entry = self.metadata[idx]
                results.append({
                    'title': entry['title'],
                    'filename': entry['filename'],
                    'score': float(distances[0][i])
                })

        return results

    def get_detailed_info(self, filename):
        """
        Get detailed information from a specific JSON file.

        Args:
            filename: Name of the JSON file

        Returns:
            dict: Full content of the file
        """
        filepath = os.path.join(MOH_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return {}

    def extract_relevant_sections(self, content, query_keywords=None):
        """
        Extract relevant sections from content based on query.

        Args:
            content: Full JSON content
            query_keywords: Keywords from the query

        Returns:
            dict: Relevant sections
        """
        sections = content.get('sections', {})
        table = sections.get('Table', {})

        relevant = {}

        # Always include title
        relevant['title'] = content.get('title', '')

        # Extract sections based on keywords
        if query_keywords:
            keywords = [kw.lower() for kw in query_keywords]

            if any(kw in ['prevention', 'prevent', 'éviter', 'avoid'] for kw in keywords):
                relevant['Prevention'] = table.get('Prevention', [])

            if any(kw in ['monitoring', 'surveillance', 'observer', 'look', 'symptoms'] for kw in keywords):
                relevant['Monitoring'] = table.get('Monitoring', [])

            if any(kw in ['control', 'contrôle', 'traiter', 'gérer', 'manage', 'treat'] for kw in keywords):
                relevant['Direct Control'] = table.get('Direct Control', [])

        # If no specific keywords, include all sections
        if not relevant or len(relevant) == 1:  # Only title
            relevant.update(table)

        return relevant

    def generate_response(self, query, top_k=3):
        """
        Generate a comprehensive response to the user's query.

        Args:
            query: User's question
            top_k: Number of sources to consider

        Returns:
            dict: Response with answer and sources
        """
        if not self.index:
            return {
                'answer': "The agricultural assistant is not available. Please ensure the FAISS index is built.",
                'sources': []
            }

        # Search for relevant entries
        search_results = self.search(query, top_k=top_k)

        if not search_results:
            return {
                'answer': "Sorry, I couldn't find relevant information in the knowledge base. Please try rephrasing your question.",
                'sources': []
            }

        # Extract keywords from query for section filtering
        query_keywords = query.lower().split()

        # Collect information from top results
        collected_info = []
        sources = []

        for result in search_results:
            content = self.get_detailed_info(result['filename'])
            if content:
                relevant_sections = self.extract_relevant_sections(content, query_keywords)
                collected_info.append(relevant_sections)
                sources.append({
                    'title': result['title'],
                    'score': result['score']
                })

        # Generate answer (placeholder - in real implementation, use LLM)
        answer = self._synthesize_answer(query, collected_info)

        return {
            'answer': answer,
            'sources': sources
        }

    def _synthesize_answer(self, query, collected_info):
        """
        Synthesize a natural language answer from collected information.

        Args:
            query: Original query
            collected_info: List of relevant section data

        Returns:
            str: Synthesized answer
        """
        # Placeholder implementation
        # In real implementation, use an LLM to generate a coherent response

        if not collected_info:
            return "No specific information found."

        # Simple concatenation for now
        response_parts = []

        for info in collected_info[:2]:  # Limit to top 2
            title = info.get('title', '')
            response_parts.append(f"For {title}:")

            for section_name, section_data in info.items():
                if section_name != 'title' and isinstance(section_data, list):
                    if len(section_data) > 1:
                        # Skip title if present
                        content = ' '.join(str(item) for item in section_data[1:] if item)
                        if content:
                            response_parts.append(f"{section_name}: {content}")

        return ' '.join(response_parts) if response_parts else "Information collected, but synthesis needs improvement."