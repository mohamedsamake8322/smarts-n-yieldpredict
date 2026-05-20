import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Directory paths
MOH_DIR = 'Moh'
INDEX_FILE = 'moh_index.faiss'
METADATA_FILE = 'moh_metadata.json'

def extract_text_from_moh(data):
    """Extract searchable text from Moh JSON."""
    text_parts = []

    # Title
    text_parts.append(data.get('title', ''))

    # Header
    header = data.get('sections', {}).get('Header', [])
    text_parts.extend(header)

    # Table sections
    table = data.get('sections', {}).get('Table', {})
    for section, content in table.items():
        if isinstance(content, list):
            text_parts.extend(content)

    # Figures captions
    figures = data.get('sections', {}).get('Figures', [])
    for fig in figures:
        text_parts.append(fig.get('caption', ''))

    return ' '.join(text_parts)

def build_index():
    """Build FAISS index for Moh JSON files."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model

    texts = []
    metadata = []

    for filename in os.listdir(MOH_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(MOH_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                text = extract_text_from_moh(data)
                if text.strip():
                    texts.append(text)
                    metadata.append({
                        'title': data.get('title', ''),
                        'filename': filename,
                        'filepath': filepath
                    })
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Loaded {len(texts)} documents.")

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Index saved to {INDEX_FILE}, metadata to {METADATA_FILE}")

def search_query(query, top_k=5):
    """Search the index with a query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Encode query
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    # Search
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                'title': metadata[idx]['title'],
                'filename': metadata[idx]['filename'],
                'score': float(distances[0][i])
            })

    return results

if __name__ == "__main__":
    build_index()

    # Example search
    query = "Comment contrôler la bruche du haricot ?"
    results = search_query(query)
    print(f"Query: {query}")
    for res in results:
        print(f"- {res['title']} (score: {res['score']:.3f})")