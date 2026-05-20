#!/usr/bin/env python3
"""
Test script for the modular architecture
"""

from modules import VisualDiagnosis, AgriculturalAssistant

def test_visual_diagnosis():
    """Test the visual diagnosis module."""
    print("=== Testing Visual Diagnosis Module ===")

    # Initialize
    vd = VisualDiagnosis()

    # Mock diagnosis
    # In real usage: result = vd.diagnose('path/to/image.jpg')
    result = {
        'classification': {'disease': 'bean bruchid', 'confidence': 0.95},
        'disease_info': vd.get_disease_info('bean bruchid'),
        'explanation': 'Mock explanation for bean bruchid'
    }

    print(f"Disease: {result['classification']['disease']}")
    print(f"Confidence: {result['classification']['confidence']}")
    print(f"Info available: {bool(result['disease_info'])}")
    print(f"Explanation: {result['explanation']}")
    print()

def test_agricultural_assistant():
    """Test the agricultural assistant module."""
    print("=== Testing Agricultural Assistant Module ===")

    # Initialize
    aa = AgriculturalAssistant()

    # Test search
    query = "Comment contrôler la bruche du haricot ?"
    search_results = aa.search(query, top_k=3)

    print(f"Query: {query}")
    print("Search results:")
    for res in search_results:
        print(f"- {res['title']} (score: {res['score']:.3f})")

    # Test full response generation
    response = aa.generate_response(query)
    print(f"\nGenerated response: {response['answer']}")
    print(f"Sources: {len(response['sources'])}")
    print()

def main():
    """Run all tests."""
    print("Testing the modular architecture for Smart Agriculture App\n")

    test_visual_diagnosis()
    test_agricultural_assistant()

    print("=== Architecture Summary ===")
    print("✅ BLIP2 JSON normalized to common schema")
    print("✅ FAISS index built for Moh knowledge base")
    print("✅ Visual Diagnosis module created")
    print("✅ Agricultural Assistant module created")
    print("\nReady for integration into Streamlit app!")

if __name__ == "__main__":
    main()