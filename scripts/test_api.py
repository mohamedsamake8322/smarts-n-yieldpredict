#!/usr/bin/env python3
"""
Test script for Plant Disease Detection API
"""

import requests
import time
from pathlib import Path

def test_api(base_url: str = "http://localhost:7860"):
    """Test all API endpoints"""

    print(f"🧪 Testing API at {base_url}")

    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check: {data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")

    # Test 3: Prediction with sample image
    print("\n3. Testing prediction endpoint...")

    # Create a simple test image (1x1 pixel)
    from PIL import Image
    import io

    # Create a small green image (plant-like)
    test_image = Image.new('RGB', (224, 224), color=(50, 150, 50))
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()

    try:
        files = {"file": ("test_plant.jpg", img_bytes, "image/jpeg")}
        response = requests.post(f"{base_url}/predict", files=files, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Disease: {result.get('predicted_disease', 'N/A')}")
            print(".2%")
            print(f"   Unknown: {result.get('is_unknown', 'N/A')}")
            print(f"   Top neighbors: {len(result.get('topk_neighbors', []))}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"

    success = test_api(base_url)
    sys.exit(0 if success else 1)