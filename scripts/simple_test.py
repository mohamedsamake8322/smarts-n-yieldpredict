#!/usr/bin/env python3
"""
Simple API test script
"""

import requests

def test_api():
    url = "https://mohamedsamake8322-sene-disease-api.hf.space/health"
    print(f"Testing: {url}")

    try:
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ API is working!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()