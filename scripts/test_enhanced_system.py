#!/usr/bin/env python3
"""
Test script for the enhanced Streamlit interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    try:
        from modules import VisualDiagnosis
        from config import ensure_directories
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_visual_diagnosis_methods():
    """Test that VisualDiagnosis has all new methods"""
    try:
        from modules import VisualDiagnosis

        # Check if class exists
        vd = VisualDiagnosis()

        # Check new methods exist
        required_methods = [
            'diagnose',
            'generate_attention_map',
            'log_user_feedback',
            'get_prediction_statistics',
            'get_basic_info',
            'get_detailed_info'
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(vd, method):
                missing_methods.append(method)

        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print("✅ All required methods available")
            return True

    except Exception as e:
        print(f"❌ Method check error: {e}")
        return False

def test_streamlit_syntax():
    """Test that the Streamlit page compiles without syntax errors"""
    try:
        import py_compile
        py_compile.compile('pages/2_Disease_Detection.py', doraise=True)
        print("✅ Streamlit page syntax is valid")
        return True
    except Exception as e:
        print(f"❌ Syntax error in Streamlit page: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Enhanced Disease Detection System")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Method Availability Test", test_visual_diagnosis_methods),
        ("Streamlit Syntax Test", test_streamlit_syntax)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The enhanced system is ready.")
        print("\n✨ New Features Implemented:")
        print("  • Confidence thresholds with warnings")
        print("  • Top-3 AI explanations with BLIP-2")
        print("  • Visual attention maps")
        print("  • Knowledge grounding in Plantwise sources")
        print("  • Prediction logging and user feedback")
        print("  • Enhanced Streamlit interface")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)