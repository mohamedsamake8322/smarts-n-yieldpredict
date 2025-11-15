"""
Unit Tests for Fertilizer API Module

This module contains unit tests for the fertilizer prediction API,
testing input validation, model loading, and prediction functionality.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.fertilizer_api import (
    _validate_numeric_input,
    _validate_user_inputs,
    _encode_features,
    predict_fertilizer
)


class TestFertilizerAPI(unittest.TestCase):
    """Test cases for fertilizer API functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.valid_inputs = {
            "Temperature": 28,
            "Humidity": 60,
            "Moisture": 40,
            "Soil Type": "Loamy",
            "Crop Type": "Rice",
            "Nitrogen": 20,
            "Phosphorous": 30,
            "Potassium": 10
        }
    
    def test_validate_numeric_input_int(self):
        """Test validation of integer input"""
        result = _validate_numeric_input(25, "Temperature")
        self.assertEqual(result, 25)
        self.assertIsInstance(result, int)
    
    def test_validate_numeric_input_float(self):
        """Test validation of float input"""
        result = _validate_numeric_input(28.5, "Temperature")
        self.assertEqual(result, 28.5)
        self.assertIsInstance(result, float)
    
    def test_validate_numeric_input_string(self):
        """Test validation of string numeric input"""
        result = _validate_numeric_input("30", "Temperature")
        self.assertEqual(result, 30)
        self.assertIsInstance(result, int)
    
    def test_validate_numeric_input_negative(self):
        """Test that negative values raise ValueError"""
        with self.assertRaises(ValueError):
            _validate_numeric_input(-5, "Temperature")
    
    def test_validate_numeric_input_invalid_string(self):
        """Test that invalid string raises ValueError"""
        with self.assertRaises(ValueError):
            _validate_numeric_input("abc", "Temperature")
    
    def test_validate_numeric_input_invalid_type(self):
        """Test that invalid type raises TypeError"""
        with self.assertRaises(TypeError):
            _validate_numeric_input([1, 2, 3], "Temperature")
    
    def test_validate_user_inputs_valid(self):
        """Test validation of valid user inputs"""
        result = _validate_user_inputs(self.valid_inputs)
        self.assertEqual(result["Temperature"], 28)
        self.assertEqual(result["Humidity"], 60)
        self.assertEqual(result["Crop Type"], "Rice")
    
    def test_validate_user_inputs_with_missing(self):
        """Test validation with missing optional parameters"""
        partial_inputs = {"Temperature": 25, "Crop Type": "Wheat"}
        result = _validate_user_inputs(partial_inputs)
        self.assertIn("Temperature", result)
        self.assertIn("Crop Type", result)
    
    def test_encode_features(self):
        """Test feature encoding functionality"""
        model_cols = ["temperature", "humidity", "moisture", "nitrogen"]
        result = _encode_features(self.valid_inputs, model_cols)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), len(model_cols))
    
    def test_predict_fertilizer_integration(self):
        """Integration test for full prediction pipeline"""
        # Note: This test requires model files to exist
        # Skip if files are not available
        model_path = r"C:\smarts-n-yieldpredict.git\models\fertilizer_model.bin"
        
        if not os.path.exists(model_path):
            self.skipTest("Model files not found, skipping integration test")
        
        try:
            result = predict_fertilizer(self.valid_inputs)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
        except Exception as e:
            self.fail(f"Prediction failed with error: {e}")
    
    def test_predict_fertilizer_invalid_input(self):
        """Test prediction with invalid input"""
        invalid_inputs = {"Temperature": -10}
        
        with self.assertRaises((ValueError, TypeError)):
            predict_fertilizer(invalid_inputs)


class TestInputValidation(unittest.TestCase):
    """Additional tests for input validation edge cases"""
    
    def test_empty_inputs(self):
        """Test validation with empty inputs"""
        result = _validate_user_inputs({})
        # Should have default values
        self.assertIn("Temperature", result)
    
    def test_extreme_values(self):
        """Test validation with extreme but valid values"""
        extreme_inputs = {
            "Temperature": 50,
            "Humidity": 100,
            "Moisture": 100,
            "Nitrogen": 1000
        }
        result = _validate_user_inputs(extreme_inputs)
        self.assertEqual(result["Temperature"], 50)


if __name__ == '__main__':
    unittest.main()

