"""
Fertilizer Prediction API Module

This module provides a robust fertilizer recommendation system using XGBoost.
It includes input validation, automatic encoding, error handling, and model caching.
"""

import pandas as pd
import xgboost as xgb  # type: ignore
import json
import numpy as np
import os
from typing import Dict, Union, Optional, Tuple
import streamlit as st

# 📦 Model and metadata paths
MODEL_PATH = r"C:\smarts-n-yieldpredict.git\models\fertilizer_model.bin"
COLS_PATH = r"C:\smarts-n-yieldpredict.git\models\fertilizer_columns.json"
LABELS_PATH = r"C:\smarts-n-yieldpredict.git\models\fertilizer_labels.json"


def _validate_file_exists(file_path: str, file_name: str) -> None:
    """
    Validates that a file exists before attempting to load it.
    
    Args:
        file_path: Path to the file to check
        file_name: Name of the file for error messages
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ {file_name} not found at {file_path}. "
            f"Please ensure the model files are in the correct location."
        )


@st.cache_resource
def _load_model() -> xgb.XGBClassifier:
    """
    Loads the XGBoost fertilizer prediction model with caching.
    
    Uses @st.cache_resource to prevent reloading the model on every prediction.
    
    Returns:
        XGBoost classifier model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        xgb.core.XGBoostError: If model loading fails
    """
    _validate_file_exists(MODEL_PATH, "Model file")
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        return model
    except xgb.core.XGBoostError as e:
        raise xgb.core.XGBoostError(
            f"❌ Error loading XGBoost model: {str(e)}. "
            f"Please check the model file format."
        ) from e


@st.cache_resource
def _load_metadata() -> Tuple[list, Dict[int, str]]:
    """
    Loads model columns and label mapping with caching.
    
    Returns:
        Tuple of (model_columns_list, label_mapping_dict)
        
    Raises:
        FileNotFoundError: If metadata files don't exist
        json.JSONDecodeError: If JSON files are corrupted
    """
    _validate_file_exists(COLS_PATH, "Columns file")
    _validate_file_exists(LABELS_PATH, "Labels file")
    
    try:
        with open(COLS_PATH, "r", encoding="utf-8") as f:
            model_cols = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"❌ Error parsing columns JSON: {str(e)}",
            e.doc,
            e.pos
        ) from e
    
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"❌ Error parsing labels JSON: {str(e)}",
            e.doc,
            e.pos
        ) from e
    
    return model_cols, label_map


def _validate_numeric_input(value: Union[int, float, str], param_name: str) -> Union[int, float]:
    """
    Validates that a numeric input is a valid number and positive.
    
    Args:
        value: Input value to validate
        param_name: Name of the parameter for error messages
        
    Returns:
        Validated numeric value (int or float)
        
    Raises:
        ValueError: If value is not numeric or is negative
        TypeError: If value type is invalid
    """
    if isinstance(value, str):
        try:
            # Try to convert string to number
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            raise ValueError(
                f"❌ Invalid numeric value for {param_name}: '{value}'. "
                f"Expected a number (int or float)."
            ) from None
    
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"❌ Invalid type for {param_name}: {type(value).__name__}. "
            f"Expected int or float."
        )
    
    if value < 0:
        raise ValueError(
            f"❌ Negative value for {param_name}: {value}. "
            f"Expected a positive number."
        )
    
    return value


def _validate_user_inputs(user_inputs: Dict) -> Dict:
    """
    Validates and sanitizes all user inputs.
    
    Args:
        user_inputs: Dictionary of user input parameters
        
    Returns:
        Validated and sanitized input dictionary
        
    Raises:
        ValueError: If validation fails for any parameter
        TypeError: If parameter types are invalid
    """
    validated_inputs = {}
    
    # Numeric parameters that must be positive
    numeric_params = [
        "Temperature", "Humidity", "Moisture", 
        "Nitrogen", "Phosphorous", "Potassium"
    ]
    
    # Validate numeric inputs
    for param in numeric_params:
        if param in user_inputs:
            validated_inputs[param] = _validate_numeric_input(
                user_inputs[param], 
                param
            )
        else:
            # Set default value if missing
            defaults = {
                "Temperature": 28,
                "Humidity": 60,
                "Moisture": 40,
                "Nitrogen": 20,
                "Phosphorous": 30,
                "Potassium": 10
            }
            if param in defaults:
                validated_inputs[param] = defaults[param]
    
    # Categorical parameters (no validation needed, just copy)
    categorical_params = ["Soil Type", "Crop Type"]
    for param in categorical_params:
        if param in user_inputs:
            validated_inputs[param] = str(user_inputs[param]).strip()
    
    # Copy any other parameters
    for key, value in user_inputs.items():
        if key not in validated_inputs:
            validated_inputs[key] = value
    
    return validated_inputs


def _encode_features(user_inputs: Dict, model_cols: list) -> pd.DataFrame:
    """
    Automatically encodes user inputs into model feature format.
    
    Uses automatic feature encoding instead of manual mapping.
    
    Args:
        user_inputs: Validated user input dictionary
        model_cols: List of expected model columns
        
    Returns:
        DataFrame with encoded features matching model requirements
    """
    # Initialize feature dictionary with zeros
    X_input_dict = {col: 0 for col in model_cols}
    
    # Process each user input
    for key, value in user_inputs.items():
        normalized_key = key.lower().strip().replace(" ", "_")
        normalized_val = str(value).lower().strip()
        
        # One-hot encoding for categorical columns
        for col in model_cols:
            if col.startswith(normalized_key + "_") and normalized_val in col.lower():
                X_input_dict[col] = 1
        
        # Direct numeric value assignment
        if normalized_key in model_cols:
            X_input_dict[normalized_key] = value
    
    # Create DataFrame with correct column order
    X_input = pd.DataFrame([X_input_dict], columns=model_cols)
    
    return X_input


def predict_fertilizer(user_inputs: Dict) -> str:
    """
    Predicts the optimal fertilizer based on user inputs.
    
    This function performs input validation, feature encoding, and model prediction.
    It uses cached model and metadata loading for performance.
    
    Args:
        user_inputs: Dictionary containing:
            - Temperature (numeric): Temperature in Celsius
            - Humidity (numeric): Air humidity percentage
            - Moisture (numeric): Soil moisture percentage
            - Soil Type (string): Type of soil (e.g., "Sandy", "Loamy")
            - Crop Type (string): Type of crop (e.g., "Rice", "Wheat")
            - Nitrogen (numeric): Nitrogen level in soil
            - Phosphorous (numeric): Phosphorous level in soil
            - Potassium (numeric): Potassium level in soil
    
    Returns:
        Recommended fertilizer name as string
        
    Raises:
        FileNotFoundError: If model or metadata files are missing
        ValueError: If input validation fails
        TypeError: If input types are invalid
        xgb.core.XGBoostError: If model prediction fails
        IndexError: If prediction index is out of bounds
    """
    try:
        # Validate inputs
        validated_inputs = _validate_user_inputs(user_inputs)
        
        # Load model and metadata (cached)
        model = _load_model()
        model_cols, label_map = _load_metadata()
        
        # Encode features
        X_input = _encode_features(validated_inputs, model_cols)
        
        # Make prediction
        pred = model.predict(X_input)
        pred_label = int(np.ravel(pred)[0])
        
        # Validate prediction index
        if not (0 <= pred_label < len(label_map)):
            raise IndexError(
                f"❌ Prediction index {pred_label} out of bounds. "
                f"Expected index between 0 and {len(label_map) - 1}."
            )
        
        fertilizer_name = label_map[pred_label]
        return fertilizer_name
        
    except (FileNotFoundError, ValueError, TypeError, 
            xgb.core.XGBoostError, IndexError, json.JSONDecodeError) as e:
        # Re-raise known exceptions with context
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(
            f"❌ Unexpected error during fertilizer prediction: {str(e)}. "
            f"Please check your inputs and try again."
        ) from e
