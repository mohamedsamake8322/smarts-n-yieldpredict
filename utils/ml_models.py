import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class YieldPredictor:
    """
    Advanced yield prediction system using multiple ML algorithms.
    Supports Random Forest, Linear Regression, and XGBoost models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_name = 'yield'
        self.is_trained = False
        self.training_history = []
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'linear_regression': {
                'class': LinearRegression,
                'params': {}
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'class': XGBRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning models.
        
        Args:
            data: Raw agricultural data
            
        Returns:
            Processed feature DataFrame
        """
        df = data.copy()
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                # For prediction, handle unseen categories
                try:
                    df[col] = self.encoders[col].transform(df[col])
                except ValueError:
                    # Handle unseen categories by assigning the most frequent class
                    most_frequent = self.encoders[col].classes_[0]
                    df[col] = df[col].apply(lambda x: most_frequent if x not in self.encoders[col].classes_ else x)
                    df[col] = self.encoders[col].transform(df[col])
        
        # Convert date columns if present
        date_columns = ['date', 'planting_date', 'harvest_date']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    # Extract useful date features
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day_of_year'] = df[col].dt.dayofyear
                    df = df.drop(columns=[col])
                except:
                    # If date conversion fails, drop the column
                    df = df.drop(columns=[col])
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Create interaction features for important combinations
        if 'temperature' in df.columns and 'rainfall' in df.columns:
            df['temp_rainfall_ratio'] = df['temperature'] / (df['rainfall'] + 1)
        
        if 'soil_ph' in df.columns and 'soil_nitrogen' in df.columns:
            df['ph_nitrogen_interaction'] = df['soil_ph'] * df['soil_nitrogen']
        
        if 'area' in df.columns and 'temperature' in df.columns:
            df['area_temp_product'] = df['area'] * df['temperature']
        
        return df
    
    def train_model(self, data: pd.DataFrame, model_type: str = 'random_forest', 
                   test_size: float = 0.2, random_state: int = 42) -> Optional[Dict]:
        """
        Train a yield prediction model.
        
        Args:
            data: Training data with features and target
            model_type: Type of model to train
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Training results dictionary or None if failed
        """
        try:
            # Prepare features
            processed_data = self.prepare_features(data)
            
            # Separate features and target
            if self.target_name not in processed_data.columns:
                raise ValueError(f"Target column '{self.target_name}' not found in data")
            
            X = processed_data.drop(columns=[self.target_name])
            y = processed_data[self.target_name]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler_name = f"{model_type}_scaler"
            self.scalers[scaler_name] = StandardScaler()
            X_train_scaled = self.scalers[scaler_name].fit_transform(X_train)
            X_test_scaled = self.scalers[scaler_name].transform(X_test)
            
            # Initialize and train model
            if model_type not in self.model_configs:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model_config = self.model_configs[model_type]
            model = model_config['class'](**model_config['params'])
            model.fit(X_train_scaled, y_train)
            
            # Store model
            self.models[model_type] = model
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_)
            
            # Prepare results
            results = {
                'model_type': model_type,
                'training_date': datetime.now().isoformat(),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'r2_score': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_accuracy': test_r2 * 100,
                'n_features': len(self.feature_names),
                'n_samples': len(data),
                'feature_names': self.feature_names,
                'predictions': y_pred_test.tolist(),
                'actual': y_test.tolist()
            }
            
            if feature_importance is not None:
                results['feature_importance'] = feature_importance.tolist()
            
            # Store training history
            self.training_history.append(results)
            self.is_trained = True
            
            return results
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None
    
    def predict(self, input_data: Dict, model_type: str = 'random_forest') -> Optional[Dict]:
        """
        Make yield predictions using trained model.
        
        Args:
            input_data: Dictionary with input features
            model_type: Type of model to use for prediction
            
        Returns:
            Prediction results dictionary or None if failed
        """
        try:
            if model_type not in self.models:
                # If model not trained, use heuristic prediction
                return self._heuristic_prediction(input_data)
            
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Prepare features
            processed_input = self.prepare_features(input_df)
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(processed_input.columns)
            for feature in missing_features:
                processed_input[feature] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            processed_input = processed_input[self.feature_names]
            
            # Scale features
            scaler_name = f"{model_type}_scaler"
            if scaler_name in self.scalers:
                input_scaled = self.scalers[scaler_name].transform(processed_input)
            else:
                input_scaled = processed_input.values
            
            # Make prediction
            model = self.models[model_type]
            predicted_yield = model.predict(input_scaled)[0]
            
            # Calculate total production
            area = input_data.get('area', 1.0)
            total_production = predicted_yield * area
            
            # Calculate confidence (simplified approach)
            confidence = min(95, max(60, 85 - abs(predicted_yield - 5.0) * 10))
            
            return {
                'yield': max(0, predicted_yield),  # Ensure non-negative yield
                'total_production': max(0, total_production),
                'confidence': confidence,
                'model_used': model_type,
                'prediction_date': datetime.now().isoformat(),
                'input_features': input_data
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return self._heuristic_prediction(input_data)
    
    def _heuristic_prediction(self, input_data: Dict) -> Dict:
        """
        Fallback heuristic prediction when ML model is not available.
        
        Args:
            input_data: Dictionary with input features
            
        Returns:
            Heuristic prediction results
        """
        # Simple heuristic based on crop type and basic conditions
        crop_type = input_data.get('crop_type', 'wheat').lower()
        area = input_data.get('area', 1.0)
        temperature = input_data.get('temperature', 20.0)
        rainfall = input_data.get('rainfall', 500.0)
        soil_ph = input_data.get('soil_ph', 6.5)
        
        # Base yields for different crops (tons/ha)
        base_yields = {
            'wheat': 4.0,
            'corn': 7.0,
            'rice': 5.5,
            'soybeans': 3.0,
            'barley': 3.5,
            'cotton': 2.0
        }
        
        base_yield = base_yields.get(crop_type, 4.0)
        
        # Adjust based on conditions
        temp_factor = 1.0
        if temperature < 10 or temperature > 35:
            temp_factor = 0.7
        elif 15 <= temperature <= 25:
            temp_factor = 1.2
        
        rain_factor = 1.0
        if rainfall < 200:
            rain_factor = 0.6
        elif rainfall > 1500:
            rain_factor = 0.8
        elif 400 <= rainfall <= 800:
            rain_factor = 1.15
        
        ph_factor = 1.0
        if 6.0 <= soil_ph <= 7.5:
            ph_factor = 1.1
        elif soil_ph < 5.5 or soil_ph > 8.0:
            ph_factor = 0.8
        
        # Calculate predicted yield
        predicted_yield = base_yield * temp_factor * rain_factor * ph_factor
        total_production = predicted_yield * area
        
        # Confidence is lower for heuristic predictions
        confidence = 70.0
        
        return {
            'yield': predicted_yield,
            'total_production': total_production,
            'confidence': confidence,
            'model_used': 'heuristic',
            'prediction_date': datetime.now().isoformat(),
            'input_features': input_data
        }
    
    def get_recommendations(self, input_data: Dict, prediction_result: Dict) -> List[str]:
        """
        Generate agricultural recommendations based on input conditions and predictions.
        
        Args:
            input_data: Input feature data
            prediction_result: Prediction results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        predicted_yield = prediction_result.get('yield', 0)
        temperature = input_data.get('temperature', 20)
        rainfall = input_data.get('rainfall', 500)
        soil_ph = input_data.get('soil_ph', 6.5)
        soil_nitrogen = input_data.get('soil_nitrogen', 40)
        humidity = input_data.get('humidity', 60)
        
        # Yield-based recommendations
        if predicted_yield < 3.0:
            recommendations.append("Consider reviewing soil fertility and water management practices")
            recommendations.append("Evaluate crop variety selection for better adaptation to local conditions")
        elif predicted_yield > 7.0:
            recommendations.append("Excellent yield potential - maintain current practices")
            recommendations.append("Consider increasing planting area if resources allow")
        
        # Temperature recommendations
        if temperature < 10:
            recommendations.append("Cold temperatures may affect growth - consider frost protection measures")
        elif temperature > 30:
            recommendations.append("High temperatures detected - ensure adequate irrigation and shade if needed")
        
        # Rainfall recommendations
        if rainfall < 300:
            recommendations.append("Low rainfall predicted - plan for supplemental irrigation")
        elif rainfall > 1200:
            recommendations.append("High rainfall expected - ensure proper drainage to prevent waterlogging")
        
        # Soil pH recommendations
        if soil_ph < 6.0:
            recommendations.append("Soil is acidic - consider lime application to raise pH")
        elif soil_ph > 7.5:
            recommendations.append("Soil is alkaline - consider sulfur application to lower pH")
        
        # Nitrogen recommendations
        if soil_nitrogen < 30:
            recommendations.append("Low nitrogen levels - consider nitrogen fertilizer application")
        elif soil_nitrogen > 60:
            recommendations.append("High nitrogen levels - monitor for potential nitrogen burn")
        
        # Humidity recommendations
        if humidity > 80:
            recommendations.append("High humidity may increase disease risk - monitor for fungal infections")
        elif humidity < 40:
            recommendations.append("Low humidity conditions - increase irrigation frequency if needed")
        
        # General recommendations
        recommendations.append("Regular monitoring of crop health and soil conditions is recommended")
        recommendations.append("Consider soil testing every 6 months for optimal nutrient management")
        
        return recommendations
    
    def get_model_performance(self, model_type: str = 'random_forest') -> Optional[Dict]:
        """
        Get performance metrics for a specific model.
        
        Args:
            model_type: Type of model to get performance for
            
        Returns:
            Performance metrics dictionary or None
        """
        if not self.training_history:
            return None
        
        # Find the latest training results for the specified model
        for result in reversed(self.training_history):
            if result.get('model_type') == model_type:
                return result
        
        return None
    
    def save_model(self, model_type: str, filepath: str) -> bool:
        """
        Save trained model to disk.
        
        Args:
            model_type: Type of model to save
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_type not in self.models:
                return False
            
            model_data = {
                'model': self.models[model_type],
                'scaler': self.scalers.get(f"{model_type}_scaler"),
                'encoders': self.encoders,
                'feature_names': self.feature_names,
                'training_history': self.training_history
            }
            
            joblib.dump(model_data, filepath)
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                return False
            
            model_data = joblib.load(filepath)
            
            # Extract model components
            model_type = 'loaded_model'
            self.models[model_type] = model_data['model']
            self.scalers[f"{model_type}_scaler"] = model_data.get('scaler')
            self.encoders = model_data.get('encoders', {})
            self.feature_names = model_data.get('feature_names', [])
            self.training_history = model_data.get('training_history', [])
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


def prepare_prediction_data(raw_data: Dict) -> pd.DataFrame:
    """
    Prepare raw input data for prediction.
    
    Args:
        raw_data: Raw input data dictionary
        
    Returns:
        Processed DataFrame ready for prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame([raw_data])
    
    # Ensure numeric columns are properly typed
    numeric_columns = [
        'area', 'temperature', 'rainfall', 'humidity', 'sunlight',
        'soil_ph', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
