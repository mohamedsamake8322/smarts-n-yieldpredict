"""
Service d'optimisation des modèles IA pour performances maximales
Optimisation automatique des hyperparamètres et pipeline ML
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from typing import Dict, List, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    """Optimiseur automatique de modèles ML pour l'agriculture"""
    
    def __init__(self):
        self.best_models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.optimization_history = []
        
        # Configurations des modèles à optimiser
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9, 12],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'saga']
                }
            }
        }
    
    def optimize_preprocessing(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimise le preprocessing des données"""
        best_score = -np.inf
        best_config = {}
        
        # Test différents scalers
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Test différents sélecteurs de features
        feature_selectors = {
            'selectk_10': SelectKBest(f_regression, k=min(10, X.shape[1])),
            'selectk_15': SelectKBest(f_regression, k=min(15, X.shape[1])),
            'rfe_rf': RFE(RandomForestRegressor(n_estimators=50, random_state=42), 
                         n_features_to_select=min(12, X.shape[1]))
        }
        
        for scaler_name, scaler in scalers.items():
            for selector_name, selector in feature_selectors.items():
                try:
                    # Pipeline de preprocessing
                    X_scaled = scaler.fit_transform(X)
                    X_selected = selector.fit_transform(X_scaled, y)
                    
                    # Évaluation rapide avec Random Forest
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    scores = cross_val_score(rf, X_selected, y, cv=3, scoring='r2')
                    score = scores.mean()
                    
                    if score > best_score:
                        best_score = score
                        best_config = {
                            'scaler': scaler_name,
                            'selector': selector_name,
                            'score': score,
                            'n_features': X_selected.shape[1]
                        }
                
                except Exception as e:
                    continue
        
        return best_config
    
    def optimize_model_hyperparameters(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                                     preprocessing_config: Dict) -> Dict[str, Any]:
        """Optimise les hyperparamètres d'un modèle spécifique"""
        
        if model_name not in self.model_configs:
            raise ValueError(f"Modèle {model_name} non supporté")
        
        # Application du preprocessing optimal
        scaler_name = preprocessing_config['scaler']
        selector_name = preprocessing_config['selector']
        
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        scaler = scalers[scaler_name]
        X_scaled = scaler.fit_transform(X)
        
        if 'selectk' in selector_name:
            k = int(selector_name.split('_')[1])
            selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
        else:  # RFE
            selector = RFE(RandomForestRegressor(n_estimators=50, random_state=42), 
                          n_features_to_select=min(12, X.shape[1]))
        
        X_selected = selector.fit_transform(X_scaled, y)
        
        # Optimisation des hyperparamètres
        model_config = self.model_configs[model_name]
        base_model = model_config['model'](random_state=42)
        
        # Utilisation de RandomizedSearchCV pour efficacité
        search = RandomizedSearchCV(
            base_model,
            model_config['params'],
            n_iter=50,
            cv=5,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_selected, y)
        
        # Évaluation finale
        best_model = search.best_estimator_
        cv_scores = cross_val_score(best_model, X_selected, y, cv=5, scoring='r2')
        
        return {
            'model': best_model,
            'best_params': search.best_params_,
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'scaler': scaler,
            'selector': selector,
            'preprocessing': preprocessing_config
        }
    
    def full_optimization_pipeline(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Pipeline complet d'optimisation"""
        
        print("Démarrage de l'optimisation complète des modèles...")
        
        # 1. Optimisation du preprocessing
        print("1. Optimisation du preprocessing...")
        best_preprocessing = self.optimize_preprocessing(X, y)
        print(f"   Meilleure configuration: {best_preprocessing}")
        
        # 2. Optimisation de chaque modèle
        optimized_models = {}
        
        for model_name in self.model_configs.keys():
            print(f"2. Optimisation du modèle {model_name}...")
            try:
                result = self.optimize_model_hyperparameters(model_name, X, y, best_preprocessing)
                optimized_models[model_name] = result
                print(f"   Score CV: {result['cv_score_mean']:.4f} ± {result['cv_score_std']:.4f}")
            except Exception as e:
                print(f"   Erreur lors de l'optimisation de {model_name}: {e}")
                continue
        
        # 3. Sélection du meilleur modèle
        best_model_name = max(optimized_models.keys(), 
                             key=lambda k: optimized_models[k]['cv_score_mean'])
        best_result = optimized_models[best_model_name]
        
        print(f"3. Meilleur modèle: {best_model_name} (Score: {best_result['cv_score_mean']:.4f})")
        
        # Sauvegarde
        self.best_models = optimized_models
        self.scalers[best_model_name] = best_result['scaler']
        self.feature_selectors[best_model_name] = best_result['selector']
        
        return {
            'best_model_name': best_model_name,
            'best_model': best_result['model'],
            'all_models': optimized_models,
            'preprocessing_config': best_preprocessing,
            'optimization_summary': {
                'models_tested': len(optimized_models),
                'best_score': best_result['cv_score_mean'],
                'improvement_over_baseline': best_result['cv_score_mean'] - 0.7  # Baseline supposée
            }
        }
    
    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Crée un modèle d'ensemble optimisé"""
        
        if not self.best_models:
            raise ValueError("Aucun modèle optimisé disponible. Lancez d'abord l'optimisation.")
        
        # Préparation des prédictions de base
        base_predictions = {}
        weights = {}
        
        for model_name, model_data in self.best_models.items():
            try:
                # Application du preprocessing spécifique
                scaler = model_data['scaler']
                selector = model_data['selector']
                model = model_data['model']
                
                X_scaled = scaler.transform(X)
                X_selected = selector.transform(X_scaled)
                
                # Prédictions cross-validation
                predictions = []
                for train_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(X_selected):
                    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    predictions.extend(list(zip(val_idx, pred)))
                
                # Reconstruction des prédictions ordonnées
                predictions.sort(key=lambda x: x[0])
                base_predictions[model_name] = [p[1] for p in predictions]
                
                # Poids basé sur le score CV
                weights[model_name] = model_data['cv_score_mean']
                
            except Exception as e:
                print(f"Erreur avec le modèle {model_name}: {e}")
                continue
        
        # Normalisation des poids
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calcul des prédictions d'ensemble
        ensemble_predictions = np.zeros(len(y))
        for model_name, predictions in base_predictions.items():
            ensemble_predictions += np.array(predictions) * weights[model_name]
        
        # Évaluation de l'ensemble
        ensemble_score = r2_score(y, ensemble_predictions)
        ensemble_mae = mean_absolute_error(y, ensemble_predictions)
        
        return {
            'ensemble_predictions': ensemble_predictions,
            'weights': weights,
            'ensemble_score': ensemble_score,
            'ensemble_mae': ensemble_mae,
            'base_models': self.best_models,
            'improvement_over_best': ensemble_score - max(m['cv_score_mean'] for m in self.best_models.values())
        }
    
    def save_optimized_models(self, filepath_prefix: str = "models/optimized_"):
        """Sauvegarde tous les modèles optimisés"""
        
        for model_name, model_data in self.best_models.items():
            # Sauvegarde du modèle
            model_path = f"{filepath_prefix}{model_name}_model.joblib"
            joblib.dump(model_data['model'], model_path)
            
            # Sauvegarde du scaler
            scaler_path = f"{filepath_prefix}{model_name}_scaler.joblib"
            joblib.dump(model_data['scaler'], scaler_path)
            
            # Sauvegarde du sélecteur de features
            selector_path = f"{filepath_prefix}{model_name}_selector.joblib"
            joblib.dump(model_data['selector'], selector_path)
            
            print(f"Modèle {model_name} sauvegardé avec succès")
        
        # Sauvegarde de l'historique d'optimisation
        history_path = f"{filepath_prefix}optimization_history.joblib"
        joblib.dump(self.optimization_history, history_path)
        
        return True
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Génère un rapport détaillé de l'optimisation"""
        
        if not self.best_models:
            return {"error": "Aucune optimisation effectuée"}
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "models_optimized": len(self.best_models),
            "best_model": max(self.best_models.keys(), 
                            key=lambda k: self.best_models[k]['cv_score_mean']),
            "performance_summary": {},
            "recommendations": []
        }
        
        # Résumé des performances
        for model_name, model_data in self.best_models.items():
            report["performance_summary"][model_name] = {
                "cv_score": model_data['cv_score_mean'],
                "cv_std": model_data['cv_score_std'],
                "best_params": model_data['best_params'],
                "n_features_selected": model_data['preprocessing']['n_features']
            }
        
        # Recommandations
        best_score = max(m['cv_score_mean'] for m in self.best_models.values())
        
        if best_score > 0.9:
            report["recommendations"].append("Performance excellente - Modèle prêt pour la production")
        elif best_score > 0.8:
            report["recommendations"].append("Performance bonne - Possible amélioration avec plus de données")
        else:
            report["recommendations"].append("Performance à améliorer - Collecte de données supplémentaires recommandée")
        
        return report

# Utilitaires pour l'optimisation en lot
def optimize_models_batch(data_sources: List[pd.DataFrame]) -> List[Dict]:
    """Optimise les modèles sur plusieurs jeux de données"""
    results = []
    
    for i, data in enumerate(data_sources):
        print(f"Optimisation du dataset {i+1}/{len(data_sources)}")
        
        # Séparation features/target
        if 'yield' in data.columns:
            X = data.drop('yield', axis=1)
            y = data['yield']
            
            # Nettoyage des données catégorielles
            X_numeric = X.select_dtypes(include=[np.number])
            
            optimizer = ModelOptimizer()
            result = optimizer.full_optimization_pipeline(X_numeric, y)
            result['dataset_index'] = i
            results.append(result)
        
    return results

from sklearn.model_selection import KFold