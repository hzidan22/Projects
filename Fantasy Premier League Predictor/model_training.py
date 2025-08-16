

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None
from sklearn.ensemble import VotingRegressor
import joblib
import logging
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FPLModelTrainer:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.model_scores = {}
        
    def get_model_configurations(self) -> Dict:
        
        return {
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1) if LIGHTGBM_AVAILABLE else None,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                } if LIGHTGBM_AVAILABLE else {}
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky']
                }
            }
        }
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series, 
                          use_grid_search: bool = True, cv_folds: int = 5) -> Dict:
        
        logger.info(f"Training {model_name}...")
        
        config = self.get_model_configurations()[model_name]
        model = config['model']
        params = config['params']
        
        if use_grid_search and len(params) > 0:
            search = RandomizedSearchCV(
                model, params, cv=cv_folds, scoring='neg_mean_squared_error',
                n_iter=20, random_state=self.random_state, n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
        
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, 
                                   scoring='neg_mean_squared_error')
        cv_mean = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        results = {
            'model': best_model,
            'best_params': best_params,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mse_mean': cv_mean,
            'cv_mse_std': cv_std,
            'predictions': y_test_pred
        }
        
        logger.info(f"{model_name} - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        
        logger.info("Training all models...")
        
        results = {}
        model_configs = self.get_model_configurations()
        
        for model_name in model_configs.keys():
            if model_name == 'lightgbm' and not LIGHTGBM_AVAILABLE:
                logger.warning("Skipping LightGBM - not installed")
                continue
                
            try:
                results[model_name] = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test
                )
                self.models[model_name] = results[model_name]['model']
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mse'])
        self.best_model = results[best_model_name]['model']
        self.model_scores = results
        
        logger.info(f"Best model: {best_model_name}")
        return results
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        
        logger.info("Creating ensemble model...")
        
        if not self.models:
            logger.error("No trained models available for ensemble")
            return {}
        
        sorted_models = sorted(self.model_scores.items(), 
                             key=lambda x: x[1]['test_mse'])[:3]
        
        ensemble_models = []
        for model_name, _ in sorted_models:
            ensemble_models.append((model_name, self.models[model_name]))
        
        ensemble = VotingRegressor(estimators=ensemble_models)
        ensemble.fit(X_train, y_train)
        
        y_train_pred = ensemble.predict(X_train)
        y_test_pred = ensemble.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, 
                                   scoring='neg_mean_squared_error')
        
        ensemble_results = {
            'model': ensemble,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mse_mean': -cv_scores.mean(),
            'cv_mse_std': cv_scores.std(),
            'component_models': [name for name, _ in sorted_models],
            'predictions': y_test_pred
        }
        
        logger.info(f"Ensemble - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")
        
        if test_mse < min([scores['test_mse'] for scores in self.model_scores.values()]):
            self.best_model = ensemble
            logger.info("Ensemble model is the new best model!")
        
        return ensemble_results
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            logger.warning("Model doesn't support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict_player_points(self, X: pd.DataFrame, model=None) -> np.ndarray:
        
        if model is None:
            model = self.best_model
        
        if model is None:
            logger.error("No trained model available")
            return np.array([])
        
        predictions = model.predict(X)
        return predictions
    
    def save_model(self, filepath: str = "fpl_model.joblib", model=None) -> None:
        
        if model is None:
            model = self.best_model
        
        if model is None:
            logger.error("No model to save")
            return
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "fpl_model.joblib"):
        
        try:
            model = joblib.load(filepath)
            self.best_model = model
            logger.info(f"Model loaded from {filepath}")
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            return None
    
    def generate_model_report(self, results: Dict) -> pd.DataFrame:
        
        report_data = []
        
        for model_name, metrics in results.items():
            report_data.append({
                'Model': model_name,
                'Test_R2': metrics['test_r2'],
                'Test_MSE': metrics['test_mse'],
                'Test_MAE': metrics.get('test_mae', 0),
                'CV_MSE_Mean': metrics['cv_mse_mean'],
                'CV_MSE_Std': metrics['cv_mse_std'],
                'Overfit_Score': metrics['train_r2'] - metrics['test_r2']
            })
        
        report_df = pd.DataFrame(report_data).sort_values('Test_R2', ascending=False)
        return report_df

def main():
    
    from data_preprocessing import FPLDataPreprocessor
    
    try:
        preprocessor = FPLDataPreprocessor()
        df = preprocessor.load_data("enhanced_players.csv")
        cleaned_df = preprocessor.clean_data(df)
        enhanced_df = preprocessor.engineer_features(cleaned_df)
        data_dict = preprocessor.prepare_modeling_data(enhanced_df)
        
        trainer = FPLModelTrainer()
        
        results = trainer.train_all_models(
            data_dict['X_train'], data_dict['y_train'],
            data_dict['X_test'], data_dict['y_test']
        )
        
        ensemble_results = trainer.create_ensemble_model(
            data_dict['X_train'], data_dict['y_train'],
            data_dict['X_test'], data_dict['y_test']
        )
        
        if ensemble_results:
            results['ensemble'] = ensemble_results
        
        report_df = trainer.generate_model_report(results)
        report_df.to_csv("model_performance_report.csv", index=False)
        
        feature_importance = trainer.get_feature_importance(
            trainer.best_model, data_dict['feature_names']
        )
        feature_importance.to_csv("model_feature_importance.csv", index=False)
        
        trainer.save_model("best_fpl_model.joblib")
        
        predictions = trainer.predict_player_points(data_dict['X_full'])
        
        results_df = data_dict['original_data'].copy()
        results_df['predicted_points'] = predictions
        results_df.to_csv("player_predictions.csv", index=False)
        
        print(f"\n✅ Model training complete!")
        print(f"\n📊 Model Performance Report:")
        print(report_df.to_string(index=False))
        
        print(f"\n🏆 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        print(f"\n🎯 Top 10 Predicted Players:")
        top_players = results_df.nlargest(10, 'predicted_points')[
            ['first_name', 'second_name', 'position', 'team_name', 'predicted_points', 'now_cost']
        ]
        print(top_players.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

if __name__ == "__main__":
    main()
