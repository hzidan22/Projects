

import os
from typing import Dict, Any

class FPLConfig:

    FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    DATA_DIR = "data"
    MODEL_DIR = "models"
    OUTPUT_DIR = "output"
    
    ENHANCED_PLAYERS_FILE = "enhanced_players.csv"
    PROCESSED_PLAYERS_FILE = "processed_players.csv"
    PLAYER_PREDICTIONS_FILE = "player_predictions.csv"
    SELECTED_TEAM_FILE = "selected_team.csv"
    STARTING_XI_FILE = "starting_xi.csv"
    TEAM_REPORT_FILE = "team_report.txt"
    MODEL_FILE = "best_fpl_model.joblib"
    FEATURE_IMPORTANCE_FILE = "model_feature_importance.csv"
    PERFORMANCE_REPORT_FILE = "model_performance_report.csv"
    
    BUDGET = 100.0
    MAX_PLAYERS_PER_TEAM = 3
    SQUAD_SIZE = 15
    STARTING_XI_SIZE = 11
    
    POSITION_LIMITS = {
        'GK': {'min': 2, 'max': 2, 'starting': 1},
        'DEF': {'min': 5, 'max': 5, 'starting': 3},
        'MID': {'min': 5, 'max': 5, 'starting': 4},
        'FWD': {'min': 3, 'max': 3, 'starting': 3}
    }
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    N_FEATURES_SELECT = 20
    
    SCALER_TYPE = 'robust'
    
    FEATURE_SELECTION_METHOD = 'f_regression'
    
    CAPTAIN_WEIGHTS = {
        'predicted_points': 0.5,
        'form_score': 0.2,
        'consistency': 0.15,
        'fixture_bonus': 0.1,
        'position_bonus': 0.05
    }
    
    CAPTAIN_POSITION_BONUS = {
        'GK': 0.8,
        'DEF': 0.9,
        'MID': 1.0,
        'FWD': 1.1
    }
    
    TRANSFER_COST = 4.0
    MAX_FREE_TRANSFERS = 1
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls):
        
        directories = [cls.DATA_DIR, cls.MODEL_DIR, cls.OUTPUT_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_file_path(cls, filename: str, directory: str = None) -> str:
        
        if directory is None:
            if filename.endswith('.csv') or filename.endswith('.txt'):
                directory = cls.OUTPUT_DIR
            elif filename.endswith('.joblib') or filename.endswith('.pkl'):
                directory = cls.MODEL_DIR
            else:
                directory = cls.DATA_DIR
        
        return os.path.join(directory, filename)
    
    @classmethod
    def get_model_hyperparameters(cls) -> Dict[str, Dict[str, Any]]:
        
        return {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        }

class DevelopmentConfig(FPLConfig):
    
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(FPLConfig):
    
    DEBUG = False
    LOG_LEVEL = "WARNING"

config_env = os.getenv('FPL_ENV', 'development').lower()
if config_env == 'production':
    Config = ProductionConfig
else:
    Config = DevelopmentConfig
