

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FPLDataPreprocessor:

    def __init__(self, scaler_type: str = 'robust'):
        
        self.scaler_type = scaler_type
        self.scaler = self._get_scaler()
        self.feature_selector = None
        self.selected_features = None
        
    def _get_scaler(self):
        
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers.get(self.scaler_type, RobustScaler())
    
    def load_data(self, filepath: str = "enhanced_players.csv") -> pd.DataFrame:
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Cleaning data...")
        
        cleaned_df = df.copy()
        
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df[cleaned_df['minutes'] > 0].copy()
        logger.info(f"Removed {initial_count - len(cleaned_df)} players with 0 minutes")
        
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)
        
        for col in ['total_points', 'now_cost', 'minutes']:
            if col in cleaned_df.columns:
                mean_val = cleaned_df[col].mean()
                std_val = cleaned_df[col].std()
                cleaned_df = cleaned_df[
                    (cleaned_df[col] >= mean_val - 3*std_val) & 
                    (cleaned_df[col] <= mean_val + 3*std_val)
                ].copy()
        
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        logger.info(f"Cleaned data shape: {cleaned_df.shape}")
        return cleaned_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Engineering advanced features...")
        
        enhanced_df = df.copy()
        
        position_mapping = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        if 'position' in enhanced_df.columns:
            enhanced_df['position_encoded'] = enhanced_df['position'].map(position_mapping)
        else:
            element_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
            enhanced_df['position_encoded'] = enhanced_df['element_type'].map(element_mapping)
        
        enhanced_df['points_per_90'] = enhanced_df['total_points'] / (enhanced_df['minutes'] / 90).replace(0, 1)
        enhanced_df['goals_per_90'] = enhanced_df['goals_scored'] / (enhanced_df['minutes'] / 90).replace(0, 1)
        enhanced_df['assists_per_90'] = enhanced_df['assists'] / (enhanced_df['minutes'] / 90).replace(0, 1)
        
        enhanced_df['points_per_cost'] = enhanced_df['total_points'] / enhanced_df['now_cost'].replace(0, 0.1)
        enhanced_df['cost_efficiency'] = enhanced_df['points_per_90'] / enhanced_df['now_cost'].replace(0, 0.1)
        
        enhanced_df['creativity_threat'] = enhanced_df['creativity'] * enhanced_df['threat']
        enhanced_df['influence_minutes'] = enhanced_df['influence'] * (enhanced_df['minutes'] / 90)
        
        enhanced_df['gk_saves_value'] = np.where(
            enhanced_df['position_encoded'] == 0,
            enhanced_df.get('saves_per_90', 0) * 0.1,
            0
        )
        
        enhanced_df['def_clean_sheet_value'] = np.where(
            enhanced_df['position_encoded'] == 1,
            enhanced_df['clean_sheets'] * 4,
            0
        )
        
        enhanced_df['mid_creativity_value'] = np.where(
            enhanced_df['position_encoded'] == 2,
            enhanced_df['creativity'] * 0.1,
            0
        )
        
        enhanced_df['fwd_goal_value'] = np.where(
            enhanced_df['position_encoded'] == 3,
            enhanced_df['goals_scored'] * 6,
            0
        )
        
        enhanced_df['recent_form'] = enhanced_df.get('form_score', enhanced_df['points_per_90'])
        enhanced_df['consistency'] = enhanced_df['total_points'] / (enhanced_df['minutes'] / 90).replace(0, 1)
        
        enhanced_df['card_risk'] = enhanced_df['red_cards'] * 3 + enhanced_df['yellow_cards']
        enhanced_df['injury_risk'] = np.where(enhanced_df['minutes'] < enhanced_df['minutes'].quantile(0.25), 1, 0)
        
        enhanced_df['ownership_trend'] = enhanced_df['selected_by_percent']
        enhanced_df['value_vs_ownership'] = enhanced_df['points_per_cost'] / (enhanced_df['selected_by_percent'] + 1)
        
        enhanced_df['captain_potential'] = (
            enhanced_df['points_per_90'] * 0.4 +
            enhanced_df['consistency'] * 0.3 +
            (5 - enhanced_df.get('fixture_difficulty', 3)) * 0.2 +
            enhanced_df['influence'] * 0.1
        )
        
        enhanced_df['overall_value'] = (
            enhanced_df['total_points'] * 0.3 +
            enhanced_df['points_per_cost'] * 0.3 +
            enhanced_df['consistency'] * 0.2 +
            enhanced_df['captain_potential'] * 0.2
        )
        
        logger.info(f"Feature engineering complete. New shape: {enhanced_df.shape}")
        return enhanced_df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       k_features: int = 20, method: str = 'f_regression') -> Tuple[pd.DataFrame, List[str]]:
        
        logger.info(f"Selecting top {k_features} features using {method}...")
        
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k_features, X.shape[1]))
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k_features, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.feature_selector = selector
        self.selected_features = selected_features
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        logger.info("Top 10 features:")
        logger.info("\n" + feature_scores.head(10).to_string(index=False))
        
        return X_selected_df, selected_features
    
    def prepare_modeling_data(self, df: pd.DataFrame, target_col: str = 'total_points',
                             test_size: float = 0.2, random_state: int = 42) -> Dict:
        
        logger.info("Preparing data for modeling...")
        
        exclude_cols = [
            target_col, 'first_name', 'second_name', 'web_name', 'team_name',
            'position', 'id', 'code', 'photo'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X[col] = X[col].astype('category').cat.codes
        
        X = X.fillna(0)
        
        X_selected, selected_features = self.select_features(X, y)
        
        X_scaled = self.scaler.fit_transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_full': X_scaled_df,
            'y_full': y,
            'feature_names': selected_features,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'original_data': df
        }
        
        logger.info(f"Data preparation complete:")
        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Test set: {X_test.shape}")
        logger.info(f"  Features: {len(selected_features)}")
        
        return data_dict
    
    def process_current_season(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Processing current season data...")
        
        cleaned_df = self.clean_data(df)
        enhanced_df = self.engineer_features(cleaned_df)
        
        exclude_cols = [
            'total_points', 'first_name', 'second_name', 'web_name', 'team_name',
            'position', 'id', 'code', 'photo'
        ]
        
        feature_cols = [col for col in enhanced_df.columns if col not in exclude_cols]
        X = enhanced_df[feature_cols].copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X[col] = X[col].astype('category').cat.codes
        
        X = X.fillna(0)
        
        if self.feature_selector and self.selected_features:
            available_features = [f for f in self.selected_features if f in X.columns]
            X_selected = X[available_features].copy()
            
            for feature in self.selected_features:
                if feature not in X_selected.columns:
                    X_selected[feature] = 0
            
            X_selected = X_selected[self.selected_features]
        else:
            X_selected = X
        
        X_scaled = self.scaler.transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
        
        processed_df = enhanced_df.copy()
        for i, col in enumerate(X_selected.columns):
            processed_df[f'{col}_scaled'] = X_scaled_df.iloc[:, i]
        
        logger.info(f"Current season processing complete: {processed_df.shape}")
        return processed_df

def main():
    
    preprocessor = FPLDataPreprocessor(scaler_type='robust')
    
    try:
        df = preprocessor.load_data("enhanced_players.csv")
        
        cleaned_df = preprocessor.clean_data(df)
        enhanced_df = preprocessor.engineer_features(cleaned_df)
        
        data_dict = preprocessor.prepare_modeling_data(enhanced_df)
        
        enhanced_df.to_csv("processed_players.csv", index=False)
        
        feature_info = pd.DataFrame({
            'feature': data_dict['feature_names'],
            'importance': preprocessor.feature_selector.scores_[preprocessor.feature_selector.get_support()]
        }).sort_values('importance', ascending=False)
        
        feature_info.to_csv("feature_importance.csv", index=False)
        
        print(f"\n✅ Data preprocessing complete!")
        print(f"📊 Processed players: {len(enhanced_df)}")
        print(f"📈 Features: {len(enhanced_df.columns)}")
        print(f"🎯 Selected features: {len(data_dict['feature_names'])}")
        print(f"🏋️  Training samples: {len(data_dict['X_train'])}")
        print(f"🧪 Test samples: {len(data_dict['X_test'])}")
        
        print(f"\n🏆 Top 10 Most Important Features:")
        print(feature_info.head(10).to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
