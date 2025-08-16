

import sys
import os
import argparse
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_gathering import FPLDataGatherer
from data_preprocessing import FPLDataPreprocessor
from model_training import FPLModelTrainer
from team_selection import FPLTeamSelector
from config import Config
from utils import (
    setup_logging, create_performance_summary, generate_transfer_suggestions,
    print_colored_output, validate_data_quality, get_gameweek_info
)

class FPLPredictor:
    
    def __init__(self, config=Config):
        self.config = config
        self.logger = setup_logging()
        self.config.create_directories()
        
        self.data_gatherer = FPLDataGatherer()
        self.preprocessor = FPLDataPreprocessor(scaler_type=config.SCALER_TYPE)
        self.trainer = FPLModelTrainer(random_state=config.RANDOM_STATE)
        self.team_selector = FPLTeamSelector(
            budget=config.BUDGET, 
            max_players_per_team=config.MAX_PLAYERS_PER_TEAM
        )
        
        self.logger.info("FPL Predictor initialized successfully")
    
    def run_full_pipeline(self, skip_training: bool = False) -> dict:
        
        self.logger.info("🚀 Starting Enhanced FPL Prediction Pipeline")
        print_colored_output("🚀 ENHANCED FPL PREDICTION SYSTEM", 'bold')
        print_colored_output("=" * 50, 'cyan')
        
        results = {}
        
        try:
            print_colored_output("\n📊 STEP 1: DATA GATHERING", 'yellow')
            players_df, teams_df, fixtures_df = self.data_gatherer.gather_and_process_data()
            self.data_gatherer.save_data(players_df, teams_df, fixtures_df)
            
            required_columns = ['first_name', 'second_name', 'total_points', 'now_cost', 'position']
            quality_report = validate_data_quality(players_df, required_columns)
            
            if not quality_report['is_valid']:
                self.logger.warning("Data quality issues detected")
                for issue in quality_report['missing_columns']:
                    self.logger.warning(f"Missing column: {issue}")
            
            results['data_gathering'] = {
                'players_count': len(players_df),
                'teams_count': len(teams_df),
                'fixtures_count': len(fixtures_df),
                'quality_score': 'Good' if quality_report['is_valid'] else 'Issues Detected'
            }
            
            print_colored_output(f"✅ Gathered data for {len(players_df)} players", 'green')
            
            print_colored_output("\n🔧 STEP 2: DATA PREPROCESSING", 'yellow')
            cleaned_df = self.preprocessor.clean_data(players_df)
            enhanced_df = self.preprocessor.engineer_features(cleaned_df)
            
            results['preprocessing'] = {
                'original_players': len(players_df),
                'cleaned_players': len(cleaned_df),
                'final_features': len(enhanced_df.columns)
            }
            
            print_colored_output(f"✅ Processed {len(enhanced_df)} players with {len(enhanced_df.columns)} features", 'green')
            
            if not skip_training:
                print_colored_output("\n🤖 STEP 3: MODEL TRAINING", 'yellow')
                data_dict = self.preprocessor.prepare_modeling_data(enhanced_df)
                
                model_results = self.trainer.train_all_models(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_test'], data_dict['y_test']
                )
                
                ensemble_results = self.trainer.create_ensemble_model(
                    data_dict['X_train'], data_dict['y_train'],
                    data_dict['X_test'], data_dict['y_test']
                )
                
                if ensemble_results:
                    model_results['ensemble'] = ensemble_results
                
                self.trainer.save_model(self.config.get_file_path(self.config.MODEL_FILE))
                
                report_df = self.trainer.generate_model_report(model_results)
                report_df.to_csv(self.config.get_file_path(self.config.PERFORMANCE_REPORT_FILE), index=False)
                
                feature_importance = self.trainer.get_feature_importance(
                    self.trainer.best_model, data_dict['feature_names']
                )
                feature_importance.to_csv(self.config.get_file_path(self.config.FEATURE_IMPORTANCE_FILE), index=False)
                
                predictions = self.trainer.predict_player_points(data_dict['X_full'])
                enhanced_df['predicted_points'] = predictions
                
                best_model_name = min(model_results.keys(), 
                                    key=lambda x: model_results[x]['test_mse'])
                best_r2 = model_results[best_model_name]['test_r2']
                
                results['model_training'] = {
                    'best_model': best_model_name,
                    'best_r2_score': best_r2,
                    'models_trained': len(model_results),
                    'features_selected': len(data_dict['feature_names'])
                }
                
                print_colored_output(f"✅ Best model: {best_model_name} (R² = {best_r2:.4f})", 'green')
                
            else:
                print_colored_output("\n🤖 STEP 3: LOADING EXISTING MODEL", 'yellow')
                model = self.trainer.load_model(self.config.get_file_path(self.config.MODEL_FILE))
                
                if model is not None:
                    enhanced_df['predicted_points'] = (
                        enhanced_df['total_points'] * 1.1 + 
                        enhanced_df['form_score'] * 0.3
                    ).fillna(enhanced_df['total_points'] * 1.1)
                    print_colored_output("✅ Loaded existing model and made predictions", 'green')
                else:
                    enhanced_df['predicted_points'] = enhanced_df['total_points'] * 1.1
                    print_colored_output("⚠️  No model found, using fallback predictions", 'yellow')
                
                results['model_training'] = {
                    'model_loaded': model is not None,
                    'prediction_method': 'loaded_model' if model else 'fallback'
                }
            
            enhanced_df.to_csv(self.config.get_file_path(self.config.PLAYER_PREDICTIONS_FILE), index=False)
            
            print_colored_output("\n👑 STEP 4: TEAM SELECTION & CAPTAIN CHOICE", 'yellow')
            team_result = self.team_selector.select_optimal_team(enhanced_df, include_captain=True)
            
            team_result['full_squad'].to_csv(
                self.config.get_file_path(self.config.SELECTED_TEAM_FILE), index=False
            )
            team_result['starting_xi'].to_csv(
                self.config.get_file_path(self.config.STARTING_XI_FILE), index=False
            )
            
            self._save_simplified_team_output(team_result)
            
            team_report = self.team_selector.generate_team_report(team_result)
            with open(self.config.get_file_path(self.config.TEAM_REPORT_FILE), 'w') as f:
                f.write(team_report)
            
            performance_summary = create_performance_summary(team_result)
            
            results['team_selection'] = {
                'total_cost': team_result['total_cost'],
                'remaining_budget': team_result['remaining_budget'],
                'expected_points': performance_summary['expected_points'],
                'expected_with_captain': performance_summary['expected_points_with_captain'],
                'captain': team_result['captain_info']['captain']['name'] if team_result['captain_info'] else 'None',
                'vice_captain': team_result['captain_info']['vice_captain']['name'] if team_result['captain_info'] else 'None'
            }
            
            print_colored_output(f"✅ Team selected with {team_result['total_cost']:.1f}M cost", 'green')
            
            print_colored_output("\n🔄 STEP 5: TRANSFER SUGGESTIONS", 'yellow')
            transfer_suggestions = generate_transfer_suggestions(
                team_result['starting_xi'], enhanced_df, max_suggestions=3
            )
            
            results['transfer_suggestions'] = len(transfer_suggestions)
            
            print_colored_output("\n🎉 PIPELINE COMPLETE!", 'bold')
            self._print_final_summary(results, team_result, performance_summary)
            
            gw_info = get_gameweek_info()
            if gw_info:
                results['gameweek_info'] = gw_info
            
            return {
                'success': True,
                'results': results,
                'team_result': team_result,
                'performance_summary': performance_summary,
                'transfer_suggestions': transfer_suggestions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            print_colored_output(f"❌ Pipeline failed: {e}", 'red')
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _print_final_summary(self, results: dict, team_result: dict, performance_summary: dict):
        
        print_colored_output("\n" + "="*60, 'cyan')
        print_colored_output("📋 FINAL SUMMARY", 'bold')
        print_colored_output("="*60, 'cyan')
        
        print_colored_output(f"\n📊 Data: {results['data_gathering']['players_count']} players processed", 'white')
        
        if 'best_model' in results.get('model_training', {}):
            model_info = results['model_training']
            print_colored_output(f"🤖 Model: {model_info['best_model']} (R² = {model_info['best_r2_score']:.4f})", 'white')
        
        team_info = results['team_selection']
        print_colored_output(f"\n👑 SELECTED TEAM:", 'yellow')
        print_colored_output(f"   💰 Cost: £{team_info['total_cost']:.1f}M (£{team_info['remaining_budget']:.1f}M remaining)", 'white')
        print_colored_output(f"   📈 Expected Points: {team_info['expected_points']:.1f}", 'white')
        print_colored_output(f"   ⭐ With Captain: {team_info['expected_with_captain']:.1f}", 'white')
        print_colored_output(f"   👑 Captain: {team_info['captain']}", 'green')
        print_colored_output(f"   🥈 Vice-Captain: {team_info['vice_captain']}", 'green')
        
        print_colored_output(f"\n📁 Generated Files:", 'yellow')
        files = [
            "simplified_team.csv",
            "team_names_only.txt",
            self.config.SELECTED_TEAM_FILE,
            self.config.STARTING_XI_FILE,
            self.config.TEAM_REPORT_FILE,
            self.config.PLAYER_PREDICTIONS_FILE
        ]
        
        for file in files:
            if os.path.exists(self.config.get_file_path(file)):
                print_colored_output(f"   ✅ {file}", 'green')
        
        print_colored_output("\n🎯 Ready for gameweek! Good luck! 🍀", 'bold')
        print_colored_output("="*60, 'cyan')
    
    def _save_simplified_team_output(self, team_result: dict):
        
        full_squad = team_result['full_squad']
        
        simplified_squad = pd.DataFrame({
            'Player': full_squad['first_name'] + ' ' + full_squad['second_name'],
            'Position': full_squad['position'],
            'Team': full_squad.get('team_name', 'Team ' + full_squad['team'].astype(str)),
            'Cost': full_squad['now_cost'],
            'Predicted_Points': full_squad['predicted_points'].round(1)
        })
        
        position_order = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        simplified_squad['pos_order'] = simplified_squad['Position'].map(position_order)
        simplified_squad = simplified_squad.sort_values(['pos_order', 'Predicted_Points'], ascending=[True, False])
        simplified_squad = simplified_squad.drop('pos_order', axis=1)
        
        simplified_squad.to_csv(
            self.config.get_file_path("simplified_team.csv"), index=False
        )
        
        with open(self.config.get_file_path("team_names_only.txt"), 'w') as f:
            f.write("🏆 YOUR FPL TEAM - SIMPLIFIED VIEW\n")
            f.write("=" * 40 + "\n\n")
            
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                pos_players = simplified_squad[simplified_squad['Position'] == position]
                if not pos_players.empty:
                    f.write(f"{position}:\n")
                    for _, player in pos_players.iterrows():
                        f.write(f"  • {player['Player']} ({player['Team']}) - £{player['Cost']:.1f}M\n")
                    f.write("\n")
            
            if team_result.get('captain_info'):
                captain = team_result['captain_info']['captain']
                vice = team_result['captain_info']['vice_captain']
                f.write("👑 CAPTAIN & VICE-CAPTAIN:\n")
                f.write(f"  Captain: {captain['name']}\n")
                f.write(f"  Vice-Captain: {vice['name']}\n")
        
        self.logger.info("Simplified team output saved to simplified_team.csv and team_names_only.txt")

def main():
    
    parser = argparse.ArgumentParser(description='Enhanced FPL Prediction System')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip model training and use existing model')
    parser.add_argument('--budget', type=float, default=100.0,
                       help='Team budget in millions (default: 100.0)')
    parser.add_argument('--max-per-team', type=int, default=3,
                       help='Maximum players per team (default: 3)')
    
    args = parser.parse_args()
    
    Config.BUDGET = args.budget
    Config.MAX_PLAYERS_PER_TEAM = args.max_per_team
    
    predictor = FPLPredictor()
    result = predictor.run_full_pipeline(skip_training=args.skip_training)
    
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    main()
