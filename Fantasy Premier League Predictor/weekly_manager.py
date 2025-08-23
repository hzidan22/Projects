import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from data_gathering import FPLDataGatherer
from data_preprocessing import FPLDataPreprocessor
from model_training import FPLModelTrainer
from team_selection import FPLTeamSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FPLWeeklyManager:
    
    def __init__(self):
        self.data_gatherer = FPLDataGatherer()
        self.preprocessor = FPLDataPreprocessor()
        self.trainer = FPLModelTrainer()
        self.team_selector = FPLTeamSelector()
        self.current_team = None
        self.team_file = "current_team.json"
        self.history_file = "transfer_history.json"
        
    def get_current_gameweek(self) -> Dict:
        try:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(url)
            data = response.json()
            
            current_gw = None
            next_gw = None
            
            for gw in data['events']:
                if gw['is_current']:
                    current_gw = gw
                elif gw['is_next']:
                    next_gw = gw
                    
            return {
                'current': current_gw,
                'next': next_gw,
                'all_gameweeks': data['events']
            }
        except Exception as e:
            logger.error(f"Failed to get gameweek info: {e}")
            return {}
    
    def get_gameweek_fixtures(self, gameweek: int) -> pd.DataFrame:
        try:
            url = "https://fantasy.premierleague.com/api/fixtures/"
            response = requests.get(url)
            fixtures = response.json()
            
            gw_fixtures = [f for f in fixtures if f['event'] == gameweek]
            return pd.DataFrame(gw_fixtures)
        except Exception as e:
            logger.error(f"Failed to get fixtures: {e}")
            return pd.DataFrame()
    
    def calculate_gameweek_difficulty(self, player_team: int, gameweek: int, 
                                    fixtures_df: pd.DataFrame, teams_df: pd.DataFrame) -> float:
        player_fixtures = fixtures_df[
            (fixtures_df['team_h'] == player_team) | 
            (fixtures_df['team_a'] == player_team)
        ]
        
        if player_fixtures.empty:
            return 5.0
        
        difficulties = []
        for _, fixture in player_fixtures.iterrows():
            if fixture['team_h'] == player_team:
                opponent_id = fixture['team_a']
                is_home = True
            else:
                opponent_id = fixture['team_h']
                is_home = False
            
            opponent_strength = teams_df[teams_df['id'] == opponent_id]['strength'].iloc[0] if not teams_df[teams_df['id'] == opponent_id].empty else 3
            
            base_difficulty = min(5, max(1, opponent_strength))
            
            if is_home:
                difficulty = max(1, base_difficulty - 0.5)
            else:
                difficulty = min(5, base_difficulty + 0.5)
                
            difficulties.append(difficulty)
        
        return np.mean(difficulties) if difficulties else 3.0
    
    def predict_gameweek_points(self, players_df: pd.DataFrame, gameweek: int) -> pd.DataFrame:
        logger.info(f"Predicting points for gameweek {gameweek}")
        
        gw_info = self.get_current_gameweek()
        fixtures_df = self.get_gameweek_fixtures(gameweek)
        
        if fixtures_df.empty:
            logger.warning(f"No fixtures found for gameweek {gameweek}")
            players_df['gw_predicted_points'] = players_df.get('predicted_points', 0) * 0.8
            return players_df
        
        enhanced_players = players_df.copy()
        
        teams_df = pd.DataFrame()
        try:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(url)
            data = response.json()
            teams_df = pd.DataFrame(data['teams'])
        except:
            pass
        
        enhanced_players['gw_fixture_difficulty'] = enhanced_players['team'].apply(
            lambda x: self.calculate_gameweek_difficulty(x, gameweek, fixtures_df, teams_df)
        )
        
        enhanced_players['has_fixture'] = enhanced_players['team'].apply(
            lambda x: len(fixtures_df[
                (fixtures_df['team_h'] == x) | (fixtures_df['team_a'] == x)
            ]) > 0
        )
        
        enhanced_players['fixture_count'] = enhanced_players['team'].apply(
            lambda x: len(fixtures_df[
                (fixtures_df['team_h'] == x) | (fixtures_df['team_a'] == x)
            ])
        )
        
        base_points = enhanced_players.get('predicted_points', 0)
        
        fixture_multiplier = enhanced_players.apply(lambda x: 
            0 if not x['has_fixture'] else
            (6 - x['gw_fixture_difficulty']) / 5 * x['fixture_count'], axis=1
        )
        
        form_factor = enhanced_players.get('form_score', base_points) / base_points.replace(0, 1)
        form_factor = form_factor.fillna(1.0).clip(0.5, 2.0)
        
        position_gw_bonus = enhanced_players['position'].map({
            'FWD': 1.1,
            'MID': 1.05, 
            'DEF': 1.0,
            'GK': 0.95
        }).fillna(1.0)
        
        enhanced_players['gw_predicted_points'] = (
            base_points * fixture_multiplier * form_factor * position_gw_bonus
        ).clip(0, None)
        
        enhanced_players['gw_predicted_points'] = enhanced_players['gw_predicted_points'].fillna(0)
        
        logger.info(f"Gameweek predictions complete. Average points: {enhanced_players['gw_predicted_points'].mean():.2f}")
        
        return enhanced_players
    
    def load_current_team(self) -> Optional[pd.DataFrame]:
        try:
            with open(self.team_file, 'r') as f:
                team_data = json.load(f)
            return pd.DataFrame(team_data['players'])
        except FileNotFoundError:
            logger.info("No current team found")
            return None
        except Exception as e:
            logger.error(f"Error loading team: {e}")
            return None
    
    def save_current_team(self, team_df: pd.DataFrame, gameweek: int):
        team_data = {
            'gameweek': gameweek,
            'timestamp': datetime.now().isoformat(),
            'players': team_df.to_dict('records'),
            'total_cost': team_df['now_cost'].sum()
        }
        
        with open(self.team_file, 'w') as f:
            json.dump(team_data, f, indent=2)
        
        logger.info(f"Team saved for gameweek {gameweek}")
    
    def save_transfer_history(self, transfer_info: Dict):
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        history.append(transfer_info)
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def suggest_transfer(self, current_team: pd.DataFrame, all_players: pd.DataFrame, 
                        gameweek: int) -> Dict:
        logger.info(f"Analyzing transfer options for gameweek {gameweek}")
        
        all_players_gw = self.predict_gameweek_points(all_players, gameweek)
        current_team_gw = self.predict_gameweek_points(current_team, gameweek)
        
        transfer_suggestions = []
        
        for _, current_player in current_team_gw.iterrows():
            position = current_player['position']
            current_cost = current_player['now_cost']
            current_gw_points = current_player['gw_predicted_points']
            
            budget_range = 2.0
            
            alternatives = all_players_gw[
                (all_players_gw['position'] == position) &
                (all_players_gw['now_cost'] <= current_cost + budget_range) &
                (all_players_gw['gw_predicted_points'] > current_gw_points) &
                (~all_players_gw.index.isin(current_team.index))
            ].sort_values('gw_predicted_points', ascending=False)
            
            if not alternatives.empty:
                best_alternative = alternatives.iloc[0]
                
                point_gain = best_alternative['gw_predicted_points'] - current_gw_points
                cost_diff = best_alternative['now_cost'] - current_cost
                
                if point_gain > 1.0:
                    transfer_suggestions.append({
                        'out': {
                            'name': f"{current_player['first_name']} {current_player['second_name']}",
                            'position': current_player['position'],
                            'team': current_player.get('team_name', f"Team {current_player['team']}"),
                            'cost': current_player['now_cost'],
                            'gw_points': current_gw_points,
                            'season_points': current_player.get('predicted_points', 0)
                        },
                        'in': {
                            'name': f"{best_alternative['first_name']} {best_alternative['second_name']}",
                            'position': best_alternative['position'],
                            'team': best_alternative.get('team_name', f"Team {best_alternative['team']}"),
                            'cost': best_alternative['now_cost'],
                            'gw_points': best_alternative['gw_predicted_points'],
                            'season_points': best_alternative.get('predicted_points', 0)
                        },
                        'gw_point_gain': point_gain,
                        'cost_difference': cost_diff,
                        'value_score': point_gain / max(abs(cost_diff), 0.1)
                    })
        
        transfer_suggestions.sort(key=lambda x: x['gw_point_gain'], reverse=True)
        
        return {
            'gameweek': gameweek,
            'suggestions': transfer_suggestions[:5],
            'has_transfers': len(transfer_suggestions) > 0
        }
    
    def execute_transfer(self, current_team: pd.DataFrame, player_out_name: str, 
                        player_in_name: str, all_players: pd.DataFrame, gameweek: int) -> Dict:
        
        current_team = current_team.copy()
        
        player_out_idx = None
        for idx, player in current_team.iterrows():
            full_name = f"{player['first_name']} {player['second_name']}"
            if full_name == player_out_name:
                player_out_idx = idx
                break
        
        if player_out_idx is None:
            return {'success': False, 'error': f'Player {player_out_name} not found in team'}
        
        player_in_candidates = all_players[
            (all_players['first_name'] + ' ' + all_players['second_name']) == player_in_name
        ]
        
        if player_in_candidates.empty:
            return {'success': False, 'error': f'Player {player_in_name} not found'}
        
        player_in = player_in_candidates.iloc[0]
        player_out = current_team.loc[player_out_idx]
        
        if player_out['position'] != player_in['position']:
            return {'success': False, 'error': 'Players must be in same position'}
        
        cost_diff = player_in['now_cost'] - player_out['now_cost']
        new_total_cost = current_team['now_cost'].sum() - player_out['now_cost'] + player_in['now_cost']
        
        if new_total_cost > 100.0:
            return {'success': False, 'error': f'Transfer exceeds budget. Would cost £{new_total_cost:.1f}M'}
        
        team_counts = current_team['team'].value_counts()
        if player_out['team'] != player_in['team']:
            if team_counts.get(player_in['team'], 0) >= 3:
                return {'success': False, 'error': 'Cannot have more than 3 players from same team'}
        
        current_team.loc[player_out_idx] = player_in
        
        transfer_info = {
            'gameweek': gameweek,
            'timestamp': datetime.now().isoformat(),
            'player_out': player_out_name,
            'player_in': player_in_name,
            'cost_change': cost_diff,
            'new_total_cost': new_total_cost
        }
        
        self.save_transfer_history(transfer_info)
        self.save_current_team(current_team, gameweek)
        
        return {
            'success': True,
            'transfer': transfer_info,
            'new_team': current_team
        }
    
    def weekly_update(self, gameweek: int = None) -> Dict:
        logger.info("Starting weekly FPL update...")
        
        gw_info = self.get_current_gameweek()
        if not gameweek:
            gameweek = gw_info.get('next', {}).get('id', 1)
        
        players_df, teams_df, fixtures_df = self.data_gatherer.gather_and_process_data()
        
        cleaned_df = self.preprocessor.clean_data(players_df)
        enhanced_df = self.preprocessor.engineer_features(cleaned_df)
        
        if os.path.exists("models/best_fpl_model.joblib"):
            model = self.trainer.load_model("models/best_fpl_model.joblib")
            enhanced_df['predicted_points'] = enhanced_df['total_points'] * 1.1
        else:
            data_dict = self.preprocessor.prepare_modeling_data(enhanced_df)
            self.trainer.train_all_models(
                data_dict['X_train'], data_dict['y_train'],
                data_dict['X_test'], data_dict['y_test']
            )
            self.trainer.save_model("models/best_fpl_model.joblib")
            predictions = self.trainer.predict_player_points(data_dict['X_full'])
            enhanced_df['predicted_points'] = predictions
        
        current_team = self.load_current_team()
        
        if current_team is None:
            logger.info("No current team found. Selecting initial team...")
            team_result = self.team_selector.select_optimal_team(enhanced_df, include_captain=True)
            self.save_current_team(team_result['full_squad'], gameweek)
            
            return {
                'action': 'initial_selection',
                'gameweek': gameweek,
                'team': team_result,
                'message': 'Initial team selected'
            }
        else:
            logger.info("Analyzing transfer options...")
            transfer_analysis = self.suggest_transfer(current_team, enhanced_df, gameweek)
            
            current_team_gw = self.predict_gameweek_points(current_team, gameweek)
            
            return {
                'action': 'transfer_analysis',
                'gameweek': gameweek,
                'current_team': current_team_gw,
                'transfer_suggestions': transfer_analysis,
                'message': f'Transfer analysis complete for gameweek {gameweek}'
            }

def main():
    import sys
    import os
    
    manager = FPLWeeklyManager()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'transfer':
            if len(sys.argv) != 4:
                print("Usage: python weekly_manager.py transfer 'Player Out' 'Player In'")
                return
            
            current_team = manager.load_current_team()
            if current_team is None:
                print("No current team found. Run weekly update first.")
                return
            
            players_df, _, _ = manager.data_gatherer.gather_and_process_data()
            result = manager.execute_transfer(
                current_team, sys.argv[2], sys.argv[3], players_df, 
                manager.get_current_gameweek().get('next', {}).get('id', 1)
            )
            
            if result['success']:
                print(f"✅ Transfer completed: {sys.argv[2]} → {sys.argv[3]}")
                print(f"💰 Cost change: £{result['transfer']['cost_change']:+.1f}M")
                print(f"💰 New total: £{result['transfer']['new_total_cost']:.1f}M")
            else:
                print(f"❌ Transfer failed: {result['error']}")
                
        elif sys.argv[1] == 'update':
            gameweek = int(sys.argv[2]) if len(sys.argv) > 2 else None
            result = manager.weekly_update(gameweek)
            
            print(f"🚀 Weekly update complete for GW{result['gameweek']}")
            
            if result['action'] == 'initial_selection':
                team = result['team']['full_squad']
                captain = result['team']['captain_info']['captain']['name']
                print(f"👑 Captain: {captain}")
                print(f"💰 Team cost: £{team['now_cost'].sum():.1f}M")
                
            elif result['action'] == 'transfer_analysis':
                suggestions = result['transfer_suggestions']['suggestions']
                if suggestions:
                    print(f"\n🔄 Transfer suggestions:")
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        print(f"{i}. {suggestion['out']['name']} → {suggestion['in']['name']}")
                        print(f"   GW{result['gameweek']} gain: +{suggestion['gw_point_gain']:.1f} pts")
                        print(f"   Cost: £{suggestion['cost_difference']:+.1f}M")
                else:
                    print("✅ No beneficial transfers found")
    else:
        result = manager.weekly_update()
        print("Weekly update completed. Use 'update' or 'transfer' commands.")

if __name__ == "__main__":
    main()
