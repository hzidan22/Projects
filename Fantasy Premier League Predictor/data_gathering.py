

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FPLDataGatherer:

    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.session = requests.Session()
        
    def fetch_api_data(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        
        try:
            logger.info("Fetching main FPL data...")
            response = self.session.get(f"{self.base_url}bootstrap-static/")
            response.raise_for_status()
            main_data = response.json()
            
            logger.info("Fetching fixtures data...")
            fixtures_response = self.session.get(f"{self.base_url}fixtures/")
            fixtures_response.raise_for_status()
            fixtures_data = fixtures_response.json()
            
            logger.info("API data successfully retrieved!")
            return main_data, fixtures_data
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch API data: {e}")
            return None, None
    
    def calculate_fixture_difficulty(self, team_id: int, fixtures_df: pd.DataFrame, 
                                   teams_df: pd.DataFrame, next_gw: int = 5) -> float:
        
        team_fixtures = fixtures_df[
            ((fixtures_df['team_h'] == team_id) | (fixtures_df['team_a'] == team_id)) &
            (fixtures_df['finished'] == False)
        ].head(next_gw)
        
        if team_fixtures.empty:
            return 3.0
        
        difficulty_scores = []
        team_strength = teams_df[teams_df['id'] == team_id]['strength'].iloc[0] if not teams_df[teams_df['id'] == team_id].empty else 3
        
        for _, fixture in team_fixtures.iterrows():
            if fixture['team_h'] == team_id:
                opponent_id = fixture['team_a']
                is_home = True
            else:
                opponent_id = fixture['team_h']
                is_home = False
            
            opponent_strength = teams_df[teams_df['id'] == opponent_id]['strength'].iloc[0] if not teams_df[teams_df['id'] == opponent_id].empty else 3
            
            base_difficulty = min(5, max(1, opponent_strength - team_strength + 3))
            
            if is_home:
                difficulty = max(1, base_difficulty - 0.5)
            else:
                difficulty = min(5, base_difficulty + 0.5)
            
            difficulty_scores.append(difficulty)
        
        return np.mean(difficulty_scores) if difficulty_scores else 3.0
    
    def calculate_team_strength(self, team_id: int, players_df: pd.DataFrame) -> float:
        
        team_players = players_df[players_df['team'] == team_id]
        if team_players.empty:
            return 3.0
        
        total_value = team_players['now_cost'].sum()
        total_points = team_players['total_points'].sum()
        avg_selected = team_players['selected_by_percent'].mean()
        
        strength = min(5, max(1, (total_points / 100) + (total_value / 200) + (avg_selected / 20)))
        return strength
    
    def enhance_player_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame, 
                           fixtures_df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Enhancing player data with advanced features...")
        
        players_df['points_per_minute'] = players_df['total_points'] / players_df['minutes'].replace(0, 1)
        players_df['goals_per_minute'] = players_df['goals_scored'] / players_df['minutes'].replace(0, 1)
        players_df['assists_per_minute'] = players_df['assists'] / players_df['minutes'].replace(0, 1)
        
        position_mapping = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        players_df['position'] = players_df['element_type'].map(position_mapping)
        
        team_names = teams_df.set_index('id')['name'].to_dict()
        players_df['team_name'] = players_df['team'].map(team_names)
        
        team_strengths = {}
        for team_id in teams_df['id']:
            team_strengths[team_id] = self.calculate_team_strength(team_id, players_df)
        
        teams_df['strength'] = teams_df['id'].map(team_strengths)
        
        players_df['fixture_difficulty'] = players_df['team'].apply(
            lambda x: self.calculate_fixture_difficulty(x, fixtures_df, teams_df)
        )
        
        players_df['defensive_value'] = (
            players_df['clean_sheets'] * 4 + 
            players_df['goals_conceded'] * -0.5 +
            players_df['bonus']
        )
        
        players_df['attacking_value'] = (
            players_df['goals_scored'] * 6 + 
            players_df['assists'] * 3 +
            players_df['bonus'] +
            players_df['bps'] * 0.1
        )
        
        players_df['form_score'] = players_df['total_points'] / (players_df['minutes'] / 90).replace(0, 1)
        players_df['value_score'] = players_df['total_points'] / (players_df['now_cost'] / 10).replace(0, 0.1)
        
        players_df['captain_score'] = (
            players_df['total_points'] * 0.4 +
            players_df['form_score'] * 0.3 +
            players_df['value_score'] * 0.2 +
            (5 - players_df['fixture_difficulty']) * 10 * 0.1
        )
        
        players_df['expected_ppg'] = players_df['total_points'] / (players_df['minutes'] / 90).replace(0, 1)
        
        logger.info(f"Enhanced {len(players_df)} players with {len(players_df.columns)} features")
        return players_df
    
    def gather_and_process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        logger.info("Starting enhanced FPL data gathering...")
        
        main_data, fixtures_data = self.fetch_api_data()
        if not main_data or not fixtures_data:
            raise Exception("Failed to fetch API data")
        
        players_df = pd.DataFrame(main_data['elements'])
        teams_df = pd.DataFrame(main_data['teams'])
        fixtures_df = pd.DataFrame(fixtures_data)
        
        players_df['now_cost'] = players_df['now_cost'] / 10
        
        numeric_columns = ['creativity', 'influence', 'threat', 'ict_index', 'selected_by_percent']
        for col in numeric_columns:
            if col in players_df.columns:
                players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
        
        enhanced_players_df = self.enhance_player_data(players_df, teams_df, fixtures_df)
        
        enhanced_players_df = enhanced_players_df[enhanced_players_df['minutes'] > 0].copy()
        
        enhanced_players_df = enhanced_players_df[enhanced_players_df['element_type'] != 5].copy()
        
        logger.info("Data gathering and processing complete!")
        return enhanced_players_df, teams_df, fixtures_df
    
    def save_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame, 
                  fixtures_df: pd.DataFrame, prefix: str = "enhanced_") -> None:
        
        players_df.to_csv(f"{prefix}players.csv", index=False)
        teams_df.to_csv(f"{prefix}teams.csv", index=False)
        fixtures_df.to_csv(f"{prefix}fixtures.csv", index=False)
        logger.info(f"Data saved with prefix '{prefix}'")

def main():
    
    gatherer = FPLDataGatherer()
    
    try:
        players_df, teams_df, fixtures_df = gatherer.gather_and_process_data()
        gatherer.save_data(players_df, teams_df, fixtures_df)
        
        print(f"\n✅ Data gathering complete!")
        print(f"📊 Players: {len(players_df)}")
        print(f"🏟️  Teams: {len(teams_df)}")
        print(f"📅 Fixtures: {len(fixtures_df)}")
        print(f"📈 Features: {len(players_df.columns)}")
        
        print(f"\n🎯 Top 10 Captain Candidates:")
        top_captains = players_df.nlargest(10, 'captain_score')[
            ['first_name', 'second_name', 'position', 'team_name', 'captain_score', 'now_cost']
        ]
        print(top_captains.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error in data gathering: {e}")
        raise

if __name__ == "__main__":
    main()
