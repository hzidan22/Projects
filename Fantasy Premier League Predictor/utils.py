

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests
from config import Config

def setup_logging(log_level: str = None) -> logging.Logger:
    
    if log_level is None:
        log_level = Config.LOG_LEVEL
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fpl_predictor.log')
        ]
    )
    
    return logging.getLogger(__name__)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    
    return numerator / denominator if denominator != 0 else default

def normalize_to_range(value: float, min_val: float, max_val: float, 
                      new_min: float = 0.0, new_max: float = 1.0) -> float:
    
    if max_val == min_val:
        return new_min
    
    normalized = (value - min_val) / (max_val - min_val)
    return new_min + normalized * (new_max - new_min)

def calculate_percentile_rank(value: float, series: pd.Series) -> float:
    
    return (series <= value).mean() * 100

def format_currency(amount: float, currency: str = "£", decimals: int = 1) -> str:
    
    return f"{currency}{amount:.{decimals}f}M"

def format_percentage(value: float, decimals: int = 1) -> str:
    
    return f"{value:.{decimals}f}%"

def create_player_name(first_name: str, second_name: str) -> str:
    
    return f"{first_name} {second_name}".strip()

def get_position_emoji(position: str) -> str:
    
    position_emojis = {
        'GK': '🥅',
        'DEF': '🛡️',
        'MID': '⚽',
        'FWD': '🎯'
    }
    return position_emojis.get(position, '⚽')

def calculate_form_trend(recent_points: List[float], historical_points: List[float]) -> float:
    
    if not recent_points or not historical_points:
        return 0.0
    
    recent_avg = np.mean(recent_points)
    historical_avg = np.mean(historical_points)
    
    if historical_avg == 0:
        return 0.0
    
    return (recent_avg - historical_avg) / historical_avg

def validate_team_constraints(team_df: pd.DataFrame, config: Config = Config) -> Dict[str, bool]:
    
    constraints = {
        'squad_size': len(team_df) == config.SQUAD_SIZE,
        'budget': team_df['now_cost'].sum() <= config.BUDGET,
        'max_per_team': all(
            team_df[team_df['team'] == team_id].shape[0] <= config.MAX_PLAYERS_PER_TEAM
            for team_id in team_df['team'].unique()
        )
    }
    
    for position, limits in config.POSITION_LIMITS.items():
        position_count = len(team_df[team_df['position'] == position])
        constraints[f'{position}_count'] = (
            limits['min'] <= position_count <= limits['max']
        )
    
    constraints['all_valid'] = all(constraints.values())
    return constraints

def save_json(data: Dict, filename: str, directory: str = None) -> None:
    
    if directory is None:
        directory = Config.OUTPUT_DIR
    
    filepath = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filename: str, directory: str = None) -> Dict:
    
    if directory is None:
        directory = Config.OUTPUT_DIR
    
    filepath = os.path.join(directory, filename)
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_gameweek_info() -> Dict:
    
    try:
        response = requests.get(f"{Config.FPL_BASE_URL}bootstrap-static/", 
                              timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        current_gw = None
        next_gw = None
        
        for gw in data['events']:
            if gw['is_current']:
                current_gw = gw
            elif gw['is_next']:
                next_gw = gw
        
        return {
            'current_gameweek': current_gw,
            'next_gameweek': next_gw,
            'total_gameweeks': len(data['events'])
        }
    
    except Exception as e:
        logging.error(f"Failed to get gameweek info: {e}")
        return {}

def calculate_expected_points_with_captain(starting_xi: pd.DataFrame, 
                                         captain_name: str) -> float:
    
    base_points = starting_xi['predicted_points'].sum()
    
    captain_row = starting_xi[
        (starting_xi['first_name'] + ' ' + starting_xi['second_name']) == captain_name
    ]
    
    if not captain_row.empty:
        captain_points = captain_row['predicted_points'].iloc[0]
        return base_points + captain_points
    
    return base_points

def generate_transfer_suggestions(current_team: pd.DataFrame, 
                                all_players: pd.DataFrame,
                                max_suggestions: int = 5) -> List[Dict]:
    
    suggestions = []
    
    current_team_sorted = current_team.sort_values('predicted_points')
    
    for _, weak_player in current_team_sorted.head(max_suggestions).iterrows():
        position = weak_player['position']
        budget = weak_player['now_cost'] + 2.0  # Allow £2M upgrade
        
        alternatives = all_players[
            (all_players['position'] == position) &
            (all_players['now_cost'] <= budget) &
            (all_players['predicted_points'] > weak_player['predicted_points']) &
            (~all_players.index.isin(current_team.index))
        ].sort_values('predicted_points', ascending=False)
        
        if not alternatives.empty:
            best_alternative = alternatives.iloc[0]
            point_gain = best_alternative['predicted_points'] - weak_player['predicted_points']
            cost_diff = best_alternative['now_cost'] - weak_player['now_cost']
            
            suggestions.append({
                'out': {
                    'name': create_player_name(weak_player['first_name'], weak_player['second_name']),
                    'position': weak_player['position'],
                    'cost': weak_player['now_cost'],
                    'predicted_points': weak_player['predicted_points']
                },
                'in': {
                    'name': create_player_name(best_alternative['first_name'], best_alternative['second_name']),
                    'position': best_alternative['position'],
                    'cost': best_alternative['now_cost'],
                    'predicted_points': best_alternative['predicted_points']
                },
                'point_gain': point_gain,
                'cost_difference': cost_diff,
                'value_score': point_gain / max(cost_diff, 0.1)
            })
    
    return sorted(suggestions, key=lambda x: x['value_score'], reverse=True)

def create_performance_summary(team_result: Dict) -> Dict:
    
    starting_xi = team_result['starting_xi']
    captain_info = team_result.get('captain_info', {})
    
    total_cost = team_result['total_cost']
    total_predicted = starting_xi['predicted_points'].sum()
    
    captain_bonus = 0
    if captain_info and 'captain' in captain_info:
        captain_bonus = captain_info['captain']['predicted_points']
    
    position_stats = {}
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = starting_xi[starting_xi['position'] == position]
        position_stats[position] = {
            'count': len(pos_players),
            'total_cost': pos_players['now_cost'].sum(),
            'total_points': pos_players['predicted_points'].sum(),
            'avg_cost': pos_players['now_cost'].mean() if len(pos_players) > 0 else 0,
            'avg_points': pos_players['predicted_points'].mean() if len(pos_players) > 0 else 0
        }
    
    return {
        'total_cost': total_cost,
        'remaining_budget': Config.BUDGET - total_cost,
        'budget_utilization': (total_cost / Config.BUDGET) * 100,
        'expected_points': total_predicted,
        'expected_points_with_captain': total_predicted + captain_bonus,
        'points_per_million': total_predicted / total_cost if total_cost > 0 else 0,
        'captain_bonus': captain_bonus,
        'position_breakdown': position_stats,
        'team_count': len(team_result.get('team_distribution', {})),
        'max_from_single_team': max(team_result.get('team_distribution', {}).values()) if team_result.get('team_distribution') else 0
    }

def print_colored_output(text: str, color: str = 'white') -> None:
    
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'end': '\033[0m'
    }
    
    color_code = colors.get(color.lower(), colors['white'])
    print(f"{color_code}{text}{colors['end']}")

def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict:
    
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_columns': [],
        'columns_with_nulls': {},
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    for col in required_columns:
        if col not in df.columns:
            quality_report['missing_columns'].append(col)
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            quality_report['columns_with_nulls'][col] = {
                'null_count': null_count,
                'null_percentage': (null_count / len(df)) * 100
            }
    
    quality_report['is_valid'] = (
        len(quality_report['missing_columns']) == 0 and
        quality_report['duplicate_rows'] == 0
    )
    
    return quality_report

def backup_file(filepath: str) -> str:
    
    if not os.path.exists(filepath):
        return ""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    
    import shutil
    shutil.copy2(filepath, backup_path)
    
    return backup_path
