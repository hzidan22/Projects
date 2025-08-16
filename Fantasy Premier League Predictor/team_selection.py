

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FPLTeamSelector:

    def __init__(self, budget: float = 100.0, max_players_per_team: int = 3):
        self.budget = budget
        self.max_players_per_team = max_players_per_team
        self.position_limits = {
            'GK': {'min': 2, 'max': 2},
            'DEF': {'min': 5, 'max': 5}, 
            'MID': {'min': 5, 'max': 5},
            'FWD': {'min': 3, 'max': 3}
        }
        self.starting_formation = {
            'GK': 1,
            'DEF': 3, 
            'MID': 4,
            'FWD': 3
        }
        
    def calculate_captain_score(self, player_row: pd.Series) -> float:
        
        base_score = player_row.get('predicted_points', 0)
        
        minutes_played = player_row.get('minutes', 0)
        consistency = min(1.0, minutes_played / 2000)
        
        form_score = player_row.get('recent_form', player_row.get('form_score', 0))
        
        fixture_bonus = (5 - player_row.get('fixture_difficulty', 3)) / 5
        
        position_bonus = {
            'GK': 0.8,
            'DEF': 0.9,
            'MID': 1.0,
            'FWD': 1.1
        }.get(player_row.get('position', 'MID'), 1.0)
        
        ownership = player_row.get('selected_by_percent', 0)
        ownership_factor = max(0.8, 1.2 - (ownership / 50))
        
        captain_score = (
            base_score * 0.5 +
            form_score * 0.2 +
            consistency * base_score * 0.15 +
            fixture_bonus * base_score * 0.1 +
            base_score * position_bonus * 0.05
        ) * ownership_factor
        
        return captain_score
    
    def calculate_player_value(self, player_row: pd.Series, captain_potential: bool = False) -> float:
        
        predicted_points = player_row.get('predicted_points', 0)
        cost = player_row.get('now_cost', 0.1)
        
        base_value = predicted_points / cost if cost > 0 else 0
        
        recent_form = player_row.get('recent_form', predicted_points)
        form_multiplier = 1 + (recent_form - predicted_points) / max(predicted_points, 1) * 0.1
        
        fixture_difficulty = player_row.get('fixture_difficulty', 3)
        fixture_multiplier = (6 - fixture_difficulty) / 5
        
        minutes = player_row.get('minutes', 0)
        reliability = min(1.0, minutes / 1500)
        
        captain_bonus = 1.0
        if captain_potential:
            captain_score = self.calculate_captain_score(player_row)
            captain_bonus = 1 + (captain_score / max(predicted_points, 1) - 1) * 0.1
        
        final_value = base_value * form_multiplier * fixture_multiplier * reliability * captain_bonus
        return final_value
    
    def select_optimal_team(self, players_df: pd.DataFrame, include_captain: bool = True) -> Dict:
        
        logger.info("Selecting optimal team...")
        
        players = players_df.copy()
        
        if 'position' not in players.columns and 'element_type' in players.columns:
            position_mapping = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            players['position'] = players['element_type'].map(position_mapping)
        
        players['player_value'] = players.apply(
            lambda x: self.calculate_player_value(x, captain_potential=True), axis=1
        )
        
        if include_captain:
            players['captain_score'] = players.apply(self.calculate_captain_score, axis=1)
        
        players = players.sort_values(['position', 'player_value'], ascending=[True, False])
        
        selected_team = []
        team_counts = defaultdict(int)
        position_counts = defaultdict(int)
        total_cost = 0
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            position_players = players[players['position'] == position]
            needed = self.position_limits[position]['min']
            selected_position = []
            
            for _, player in position_players.iterrows():
                if (len(selected_position) < needed and 
                    team_counts[player['team']] < self.max_players_per_team and
                    total_cost + player['now_cost'] <= self.budget):
                    
                    selected_team.append(player)
                    selected_position.append(player)
                    team_counts[player['team']] += 1
                    position_counts[position] += 1
                    total_cost += player['now_cost']
            
            if len(selected_position) < needed:
                logger.warning(f"Could not fill {position} position completely")
        
        remaining_budget = self.budget - total_cost
        selected_indices = [p.name for p in selected_team]
        remaining_players = players[~players.index.isin(selected_indices)]
        
        for _, player in remaining_players.iterrows():
            if (len(selected_team) < 15 and
                team_counts[player['team']] < self.max_players_per_team and
                position_counts[player['position']] < self.position_limits[player['position']]['max'] and
                player['now_cost'] <= remaining_budget):
                
                selected_team.append(player)
                team_counts[player['team']] += 1
                position_counts[player['position']] += 1
                total_cost += player['now_cost']
                remaining_budget -= player['now_cost']
        
        team_df = pd.DataFrame(selected_team)
        
        starting_xi = self.select_starting_xi(team_df)
        
        captain_info = {}
        if include_captain and not starting_xi.empty:
            captain_info = self.select_captain_and_vice(starting_xi)
        
        return {
            'full_squad': team_df,
            'starting_xi': starting_xi,
            'bench': team_df[~team_df.index.isin(starting_xi.index)],
            'total_cost': total_cost,
            'remaining_budget': self.budget - total_cost,
            'captain_info': captain_info,
            'team_distribution': dict(team_counts),
            'position_distribution': dict(position_counts)
        }
    
    def select_starting_xi(self, squad_df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Selecting starting XI...")
        
        starting_players = []
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            position_players = squad_df[squad_df['position'] == position]
            needed = self.starting_formation[position]
            
            position_players = position_players.sort_values('predicted_points', ascending=False)
            selected = position_players.head(needed)
            starting_players.extend(selected.index.tolist())
        
        starting_xi = squad_df.loc[starting_players]
        return starting_xi
    
    def select_captain_and_vice(self, starting_xi: pd.DataFrame) -> Dict:
        
        logger.info("Selecting captain and vice-captain...")
        
        captain_candidates = starting_xi.copy()
        captain_candidates['captain_score'] = captain_candidates.apply(
            self.calculate_captain_score, axis=1
        )
        
        captain_candidates = captain_candidates.sort_values('captain_score', ascending=False)
        
        captain = captain_candidates.iloc[0]
        
        vice_candidates = captain_candidates.iloc[1:]
        
        different_team_vices = vice_candidates[vice_candidates['team'] != captain['team']]
        if not different_team_vices.empty:
            vice_captain = different_team_vices.iloc[0]
        else:
            vice_captain = vice_candidates.iloc[0]
        
        return {
            'captain': {
                'name': f"{captain['first_name']} {captain['second_name']}",
                'position': captain['position'],
                'team': captain.get('team_name', captain['team']),
                'predicted_points': captain['predicted_points'],
                'captain_score': captain['captain_score'],
                'cost': captain['now_cost']
            },
            'vice_captain': {
                'name': f"{vice_captain['first_name']} {vice_captain['second_name']}",
                'position': vice_captain['position'],
                'team': vice_captain.get('team_name', vice_captain['team']),
                'predicted_points': vice_captain['predicted_points'],
                'captain_score': vice_captain['captain_score'],
                'cost': vice_captain['now_cost']
            }
        }
    
    def optimize_transfers(self, current_team: pd.DataFrame, new_predictions: pd.DataFrame,
                          max_transfers: int = 1, transfer_cost: float = 4.0) -> Dict:
        
        logger.info(f"Optimizing transfers (max: {max_transfers})...")
        
        if max_transfers == 0:
            return {'transfers': [], 'net_gain': 0}
        
        current_team = current_team.copy()
        
        current_points = current_team['predicted_points'].sum()
        
        best_transfers = []
        best_gain = -transfer_cost * max_transfers
        
        for num_transfers in range(1, max_transfers + 1):
            for players_out in combinations(current_team.index, num_transfers):
                budget_freed = current_team.loc[list(players_out), 'now_cost'].sum()
                remaining_budget = budget_freed
                
                replacement_combinations = self._find_replacement_combinations(
                    current_team, players_out, new_predictions, remaining_budget, num_transfers
                )
                
                for replacements in replacement_combinations:
                    replacement_cost = sum([new_predictions.loc[p, 'now_cost'] for p in replacements])
                    replacement_points = sum([new_predictions.loc[p, 'predicted_points'] for p in replacements])
                    out_points = current_team.loc[list(players_out), 'predicted_points'].sum()
                    
                    net_gain = replacement_points - out_points - (num_transfers * transfer_cost)
                    
                    if net_gain > best_gain:
                        best_gain = net_gain
                        best_transfers = list(zip(players_out, replacements))
        
        return {
            'transfers': best_transfers,
            'net_gain': best_gain,
            'recommended': best_gain > 0
        }
    
    def _find_replacement_combinations(self, current_team: pd.DataFrame, players_out: Tuple,
                                     available_players: pd.DataFrame, budget: float, 
                                     num_transfers: int) -> List[List]:
        
        positions_needed = [current_team.loc[p, 'position'] for p in players_out]
        
        valid_replacements = []
        for i, position in enumerate(positions_needed):
            position_players = available_players[
                (available_players['position'] == position) &
                (available_players['now_cost'] <= budget) &
                (~available_players.index.isin(current_team.index))
            ].sort_values('predicted_points', ascending=False)
            
            valid_replacements.append(position_players.head(10).index.tolist())
        
        if num_transfers == 1:
            return [[p] for p in valid_replacements[0]]
        else:
            combinations_list = []
            for combo in combinations(range(len(valid_replacements)), num_transfers):
                if len(combo) == num_transfers:
                    for players in zip(*[valid_replacements[i] for i in combo]):
                        total_cost = sum([available_players.loc[p, 'now_cost'] for p in players])
                        if total_cost <= budget:
                            combinations_list.append(list(players))
            
            return combinations_list[:50]
    
    def generate_team_report(self, team_result: Dict) -> str:
        
        report = []
        report.append("=" * 60)
        report.append("🏆 FPL TEAM SELECTION REPORT")
        report.append("=" * 60)
        
        squad = team_result['full_squad']
        report.append(f"\n💰 BUDGET ANALYSIS:")
        report.append(f"Total Cost: £{team_result['total_cost']:.1f}M")
        report.append(f"Remaining: £{team_result['remaining_budget']:.1f}M")
        report.append(f"Budget Usage: {(team_result['total_cost']/self.budget)*100:.1f}%")
        
        report.append(f"\n📊 SQUAD COMPOSITION:")
        for pos, count in team_result['position_distribution'].items():
            report.append(f"{pos}: {count} players")
        
        report.append(f"\n🏟️  TEAM DISTRIBUTION:")
        for team, count in team_result['team_distribution'].items():
            report.append(f"Team {team}: {count} players")
        
        starting_xi = team_result['starting_xi']
        report.append(f"\n⭐ STARTING XI:")
        report.append("-" * 40)
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            pos_players = starting_xi[starting_xi['position'] == pos]
            report.append(f"\n{pos}:")
            for _, player in pos_players.iterrows():
                report.append(f"  {player['first_name']} {player['second_name']} "
                            f"({player.get('team_name', player['team'])}) - "
                            f"£{player['now_cost']:.1f}M - "
                            f"{player['predicted_points']:.1f} pts")
        
        if team_result['captain_info']:
            captain = team_result['captain_info']['captain']
            vice = team_result['captain_info']['vice_captain']
            
            report.append(f"\n👑 CAPTAIN SELECTION:")
            report.append(f"Captain: {captain['name']} ({captain['team']}) - "
                         f"{captain['predicted_points']:.1f} pts")
            report.append(f"Vice-Captain: {vice['name']} ({vice['team']}) - "
                         f"{vice['predicted_points']:.1f} pts")
        
        bench = team_result['bench']
        if not bench.empty:
            report.append(f"\n🪑 BENCH:")
            for _, player in bench.iterrows():
                report.append(f"  {player['first_name']} {player['second_name']} "
                            f"({player['position']}) - £{player['now_cost']:.1f}M")
        
        total_predicted = starting_xi['predicted_points'].sum()
        captain_bonus = team_result['captain_info']['captain']['predicted_points'] if team_result['captain_info'] else 0
        report.append(f"\n📈 EXPECTED PERFORMANCE:")
        report.append(f"Starting XI Points: {total_predicted:.1f}")
        report.append(f"Captain Bonus: +{captain_bonus:.1f}")
        report.append(f"Total Expected: {total_predicted + captain_bonus:.1f}")
        
        return "\n".join(report)

def main():
    
    try:
        players_df = pd.read_csv("player_predictions.csv")
        logger.info(f"Loaded {len(players_df)} player predictions")
        
        selector = FPLTeamSelector(budget=100.0, max_players_per_team=3)
        
        team_result = selector.select_optimal_team(players_df, include_captain=True)
        
        report = selector.generate_team_report(team_result)
        print(report)
        
        team_result['full_squad'].to_csv("selected_team.csv", index=False)
        team_result['starting_xi'].to_csv("starting_xi.csv", index=False)
        
        with open("team_report.txt", "w") as f:
            f.write(report)
        
        print(f"\n✅ Team selection complete!")
        print(f"📁 Files saved: selected_team.csv, starting_xi.csv, team_report.txt")
        
    except Exception as e:
        logger.error(f"Error in team selection: {e}")
        raise

if __name__ == "__main__":
    main()
