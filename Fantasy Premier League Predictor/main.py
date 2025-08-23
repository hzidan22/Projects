import sys
import os
import pandas as pd
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_gathering import FPLDataGatherer
from data_preprocessing import FPLDataPreprocessor
from model_training import FPLModelTrainer
from team_selection import FPLTeamSelector
from weekly_manager import FPLWeeklyManager

class FPLPredictor:
    
    def __init__(self):
        os.makedirs("models", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        self.data_gatherer = FPLDataGatherer()
        self.preprocessor = FPLDataPreprocessor()
        self.trainer = FPLModelTrainer()
        self.team_selector = FPLTeamSelector()
        
    def run(self):
        print("🚀 FPL TEAM PREDICTOR")
        print("=" * 30)
        
        print("\n📊 Getting player data...")
        players_df, teams_df, fixtures_df = self.data_gatherer.gather_and_process_data()
        print(f"✅ Got {len(players_df)} players")
        
        print("\n🔧 Processing data...")
        cleaned_df = self.preprocessor.clean_data(players_df)
        enhanced_df = self.preprocessor.engineer_features(cleaned_df)
        print(f"✅ Processed {len(enhanced_df)} players")
        
        if os.path.exists("models/best_fpl_model.joblib"):
            print("\n🤖 Loading existing model...")
            model = self.trainer.load_model("models/best_fpl_model.joblib")
            enhanced_df['predicted_points'] = enhanced_df['total_points'] * 1.1
        else:
            print("\n🤖 Training model...")
            data_dict = self.preprocessor.prepare_modeling_data(enhanced_df)
            model_results = self.trainer.train_all_models(
                data_dict['X_train'], data_dict['y_train'],
                data_dict['X_test'], data_dict['y_test']
            )
            self.trainer.save_model("models/best_fpl_model.joblib")
            predictions = self.trainer.predict_player_points(data_dict['X_full'])
            enhanced_df['predicted_points'] = predictions
        
        print("✅ Model ready")
        
        print("\n👑 Selecting team...")
        team_result = self.team_selector.select_optimal_team(enhanced_df, include_captain=True)
        
        self.save_results(team_result)
        self._save_starting_eleven(team_result)
        
        budget_used = team_result['total_cost']
        budget_remaining = 100.0 - budget_used
        budget_usage_pct = (budget_used / 100.0) * 100
        
        print(f"✅ Team selected!")
        print(f"💰 Budget: £{budget_used:.1f}M / £100.0M ({budget_usage_pct:.1f}% used)")
        print(f"💸 Remaining: £{budget_remaining:.1f}M")
        
        captain = team_result['captain_info']['captain']['name']
        vice = team_result['captain_info']['vice_captain']['name']
        print(f"👑 Captain: {captain}")
        print(f"🥈 Vice-Captain: {vice}")
        
        print(f"\n📁 Results saved to:")
        print(f"   • simplified_team.csv")
        print(f"   • team_names_only.txt")
        print(f"   • starting_eleven.txt")
        
    def save_results(self, team_result):
        full_squad = team_result['full_squad']
        
        simplified_squad = pd.DataFrame({
            'Player': full_squad['first_name'] + ' ' + full_squad['second_name'],
            'Position': full_squad['position'],
            'Team': full_squad.get('team_name', 'Team ' + full_squad['team'].astype(str)),
            'Cost': full_squad['now_cost'],
            'Points': full_squad['predicted_points'].round(1)
        })
        
        position_order = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        simplified_squad['pos_order'] = simplified_squad['Position'].map(position_order)
        simplified_squad = simplified_squad.sort_values(['pos_order', 'Points'], ascending=[True, False])
        simplified_squad = simplified_squad.drop('pos_order', axis=1)
        
        simplified_squad.to_csv("output/simplified_team.csv", index=False)
        
        with open("output/team_names_only.txt", 'w') as f:
            f.write("🏆 YOUR FPL TEAM\n")
            f.write("=" * 20 + "\n\n")
            
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
                f.write("👑 CAPTAIN & VICE:\n")
                f.write(f"  Captain: {captain['name']}\n")
                f.write(f"  Vice: {vice['name']}\n")
    
    def _save_starting_eleven(self, team_result):
        if 'starting_xi' not in team_result:
            return
            
        starting_xi = team_result['starting_xi']
        
        starting_df = pd.DataFrame({
            'Player': starting_xi['first_name'] + ' ' + starting_xi['second_name'],
            'Position': starting_xi['position'],
            'Team': starting_xi.get('team_name', 'Team ' + starting_xi['team'].astype(str)),
            'Cost': starting_xi['now_cost'],
            'Points': starting_xi['predicted_points'].round(1)
        })
        
        formation = self._get_formation(starting_xi)
        
        with open("output/starting_eleven.txt", 'w') as f:
            f.write("⚽ STARTING XI\n")
            f.write("=" * 20 + "\n\n")
            f.write(f"🔥 Formation: {formation}\n\n")
            
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                pos_players = starting_df[starting_df['Position'] == position]
                if not pos_players.empty:
                    pos_name = {'GK': 'GOALKEEPER', 'DEF': 'DEFENDERS', 'MID': 'MIDFIELDERS', 'FWD': 'FORWARDS'}[position]
                    f.write(f"{pos_name}:\n")
                    for _, player in pos_players.iterrows():
                        captain_mark = " (C)" if team_result.get('captain_info', {}).get('captain', {}).get('name') == player['Player'] else ""
                        vice_mark = " (VC)" if team_result.get('captain_info', {}).get('vice_captain', {}).get('name') == player['Player'] else ""
                        f.write(f"  ⭐ {player['Player']}{captain_mark}{vice_mark} ({player['Team']}) - £{player['Cost']:.1f}M - {player['Points']} pts\n")
                    f.write("\n")
            
            total_cost = starting_xi['now_cost'].sum()
            total_points = starting_xi['predicted_points'].sum()
            f.write(f"💰 Starting XI Cost: £{total_cost:.1f}M\n")
            f.write(f"📊 Expected Points: {total_points:.1f}\n")
            
            bench = team_result['full_squad'][~team_result['full_squad'].index.isin(starting_xi.index)]
            f.write(f"\n🪑 BENCH:\n")
            for _, player in bench.iterrows():
                name = f"{player['first_name']} {player['second_name']}"
                team_name = player.get('team_name', f"Team {player['team']}")
                f.write(f"  • {name} ({player['position']}) - {team_name} - £{player['now_cost']:.1f}M\n")
    
    def _get_formation(self, starting_xi):
        position_counts = starting_xi['position'].value_counts()
        defenders = position_counts.get('DEF', 0)
        midfielders = position_counts.get('MID', 0)  
        forwards = position_counts.get('FWD', 0)
        return f"{defenders}-{midfielders}-{forwards}"

def main():
    parser = argparse.ArgumentParser(description='FPL Team Predictor')
    parser.add_argument('--mode', choices=['initial', 'weekly', 'transfer'], default='initial',
                       help='Mode: initial team selection, weekly update, or transfer')
    parser.add_argument('--gameweek', type=int, help='Gameweek number for weekly update')
    parser.add_argument('--out', type=str, help='Player to transfer out')
    parser.add_argument('--player-in', type=str, help='Player to transfer in')
    
    args = parser.parse_args()
    
    if args.mode == 'weekly':
        manager = FPLWeeklyManager()
        result = manager.weekly_update(args.gameweek)
        
        print(f"🚀 Weekly update complete for GW{result['gameweek']}")
        
        if result['action'] == 'initial_selection':
            team = result['team']['full_squad']
            captain = result['team']['captain_info']['captain']['name']
            print(f"👑 Captain: {captain}")
            print(f"💰 Team cost: £{team['now_cost'].sum():.1f}M")
            
        elif result['action'] == 'transfer_analysis':
            suggestions = result['transfer_suggestions']['suggestions']
            if suggestions:
                print(f"\n🔄 Best transfer suggestions for GW{result['gameweek']}:")
                for i, suggestion in enumerate(suggestions[:3], 1):
                    print(f"{i}. {suggestion['out']['name']} → {suggestion['in']['name']}")
                    print(f"   GW points gain: +{suggestion['gw_point_gain']:.1f}")
                    print(f"   Cost change: £{suggestion['cost_difference']:+.1f}M")
                    print()
            else:
                print("✅ No beneficial transfers found")
                
    elif args.mode == 'transfer':
        if not args.out or not getattr(args, 'player_in', None):
            print("Error: --out and --player-in required for transfer mode")
            return
            
        manager = FPLWeeklyManager()
        current_team = manager.load_current_team()
        
        if current_team is None:
            print("No current team found. Run weekly update first.")
            return
            
        players_df, _, _ = manager.data_gatherer.gather_and_process_data()
        gw_info = manager.get_current_gameweek()
        gameweek = args.gameweek or gw_info.get('next', {}).get('id', 1)
        
        result = manager.execute_transfer(current_team, args.out, getattr(args, 'player_in'), players_df, gameweek)
        
        if result['success']:
            print(f"✅ Transfer completed: {args.out} → {getattr(args, 'player_in')}")
            print(f"💰 Cost change: £{result['transfer']['cost_change']:+.1f}M")
            print(f"💰 New total: £{result['transfer']['new_total_cost']:.1f}M")
        else:
            print(f"❌ Transfer failed: {result['error']}")
            
    else:
        predictor = FPLPredictor()
        predictor.run()

if __name__ == "__main__":
    main()