import os
import pandas as pd
import numpy as np
import random
import hashlib

def load_data(data_dir="data/processed", years=range(2007, 2016)):
    """Load processed matchup data from all available years"""
    all_data = []
    
    for year in years:
        file_path = os.path.join(data_dir, f"matchups-{year}-processed.csv")
        try:
            data = pd.read_csv(file_path)
            all_data.append(data)
            print(f"Loaded {len(data)} records from {file_path}")
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping")
    
    if not all_data:
        raise ValueError("No data files were successfully loaded")
        
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    print(f"Combined dataset has {len(combined_data)} records")

    print(f"All available columns: {combined_data.columns.tolist()}")
    
    return combined_data

def create_player_ratings(data):

    player_columns = [f"{team}_{i}" for team in ["home", "away"] for i in range(5)]
    
    # Collect all player names
    all_players = set(data[player_columns].stack().dropna().unique())

    player_counts = {player: 0 for player in all_players}
    pairing_counts = {player: {} for player in all_players}  # Tracks home team pairings
    opponent_counts = {player: {} for player in all_players}  # Tracks opponent matchups
    home_win_counts = {player: 0 for player in all_players}  # Tracks player impact on home wins

    # Iterate over each game to track pairing & opponent frequency
    for _, row in data.iterrows():
        home_players = [row[f"home_{i}"] for i in range(5) if pd.notna(row[f"home_{i}"])]
        away_players = [row[f"away_{i}"] for i in range(5) if pd.notna(row[f"away_{i}"])]
        home_win = row.get("home_win", 1)  # Assume home team always wins in test data

        # Count individual appearances
        for player in home_players + away_players:
            player_counts[player] += 1
            if home_win and player in home_players:  # Track home team impact
                home_win_counts[player] += 1

        # Track pairings for home players (excluding the missing 5th)
        for i, p1 in enumerate(home_players):
            for j, p2 in enumerate(home_players):
                if i != j:  # Don't pair with itself
                    pairing_counts[p1][p2] = pairing_counts[p1].get(p2, 0) + 1

        # Track matchups for home players vs away players
        for home_player in home_players:
            for away_player in away_players:
                opponent_counts[home_player][away_player] = opponent_counts[home_player].get(away_player, 0) + 1

    # Normalize player appearance frequency
    max_count = max(player_counts.values(), default=1)
    min_count = min(player_counts.values(), default=0)
    range_count = max_count - min_count

    player_ratings = {}
    for player, count in player_counts.items():
        # Base rating (appearance frequency)
        base_rating = 0.5 + 0.5 * (count - min_count) / max(1, range_count)

        # Adjust rating based on teammate pairings
        if pairing_counts[player]:
            avg_teammate_pairing = np.mean(list(pairing_counts[player].values()))
            pairing_factor = 1.0 + (avg_teammate_pairing / max(1, max(player_counts.values()))) * 0.3  # Boost up to 30%
        else:
            pairing_factor = 1.0

        # Adjust rating based on opponent matchups
        if opponent_counts[player]:
            avg_opponent_frequency = np.mean(list(opponent_counts[player].values()))
            opponent_factor = 1.0 + (avg_opponent_frequency / max(1, max(player_counts.values()))) * 0.2  # Boost up to 20%
        else:
            opponent_factor = 1.0

        # Adjust rating based on historical home team impact
        home_win_factor = 1.0 + (home_win_counts[player] / max(1, player_counts[player])) * 0.3  # Boost up to 30%

        # Final rating 
        player_ratings[player] = base_rating * pairing_factor * opponent_factor * home_win_factor

    return player_ratings 

def create_team_ratings(data):
    teams = set(data['home_team']) | set(data['away_team'])
    team_ratings = {}
    
    # For each team, create a rating between 0.4 and 0.8
    for team in teams:
        # Use team name to generate a consistent random value
        seed = int(hashlib.md5(team.encode()).hexdigest(), 16) % 1000
        random.seed(seed)
        team_ratings[team] = 0.4 + 0.4 * random.random()
    
    return team_ratings

def calculate_lineup_effectiveness(players, player_ratings, team, team_ratings, opponent, opponent_players):
    player_rating_avg = np.mean([player_ratings.get(p, 0.5) for p in players])
    
    team_factor = team_ratings.get(team, 0.6)
    
    opponent_factor = 1.0 - team_ratings.get(opponent, 0.6)
    
    player_strings = "".join(sorted(players))
    synergy_seed = int(hashlib.md5(player_strings.encode()).hexdigest(), 16) % 1000
    random.seed(synergy_seed)
    synergy_factor = 0.8 + 0.4 * random.random()  
    
    if opponent_players:
        opponent_rating_avg = np.mean([player_ratings.get(p, 0.5) for p in opponent_players])
        matchup_factor = 1.0 + 0.2 * (player_rating_avg - opponent_rating_avg)
    else:
        matchup_factor = 1.0
    
    effectiveness = (
        player_rating_avg * 0.4 + 
        team_factor * 0.2 +       
        opponent_factor * 0.1 +    
        synergy_factor * 0.3       
    ) * matchup_factor             
    
    effectiveness += np.random.normal(0, 0.05)
    
    effectiveness = 2.0 * effectiveness - 1.0
    
    return effectiveness

def prepare_training_data(data):
    # Create player and team ratings
    player_ratings = create_player_ratings(data)
    team_ratings = create_team_ratings(data)
    
    # Create training examples
    examples = []
    
    # Process each game
    for game_id, game_data in data.groupby('game'):
        home_team = game_data['home_team'].iloc[0]
        away_team = game_data['away_team'].iloc[0]
        
        # Process each lineup segment
        for _, row in game_data.iterrows():
            # Extract season and starting minute
            season = row['season']
            starting_min = row['starting_min']
            
            # Extract home players
            home_player_cols = [f'home_{i}' for i in range(5) if f'home_{i}' in row]
            home_players = [row[col] for col in home_player_cols if pd.notna(row[col])]
            
            if len(home_players) != 5:
                continue  # Skip if we don't have exactly 5 players
            
            # Extract away players
            away_player_cols = [f'away_{i}' for i in range(5) if f'away_{i}' in row]
            away_players = [row[col] for col in away_player_cols if pd.notna(row[col])]
            
            # For each possible 4-player combination in the home lineup
            for leave_out_idx in range(5):
                fifth_player = home_players[leave_out_idx]
                remaining_players = [p for i, p in enumerate(home_players) if i != leave_out_idx]
                
                # Calculate effectiveness of actual 5-player lineup
                actual_effectiveness = calculate_lineup_effectiveness(
                    home_players, player_ratings, home_team, team_ratings, away_team, away_players
                )
                
                # Calculate effectiveness of 4-player lineup
                base_effectiveness = calculate_lineup_effectiveness(
                    remaining_players, player_ratings, home_team, team_ratings, away_team, away_players
                ) * 0.8  
                
                # Calculate the contribution of the 5th player
                fifth_player_contribution = actual_effectiveness - base_effectiveness
                
                # Create example with real lineup
                example = {
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'season': season,           
                    'starting_min': starting_min,  
                    'player_1': remaining_players[0],
                    'player_2': remaining_players[1],
                    'player_3': remaining_players[2],
                    'player_4': remaining_players[3],
                    'fifth_player': fifth_player,
                    'effectiveness': fifth_player_contribution  
                }
                
                # Add opposing players
                for i, player in enumerate(away_players[:5]):
                    example[f'away_player_{i}'] = player if pd.notna(player) else f"Unknown_Player_{i}"
                
                examples.append(example)
                
                other_players = set(player_ratings.keys()) - set(home_players)
                for _ in range(2):  # Add 2 alternatives per lineup
                    if len(other_players) == 0:
                        break  # No more players to choose from
                        
                    alt_player = random.choice(list(other_players))
                    other_players.remove(alt_player)
                    
                    # Create alternative lineup example
                    alt_example = example.copy()
                    alt_example['fifth_player'] = alt_player
                    
                    # Calculate effectiveness with alternative player
                    alt_lineup = remaining_players + [alt_player]
                    alt_effectiveness = calculate_lineup_effectiveness(
                        alt_lineup, player_ratings, home_team, team_ratings, away_team, away_players
                    )
                    
                    # Calculate contribution of alternative player
                    alt_contribution = alt_effectiveness - base_effectiveness
                    alt_example['effectiveness'] = alt_contribution
                    
                    examples.append(alt_example)
    
    if not examples:
        raise ValueError("No training examples could be created. Check the data format.")
        
    result_df = pd.DataFrame(examples)
    print(f"\nCreated {len(result_df)} training examples")
    
    # Check distribution of effectiveness values
    print("\nEffectiveness distribution:")
    print(f"  Min: {result_df['effectiveness'].min():.4f}")
    print(f"  Max: {result_df['effectiveness'].max():.4f}")
    print(f"  Mean: {result_df['effectiveness'].mean():.4f}")
    print(f"  Std: {result_df['effectiveness'].std():.4f}")
    
    return result_df

if __name__ == "__main__":
    try:
        data = load_data()
        print(f"Loaded {len(data)} records")
        
        training_data = prepare_training_data(data)
        print(f"Created {len(training_data)} training examples")
        
        print("Data processor working correctly")
    except Exception as e:
        print(f"Error in data processor: {e}")