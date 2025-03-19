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
    
    # Print column list for debugging
    print(f"All available columns: {combined_data.columns.tolist()}")
    
    return combined_data

def create_player_ratings(data):
    """
    Create synthetic player ratings based on frequency of appearance
    More frequent players are likely to be better
    """
    player_columns = []
    for prefix in ['home_', 'away_']:
        for i in range(5):
            col = f"{prefix}{i}"
            if col in data.columns:
                player_columns.append(col)
    
    # Collect all player names
    all_players = set()
    for col in player_columns:
        all_players.update(data[col].dropna().unique())
    
    print(f"Found {len(all_players)} unique players")
    
    # Count player appearances
    player_counts = {}
    for player in all_players:
        count = 0
        for col in player_columns:
            count += (data[col] == player).sum()
        player_counts[player] = count
    
    # Create ratings (normalize counts)
    max_count = max(player_counts.values())
    min_count = min(player_counts.values())
    range_count = max_count - min_count
    
    player_ratings = {}
    for player, count in player_counts.items():
        # Normalize to 0.5-1.0 range
        if range_count > 0:
            rating = 0.5 + 0.5 * (count - min_count) / range_count
        else:
            rating = 0.75  # Default if all counts are the same
        player_ratings[player] = rating
    
    print(f"Created player ratings from {min_count} to {max_count} appearances")
    
    # Print some example ratings
    examples = list(player_ratings.items())[:5]
    print(f"Sample player ratings: {examples}")
    
    return player_ratings

def create_team_ratings(data):
    """Create synthetic team ratings based on win percentages"""
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
    """
    Calculate synthetic effectiveness of a lineup based on:
    1. Average player ratings
    2. Team rating
    3. Opponent team rating
    4. Player synergy (some combinations work better)
    """
    # Base effectiveness = average player rating
    player_rating_avg = np.mean([player_ratings.get(p, 0.5) for p in players])
    
    # Team factor
    team_factor = team_ratings.get(team, 0.6)
    
    # Opponent factor (inverse)
    opponent_factor = 1.0 - team_ratings.get(opponent, 0.6)
    
    # Synergy factor - using player name hashing for consistency
    player_strings = "".join(sorted(players))
    synergy_seed = int(hashlib.md5(player_strings.encode()).hexdigest(), 16) % 1000
    random.seed(synergy_seed)
    synergy_factor = 0.8 + 0.4 * random.random()  # 0.8 to 1.2
    
    # Opponent matchup factor
    if opponent_players:
        opponent_rating_avg = np.mean([player_ratings.get(p, 0.5) for p in opponent_players])
        matchup_factor = 1.0 + 0.2 * (player_rating_avg - opponent_rating_avg)
    else:
        matchup_factor = 1.0
    
    # Combine factors
    effectiveness = (
        player_rating_avg * 0.4 +  # 40% based on player quality
        team_factor * 0.2 +        # 20% based on team strength
        opponent_factor * 0.1 +    # 10% based on opponent weakness
        synergy_factor * 0.3       # 30% based on player synergy
    ) * matchup_factor             # Modified by matchup
    
    # Add small noise for variation
    effectiveness += np.random.normal(0, 0.05)
    
    # Scale to reasonable range (-1 to 1)
    effectiveness = 2.0 * effectiveness - 1.0
    
    return effectiveness

def prepare_training_data(data):
    """Transform matchup data into training examples with synthetic performance metrics"""
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
                ) * 0.8  # 4-player lineups are less effective
                
                # Calculate the contribution of the 5th player
                fifth_player_contribution = actual_effectiveness - base_effectiveness
                
                # Create example with real lineup
                example = {
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'season': season,            # Include season
                    'starting_min': starting_min,  # Include starting minute
                    'player_1': remaining_players[0],
                    'player_2': remaining_players[1],
                    'player_3': remaining_players[2],
                    'player_4': remaining_players[3],
                    'fifth_player': fifth_player,
                    'effectiveness': fifth_player_contribution  # How much this player adds
                }
                
                # Add opposing players
                for i, player in enumerate(away_players[:5]):
                    example[f'away_player_{i}'] = player if pd.notna(player) else f"Unknown_Player_{i}"
                
                examples.append(example)
                
                # Create additional examples with alternative 5th players
                # This adds variation to the dataset with better/worse choices
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
    # Quick test of the data loading functionality
    try:
        data = load_data()
        print(f"Loaded {len(data)} records")
        
        training_data = prepare_training_data(data)
        print(f"Created {len(training_data)} training examples")
        
        print("Data processor working correctly")
    except Exception as e:
        print(f"Error in data processor: {e}")