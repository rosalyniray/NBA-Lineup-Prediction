import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(df):
    """Encode categorical features like player names and teams"""
    encoders = {}
    
    # Encode teams
    for col in ['home_team', 'away_team']:
        encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = encoders[col].fit_transform(df[col])
    
    # Encode all player columns
    player_cols = [col for col in df.columns if 'player' in col]
    
    # First, get all unique player names
    all_players = set()
    for col in player_cols:
        all_players.update(df[col].dropna().unique())
    
    # Create a single encoder for all players
    player_encoder = LabelEncoder()
    player_encoder.fit(list(all_players))
    encoders['player'] = player_encoder
    
    # Apply encoding to all player columns
    for col in player_cols:
        df[f'{col}_encoded'] = player_encoder.transform(df[col])
    
    return df, encoders

def build_feature_matrix(encoded_df):
    """
    Build feature matrix for training with season and starting_min
    """
    # Input features: season, starting_min, home team, away team, 4 players, candidate 5th player, opposing lineup
    feature_cols = [
        'season',          # Add season as feature
        'starting_min',    # Add starting minute as feature
        'home_team_encoded', 'away_team_encoded',
        'player_1_encoded', 'player_2_encoded', 'player_3_encoded', 'player_4_encoded',
        'fifth_player_encoded'  # candidate player
    ]
    
    # Add opposing players
    for i in range(5):
        col = f'away_player_{i}_encoded'
        if col in encoded_df.columns:
            feature_cols.append(col)
    
    X = encoded_df[feature_cols].values
    y = encoded_df['effectiveness'].values  # Only use effectiveness as target
    
    return X, y

def get_player_candidates(df, home_team, season=None):
    """Get all players who have played for a given team in a specific season"""
    player_cols = [col for col in df.columns if col.startswith('home_') and col[-1].isdigit()]
    
    # Filter by team and season if provided
    team_games = df[df['home_team'] == home_team]
    if season is not None:
        team_games = team_games[team_games['season'] == season]
    
    players = set()
    for col in player_cols:
        players.update(team_games[col].dropna().unique())
    
    return sorted(list(players))