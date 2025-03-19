import os
import pickle
import numpy as np
import pandas as pd

def load_model(model_path="models/fifth_player_predictor.pkl"):
    """Load the trained model and encoders"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    with open(model_path, 'rb') as f:
        model, encoders = pickle.load(f)
    
    return model, encoders

def predict_lineup_effectiveness(model, encoders, four_players, home_team, away_team, 
                                 opposing_lineup, candidate, season=None, starting_min=0):
    """Predict effectiveness of a lineup with the given 5th player"""
    try:
        # Encode inputs
        home_team_enc = encoders['home_team'].transform([home_team])[0]
        away_team_enc = encoders['away_team'].transform([away_team])[0]
        player_encoder = encoders['player']
        
        # Encode the 4 players
        player_encodings = player_encoder.transform(four_players)
        
        # Encode the candidate
        candidate_encoding = player_encoder.transform([candidate])[0]
        
        # Encode opposing lineup
        opposing_encodings = player_encoder.transform(opposing_lineup)
        
        # Create feature vector
        features = [
            season if season else 2015,  # Use most recent season as default
            starting_min,                # Game timing
            home_team_enc, away_team_enc,
            player_encodings[0], player_encodings[1], 
            player_encodings[2], player_encodings[3],
            candidate_encoding  # candidate as 5th player
        ]
        
        # Add opposing players
        for i in range(len(opposing_encodings)):
            features.append(opposing_encodings[i])
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Predict effectiveness
        effectiveness = model.predict(features)[0]
        
        return effectiveness
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def find_optimal_fifth_player(model, encoders, four_players, home_team, away_team, 
                              opposing_lineup, candidates, season=None, starting_min=0):
    """Find the optimal 5th player from a list of candidates"""
    results = []
    
    for candidate in candidates:
        effectiveness = predict_lineup_effectiveness(
            model, encoders, four_players, home_team, away_team, opposing_lineup, 
            candidate, season, starting_min
        )
        
        if effectiveness is not None:
            results.append((candidate, effectiveness))
    
    # Sort by effectiveness (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results