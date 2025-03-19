import os
import argparse
import pandas as pd
from data_processor import load_data, prepare_training_data
from feature_engineering import encode_categorical_features, build_feature_matrix, get_player_candidates
from model_trainer import train_model, save_model, analyze_feature_importance
from predictor import load_model, find_optimal_fifth_player

def create_dirs():
    """Create necessary directories"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def train_pipeline(data_dir="data/processed", model_dir="models"):
    """Run the full training pipeline"""
    print("\n=== Starting Training Pipeline ===\n")

    # Load and prepare data
    print("Loading data...")
    raw_data = load_data(data_dir)

    print("\nPreparing training examples...")
    try:
        training_data = prepare_training_data(raw_data)

        # Check if we have training data
        if len(training_data) == 0:
            print("ERROR: No training data was created")
            return

        print(f"\nCreated {len(training_data)} training examples")

        # Encode features
        print("\nEncoding features...")
        encoded_data, encoders = encode_categorical_features(training_data)

        # Build feature matrix
        print("\nBuilding feature matrix...")
        X, y = build_feature_matrix(encoded_data)
        print(f"Feature matrix shape: {X.shape}")

        # Train model
        print("\nTraining model...")
        model = train_model(X, y, model_dir)

        # Analyze feature importance
        feature_names = [
            'Season',
            'Starting Minute',
            'Home Team',
            'Away Team',
            'Player 1',
            'Player 2',
            'Player 3',
            'Player 4',
            'Fifth Player'
        ]
        # Add opposing players
        for i in range(5):
            feature_names.append(f'Opposing Player {i}')

        analyze_feature_importance(model, feature_names)

        # Save model
        print("\nSaving model...")
        model_path = save_model(model, encoders, model_dir)

        print("\n=== Training Pipeline Complete ===")

        return model, encoders

    except Exception as e:
        print(f"ERROR: Failed to prepare training data - {e}")
        traceback.print_exc()
        return

def predict_interactive(model_path="models/fifth_player_predictor.pkl"):
    """Interactive function to predict the optimal 5th player"""
    print("\n=== NBA 5th Player Predictor ===\n")
    
    try:
        model, encoders = load_model(model_path)
        
        # Load data to get player lists
        print("Loading data for player lists...")
        data = load_data()
        
        # Get list of teams
        teams = sorted(data['home_team'].unique())
        print(f"\nAvailable teams: {', '.join(teams)}")
        
        # Get available seasons
        seasons = sorted(data['season'].unique())
        print(f"\nAvailable seasons: {', '.join(map(str, seasons))}")
        
        # Get user inputs
        home_team = input("\nEnter home team (e.g., LAL): ").strip().upper()
        if home_team not in teams:
            print(f"Warning: Team {home_team} not found in data")
        
        away_team = input("Enter away team (e.g., PHO): ").strip().upper()
        if away_team not in teams:
            print(f"Warning: Team {away_team} not found in data")
            
        # Get season
        try:
            season = int(input("Enter season (e.g., 2007): ").strip())
            if season not in seasons:
                print(f"Warning: Season {season} not found in data. Using most recent season.")
                season = max(seasons)
        except ValueError:
            print("Invalid season. Using most recent season as default")
            season = max(seasons)
            
        # Get starting minute
        try:
            starting_min = int(input("Enter game minute (0-48): ").strip())
            if starting_min < 0 or starting_min > 48:
                print("Invalid minute, using 0 as default")
                starting_min = 0
        except ValueError:
            print("Invalid input, using 0 as default")
            starting_min = 0
        
        # Get player candidates for the home team (filtered by season)
        team_players = get_player_candidates(data, home_team, season)
        if team_players:
            print(f"\nPlayers for {home_team} in {season} season: {', '.join(team_players)}")
        else:
            print(f"No players found for {home_team} in {season} season")
            return
        
        # Get the players in the lineup with one position marked as '?'
        lineup = []
        print("\nEnter the lineup in alphabetical order (enter '?' for the unknown player position):")
        
        for i in range(5):
            player = input(f"Player {i+1}: ").strip()
            lineup.append(player)
        
        if '?' not in lineup:
            print("Error: You must indicate one missing player with '?'. Please start over.")
            return
            
        # Find the position of the '?' in the lineup
        missing_position = lineup.index('?')
        
        # Determine alphabetical constraints
        alpha_before = None
        alpha_after = None
        
        if missing_position > 0:
            alpha_before = lineup[missing_position - 1]
        
        if missing_position < 4:  # 4 is the last index in a 5-player lineup
            alpha_after = lineup[missing_position + 1]
            if alpha_after == '?':  # Handle case where there might be another '?'
                for j in range(missing_position + 1, 5):
                    if lineup[j] != '?':
                        alpha_after = lineup[j]
                        break
        
        # Remove '?' from lineup to get the four known players
        four_players = [p for p in lineup if p != '?']
        
        print(f"\nSelected players: {', '.join(four_players)}")
        print(f"Predicting player for position {missing_position + 1}")
        
        if alpha_before:
            print(f"Player name must alphabetically come after: {alpha_before}")
        if alpha_after:
            print(f"Player name must alphabetically come before: {alpha_after}")
        
        # Get opposing team lineup
        away_players = get_player_candidates(data, away_team, season)
        
        if away_players:
            print(f"\nPlayers for {away_team} in {season} season: {', '.join(away_players)}")
        else:
            print(f"No players found for {away_team} in {season} season")
            return
        
        print(f"\nPlease select the 5 players from {away_team} in alphabetical order:")
        opposing_lineup = []
        for i in range(5):
            while True:
                player = input(f"Opposing Player {i+1}: ").strip()
                if player in away_players:
                    opposing_lineup.append(player)
                    break
                else:
                    print(f"Player '{player}' not found in {away_team} roster for {season}. Try again.")
        
        # Filter candidates based on alphabetical constraints
        candidates = [p for p in team_players if p not in four_players]
        
        if alpha_before:
            candidates = [p for p in candidates if p > alpha_before]
        if alpha_after:
            candidates = [p for p in candidates if p < alpha_after]
            
        if not candidates:
            print(f"No candidates match the alphabetical constraints between {alpha_before or 'start'} and {alpha_after or 'end'}")
            return
            
        print(f"\nFiltered {len(candidates)} candidates that match alphabetical constraints: {', '.join(candidates[:5])}{' ...' if len(candidates) > 5 else ''}")
        
        # Make prediction
        print(f"\nPredicting optimal player for {home_team} at position {missing_position + 1} at minute {starting_min} in {season}...")
        try:
            results = find_optimal_fifth_player(
                model, encoders, four_players, home_team, away_team, opposing_lineup, 
                candidates, season, starting_min
            )
            
            if results:
                # Display top 5 recommendations
                print("\nTop 5 recommended players:")
                for i, (player, score) in enumerate(results[:5]):
                    print(f"{i+1}. {player} (effectiveness score: {score:.4f})")
            else:
                print("No valid recommendations found.")
        except Exception as e:
            print(f"Error making prediction: {e}")
            print("This could be due to players or teams not seen in the training data")
    
    except FileNotFoundError:
        print(f"Model file not found. Please train the model first with --train")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main entry point with command-line arguments"""
    parser = argparse.ArgumentParser(description='NBA 5th Player Optimizer')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--data-dir', default='data/processed', help='Directory with processed data')
    parser.add_argument('--model-dir', default='models', help='Directory for saving models')
    parser.add_argument('--model-path', default='models/fifth_player_predictor.pkl',
                        help='Path to the saved model')

    args = parser.parse_args()

    # Create necessary directories
    create_dirs()

    # If no arguments provided, default to prediction
    if not (args.train or args.predict):
        args.predict = True

    if args.train:
        train_pipeline(args.data_dir, args.model_dir)

    if args.predict:
        predict_interactive(args.model_path)

if __name__ == "__main__":
    main()
