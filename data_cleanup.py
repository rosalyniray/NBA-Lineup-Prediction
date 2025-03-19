import os
import pandas as pd
from datetime import datetime

def extract_date_from_game_id(game_id):
    """Extract date from game ID in format YYYYMMDD"""
    date_str = game_id[:8]  # First 8 characters contain the date
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        return None

def process_matchup_files(metadata_path, data_dir, output_dir):
    """Process matchup files by filtering to include only allowed columns"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Read metadata
        metadata = pd.read_excel(metadata_path)
        
        # Debug: Print columns with their exact representation
        print("Metadata columns with length:")
        for col in metadata.columns:
            print(f"'{col}' (length: {len(col)})")
        
        # Find the correct column name for 'Can be used in the model'
        usable_col = None
        for col in metadata.columns:
            if 'Can be used in the model' in col:
                usable_col = col
                break
                
        if not usable_col:
            raise ValueError("Could not find 'Can be used in the model' column")
        
        print(f"Using column '{usable_col}' to determine allowed features")
        
        # Extract columns that can be used in the model
        allowed_columns = metadata[metadata[usable_col].notna()]['Feature'].tolist()
        print(f"Columns allowed to be used in model: {allowed_columns}")
        
        # Common columns to always include (like identifiers)
        essential_columns = ['game', 'season', 'home_team', 'away_team']
        for col in essential_columns:
            if col not in allowed_columns:
                allowed_columns.append(col)
        
        # Process each year's matchup file
        years = range(2007, 2016)  # 2007 to 2015 inclusive
        
        for year in years:
            input_file = os.path.join(data_dir, f'matchups-{year}.csv')
            output_file = os.path.join(output_dir, f'matchups-{year}-processed.csv')
            
            if not os.path.exists(input_file):
                print(f"Warning: File {input_file} not found, skipping.")
                continue
                
            print(f"Processing {year} data")
            
            # Read matchup data
            matchups = pd.read_csv(input_file)
            
            # Filter columns based on metadata
            available_allowed_columns = [col for col in allowed_columns if col in matchups.columns]
            filtered_matchups = matchups[available_allowed_columns]
            
            # Save processed data
            filtered_matchups.to_csv(output_file, index=False)
            
            print(f"Saved {len(filtered_matchups)} rows with {len(available_allowed_columns)} columns to {output_file}")
    
    except Exception as e:
        print(f"Error processing files: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Adjust these paths as needed
    metadata_path = "data/raw/Matchup-metadata.xlsx"
    data_dir = "data/raw"
    output_dir = "data/processed"
    
    process_matchup_files(metadata_path, data_dir, output_dir)