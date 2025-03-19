import subprocess
import os
import pandas as pd
import argparse

project_dir = os.getcwd()
main_script = os.path.join(project_dir, "main.py")
test_data = os.path.join(project_dir, "backup/NBA_test.csv")
test_labels = os.path.join(project_dir, "backup/NBA_test_labels.csv")

nba_test_df = pd.read_csv(test_data)
nba_test_labels_df = pd.read_csv(test_labels)

VALID_SEASONS = list(range(2007, 2016))  

def generate_input_string(row_index):
    row = nba_test_df.iloc[row_index]
    home_team = row["home_team"]
    away_team = row["away_team"]
    season = str(row["season"])
    
    if int(season) not in VALID_SEASONS:
        return None  

    starting_min = str(row["starting_min"])
    
    home_players = [row[f"home_{i}"] for i in range(5)]
    inputs = [home_team, away_team, season, starting_min]
    for player in home_players:
        inputs.append(player if player != "?" else "?")
    
    away_players = [row[f"away_{i}"] for i in range(5)]
    inputs.extend(away_players)
    
    return "\n".join(inputs) + "\n"

parser = argparse.ArgumentParser(description="Run NBA lineup predictor test script.")
parser.add_argument("--row", type=int, help="Specify a single row index to test.")
parser.add_argument("--range", type=str, help="Specify a range of rows to test (e.g., 1-10).")
parser.add_argument("--result", action="store_true", help="Only output final accuracy result.")
parser.add_argument("--detailed", action="store_true", help="Show full output from main script.")
args = parser.parse_args()

correct_predictions = 0
valid_predictions = 0
skipped_tests = 0  

if args.row is not None:
    row_indices = [args.row]
elif args.range:
    try:
        start, end = map(int, args.range.split("-"))
        row_indices = list(range(start, end + 1))
    except ValueError:
        print("Invalid range format. Use --range start-end (e.g., --range 1-10)")
        exit(1)
else:
    row_indices = range(len(nba_test_df))

for index in row_indices:
    input_string = generate_input_string(index)

    if input_string is None:
        skipped_tests += 1
        print(f"Skipping Row {index}: Season out of range.")
        continue

    process = subprocess.run(["python3", main_script], input=input_string, text=True, capture_output=True)
    output = process.stdout.strip()
    
    predicted_player = None
    for line in output.split("\n"):
        if "Predicted 5th Player:" in line:
            predicted_player = line.split("Predicted 5th Player:")[-1].split("(")[0].strip()
            break
    
    actual_player = nba_test_labels_df.iloc[index]["removed_value"].strip()
    
    if predicted_player:
        valid_predictions += 1
        if predicted_player == actual_player:
            correct_predictions += 1
    
    accuracy = (correct_predictions / valid_predictions * 100) if valid_predictions > 0 else 0
    print(f"\nCurrent Accuracy: {accuracy:.2f}% ({correct_predictions}/{valid_predictions} correct)")

    if not args.result:
        print(f"Row {index}: Actual = {actual_player} | Predicted = {predicted_player}")
    
    if args.detailed:
        print("\n===== FULL OUTPUT FROM MAIN SCRIPT =====")
        print(output)
        print("========================================\n")

print(f"\nModel Accuracy: {accuracy:.2f}% ({correct_predictions}/{valid_predictions} correct)")
print(f"Skipped {skipped_tests} rows due to seasons out of range.")
print("=" * 40)
