# prop_analyzer.py
import pandas as pd
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
# Create a folder named 'data' in the same directory as this script
# and place your historical data CSV files inside it.
# Example files: '2024_gamelogs.csv', '2023_gamelogs.csv'
DATA_FOLDER = Path('data')

# -------------------------------
# CORE FUNCTIONS
# -------------------------------

def load_historical_data(folder: Path) -> pd.DataFrame:
    """
    Loads and combines all historical player game log CSV files from the 'data' folder.
    """
    print("Loading historical data...")
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        print(f"---")
        print(f"ERROR: No CSV files found in the '{folder}' directory.")
        print(f"Please create a '{folder}' folder and add your historical data CSVs.")
        print(f"---")
        return pd.DataFrame()

    try:
        df_list = [pd.read_csv(file) for file in csv_files]
        full_df = pd.concat(df_list, ignore_index=True)
        print(f"Successfully loaded {len(full_df):,} game records from {len(csv_files)} files.")
        return full_df
    except Exception as e:
        print(f"Error loading or parsing CSV files: {e}")
        return pd.DataFrame()

def get_user_input() -> tuple:
    """
    Prompts the user to enter the player, stat type, and prop line for analysis.
    Returns a tuple: (player_name, stat_column_in_df, user_friendly_stat_name, prop_line)
    """
    print("\n" + "-"*50)
    player_name = input("Enter the player's full name: ")

    # This map allows you to use user-friendly names that correspond to your CSV columns
    # IMPORTANT: Adjust the keys ('Rec', 'Rush', etc.) to match the column names in your CSV files.
    stat_map = {
        "1": ("Rec_Yds", "Receiving Yards"),
        "2": ("Rush_Yds", "Rushing Yards"),
        "3": ("Pass_Yds", "Passing Yards"),
        "4": ("Rec", "Receptions"),
        "5": ("TD", "Total Touchdowns"),
    }

    print("\nSelect the stat type to analyze:")
    for key, (col, desc) in stat_map.items():
        print(f"  {key}: {desc}")

    choice = input("Enter the number for the stat: ")
    stat_col, stat_desc = stat_map.get(choice, (None, None))

    if not stat_col:
        print("Invalid stat choice.")
        return None, None, None, None

    try:
        prop_line = float(input(f"Enter the prop line for {stat_desc}: "))
    except ValueError:
        print("Invalid number for prop line. Please enter a number.")
        return None, None, None, None

    return player_name, stat_col, stat_desc, prop_line

def analyze_player_prop(df: pd.DataFrame, player_name: str, stat_col: str, stat_desc: str, prop_line: float):
    """
    Analyzes a player's historical data against a given prop line and prints a recommendation.
    """
    # Find all games for the specified player (case-insensitive search)
    # The 'Player' key must match the player name column in your CSV
    player_df = df[df['Player'].str.contains(player_name, case=False, na=False)].copy()

    if player_df.empty:
        print(f"\n--- No historical data found for a player containing the name '{player_name}'. Please check spelling. ---")
        return

    # Check if the stat column exists in the DataFrame
    if stat_col not in player_df.columns:
        print(f"\n--- ERROR: The stat column '{stat_col}' does not exist in your data. ---")
        print(f"Available columns are: {list(player_df.columns)}")
        return

    # Convert stat column to numeric, coercing errors to NaN, then fill missing values with 0
    player_df[stat_col] = pd.to_numeric(player_df[stat_col], errors='coerce').fillna(0)

    # --- The "Formula" Section ---
    games_played = len(player_df)
    average = player_df[stat_col].mean()
    median = player_df[stat_col].median()
    
    # Calculate consistency (hit rate)
    games_over = (player_df[stat_col] > prop_line).sum()
    hit_rate = (games_over / games_played) * 100 if games_played > 0 else 0
    
    # Calculate recent performance (last 5 games)
    # The .sort_values part assumes your data is not pre-sorted by date/week
    recent_df = player_df.sort_values(by=['Year', 'Week'], ascending=[True, True]).tail(5)
    recent_average = recent_df[stat_col].mean() if not recent_df.empty else 0

    # --- Generate Recommendation using a simple scoring system ---
    score = 0
    if average > prop_line: score += 1
    if median > prop_line: score += 1
    if recent_average > prop_line: score += 1
    if hit_rate > 55: score += 1

    recommendation = "Higher" if score >= 3 else "Lower"

    # --- Print Formatted Output ---
    print("\n" + "="*50)
    print(f"Analysis for: {player_df['Player'].iloc[0]} | {stat_desc}")
    print(f"Prop Line: {prop_line}")
    print("="*50)
    print(f"Games Analyzed:   {games_played}")
    print(f"Career Average:   {average:.2f}")
    print(f"Career Median:    {median:.2f}")
    print(f"Recent Avg (L5):  {recent_average:.2f}")
    print(f"Hit Rate (> {prop_line}): {hit_rate:.1f}% ({games_over} of {games_played} games)")
    print("-"*50)
    print(f"RECOMMENDATION:   ** {recommendation} **")
    print("="*50)

def main():
    """Main function to run the prop analyzer."""
    full_df = load_historical_data(DATA_FOLDER)
    if full_df.empty:
        # The error message is handled in the loading function
        return
        
    # Main loop to allow for continuous analysis
    while True:
        # Unpack all four return values from the input function
        player, stat_column, stat_description, line = get_user_input()
        
        if not player or not stat_column:
            # If input is invalid, loop will ask again or exit
            pass
        else:
            analyze_player_prop(full_df, player, stat_column, stat_description, line)

        again = input("\nAnalyze another prop? (y/n): ").lower()
        if again != 'y':
            print("Exiting analyzer. Goodbye!")
            break

if __name__ == "__main__":
    main()
