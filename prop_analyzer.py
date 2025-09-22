# prop_analyzer_v2.py
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Attempt to import nfl_data_py, guide user if not installed
try:
    import nfl_data_py as nfl
except ImportError:
    print("Module 'nfl_data_py' not found. Please install it by running: pip install nfl-data-py")
    exit()

# -------------------------------
# CONFIG
# -------------------------------
DATA_FOLDER = Path('data')
CACHE_FILE = DATA_FOLDER / 'nfl_data_cache.csv'
CACHE_MAX_AGE_HOURS = 24  # How old the cache can be before re-downloading
YEARS_TO_FETCH = [2021, 2022, 2023, 2024] # Seasons to grab data for

# -------------------------------
# DATA FETCHING & LOADING
# -------------------------------

def fetch_data_from_api(years: list) -> pd.DataFrame:
    """
    Fetches weekly player game data using the nfl_data_py library.
    """
    print(f"Fetching fresh data for seasons: {years}...")
    try:
        # Define the specific columns we need to keep the DataFrame lean
        columns_to_get = [
            'player_display_name', 'season', 'week', 'opponent', 'receptions', 
            'receiving_yards', 'receiving_tds', 'carries', 'rushing_yards', 
            'rushing_tds', 'completions', 'passing_yards', 'passing_tds'
        ]
        df = nfl.import_weekly_data(years=years, columns=columns_to_get)
        
        # Rename columns for consistency with our analyzer
        df.rename(columns={
            'player_display_name': 'Player', 'season': 'Year', 'week': 'Week',
            'receptions': 'Rec', 'receiving_yards': 'Rec_Yds', 'receiving_tds': 'Rec_TD',
            'carries': 'Rush_Att', 'rushing_yards': 'Rush_Yds', 'rushing_tds': 'Rush_TD',
            'completions': 'Pass_Cmp', 'passing_yards': 'Pass_Yds', 'passing_tds': 'Pass_TD'
        }, inplace=True)

        # Create a total TD column
        df['TD'] = df['Rec_TD'] + df['Rush_TD'] + df['Pass_TD']
        
        print("Fetch successful.")
        return df

    except Exception as e:
        print(f"\nAn error occurred while fetching data: {e}")
        return pd.DataFrame()

def get_nfl_data() -> pd.DataFrame:
    """
    Main data handler. Loads data from cache if it's recent, otherwise fetches
    fresh data from the API and updates the cache.
    """
    DATA_FOLDER.mkdir(exist_ok=True) # Ensure the data folder exists
    
    if CACHE_FILE.exists():
        # Check how old the file is
        file_mod_time = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        if datetime.now() - file_mod_time < timedelta(hours=CACHE_MAX_AGE_HOURS):
            print(f"Loading data from recent cache file: {CACHE_FILE}")
            return pd.read_csv(CACHE_FILE)
        else:
            print("Cache file is outdated.")
    
    # If we're here, cache is missing or old, so fetch new data
    df = fetch_data_from_api(YEARS_TO_FETCH)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
        print(f"Data saved to cache: {CACHE_FILE}")
    return df

# -------------------------------
# ANALYSIS & USER INTERFACE (No changes needed here)
# -------------------------------

def get_user_input() -> tuple:
    """
    Prompts the user to enter the player, stat type, and prop line for analysis.
    """
    print("\n" + "-"*50)
    player_name = input("Enter the player's full name: ")

    stat_map = {
        "1": ("Rec_Yds", "Receiving Yards"), "2": ("Rush_Yds", "Rushing Yards"),
        "3": ("Pass_Yds", "Passing Yards"), "4": ("Rec", "Receptions"),
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
    player_df = df[df['Player'].str.contains(player_name, case=False, na=False)].copy()

    if player_df.empty:
        print(f"\n--- No historical data found for a player containing '{player_name}'. Check spelling. ---")
        return

    player_df[stat_col] = pd.to_numeric(player_df[stat_col], errors='coerce').fillna(0)

    games_played = len(player_df)
    average = player_df[stat_col].mean()
    median = player_df[stat_col].median()
    games_over = (player_df[stat_col] > prop_line).sum()
    hit_rate = (games_over / games_played) * 100 if games_played > 0 else 0
    recent_df = player_df.sort_values(by=['Year', 'Week'], ascending=[True, True]).tail(5)
    recent_average = recent_df[stat_col].mean() if not recent_df.empty else 0

    score = 0
    if average > prop_line: score += 1
    if median > prop_line: score += 1
    if recent_average > prop_line: score += 1
    if hit_rate > 55: score += 1

    recommendation = "Higher" if score >= 3 else "Lower"

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

# -------------------------------
# MAIN EXECUTION
# -------------------------------

def main():
    """Main function to run the prop analyzer."""
    full_df = get_nfl_data()
    if full_df.empty:
        print("Could not load or fetch NFL data. Exiting.")
        return
        
    while True:
        player, stat_column, stat_description, line = get_user_input()
        
        if player and stat_column:
            analyze_player_prop(full_df, player, stat_column, stat_description, line)

        again = input("\nAnalyze another prop? (y/n): ").lower()
        if again != 'y':
            print("Exiting analyzer. Goodbye!")
            break

if __name__ == "__main__":
    main()
