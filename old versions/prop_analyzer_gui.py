# prop_analyzer_gui.py
import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import threading

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
CACHE_MAX_AGE_HOURS = 24
YEARS_TO_FETCH = [2022, 2023, 2024]

# -------------------------------
# DATA FETCHING & LOADING (runs in a separate thread)
# -------------------------------

def fetch_data_from_api(years: list) -> pd.DataFrame:
    print(f"Fetching fresh data for seasons: {years}...")
    try:
        columns_to_get = [
            'player_display_name', 'season', 'week', 'receptions', 'receiving_yards', 
            'receiving_tds', 'carries', 'rushing_yards', 'rushing_tds', 
            'completions', 'passing_yards', 'passing_tds'
        ]
        df = nfl.import_weekly_data(years=years, columns=columns_to_get)
        df.rename(columns={
            'player_display_name': 'Player', 'season': 'Year', 'week': 'Week',
            'receptions': 'Rec', 'receiving_yards': 'Rec_Yds', 'receiving_tds': 'Rec_TD',
            'carries': 'Rush_Att', 'rushing_yards': 'Rush_Yds', 'rushing_tds': 'Rush_TD',
            'completions': 'Pass_Cmp', 'passing_yards': 'Pass_Yds', 'passing_tds': 'Pass_TD'
        }, inplace=True)
        df['TD'] = df['Rec_TD'].fillna(0) + df['Rush_TD'].fillna(0) + df['Pass_TD'].fillna(0)
        print("Fetch successful.")
        return df
    except Exception as e:
        print(f"\nAn error occurred while fetching data: {e}")
        return pd.DataFrame()

def get_nfl_data() -> pd.DataFrame:
    DATA_FOLDER.mkdir(exist_ok=True)
    if CACHE_FILE.exists():
        file_mod_time = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        if datetime.now() - file_mod_time < timedelta(hours=CACHE_MAX_AGE_HOURS):
            print(f"Loading data from recent cache file: {CACHE_FILE}")
            return pd.read_csv(CACHE_FILE)
        else:
            print("Cache file is outdated.")
    
    df = fetch_data_from_api(YEARS_TO_FETCH)
    if not df.empty:
        df.to_csv(CACHE_FILE, index=False)
        print(f"Data saved to cache: {CACHE_FILE}")
    return df

# -------------------------------
# GUI APPLICATION
# -------------------------------

class PropAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NFL Prop Analyzer")
        self.geometry("500x550")
        
        self.df = None
        self.stat_map = {
            "Receiving Yards": "Rec_Yds",
            "Rushing Yards": "Rush_Yds",
            "Passing Yards": "Pass_Yds",
            "Receptions": "Rec",
            "Total Touchdowns": "TD",
        }

        self.create_widgets()
        self.load_data_in_background()

    def create_widgets(self):
        # --- Main Frame ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Fields ---
        ttk.Label(main_frame, text="Player Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.player_name_entry = ttk.Entry(main_frame, width=40)
        self.player_name_entry.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=2)

        ttk.Label(main_frame, text="Stat Type:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.stat_var = tk.StringVar()
        self.stat_dropdown = ttk.Combobox(main_frame, textvariable=self.stat_var)
        self.stat_dropdown['values'] = list(self.stat_map.keys())
        self.stat_dropdown.grid(row=3, column=0, sticky=tk.EW, pady=2)

        ttk.Label(main_frame, text="Prop Line:").grid(row=2, column=1, sticky=tk.W, pady=2)
        self.prop_line_entry = ttk.Entry(main_frame, width=10)
        self.prop_line_entry.grid(row=3, column=1, sticky=tk.W, pady=2, padx=5)

        # --- Analyze Button ---
        self.analyze_button = ttk.Button(main_frame, text="Analyze Prop", command=self.run_analysis)
        self.analyze_button.grid(row=4, column=0, columnspan=2, pady=10)
        self.analyze_button.config(state=tk.DISABLED) # Disabled until data is loaded

        # --- Results Area ---
        ttk.Label(main_frame, text="Results:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.results_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=15)
        self.results_text.grid(row=6, column=0, columnspan=2, sticky=tk.NSEW)
        self.results_text.insert(tk.END, "Loading historical data in the background...\nPlease wait.")
        self.results_text.config(state=tk.DISABLED)
        
        main_frame.grid_columnconfigure(0, weight=3)

    def load_data_in_background(self):
        # Run the data loading in a separate thread to not freeze the GUI
        thread = threading.Thread(target=self._load_data_task)
        thread.daemon = True
        thread.start()

    def _load_data_task(self):
        self.df = get_nfl_data()
        if self.df is not None and not self.df.empty:
            self.analyze_button.config(state=tk.NORMAL)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Data loaded successfully. Ready to analyze.")
            self.results_text.config(state=tk.DISABLED)
        else:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Failed to load data. Please check the terminal for errors and restart.")
            self.results_text.config(state=tk.DISABLED)

    def run_analysis(self):
        player_name = self.player_name_entry.get()
        stat_desc = self.stat_var.get()
        prop_line_str = self.prop_line_entry.get()

        if not all([player_name, stat_desc, prop_line_str]):
            self.display_result("--- Please fill in all fields. ---")
            return
        
        try:
            prop_line = float(prop_line_str)
        except ValueError:
            self.display_result("--- Prop Line must be a number. ---")
            return

        stat_col = self.stat_map[stat_desc]
        
        # --- Perform Analysis ---
        player_df = self.df[self.df['Player'].str.contains(player_name, case=False, na=False)].copy()

        if player_df.empty:
            self.display_result(f"--- No data found for player '{player_name}'. Check spelling. ---")
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
        
        # --- Format and Display Output ---
        output = (
            f"Analysis for: {player_df['Player'].iloc[0]} | {stat_desc}\n"
            f"Prop Line: {prop_line}\n"
            f"{'='*50}\n"
            f"Games Analyzed:   {games_played}\n"
            f"Career Average:   {average:.2f}\n"
            f"Career Median:    {median:.2f}\n"
            f"Recent Avg (L5):  {recent_average:.2f}\n"
            f"Hit Rate (> {prop_line}): {hit_rate:.1f}% ({games_over} of {games_played} games)\n"
            f"{'-'*50}\n"
            f"RECOMMENDATION:   ** {recommendation} **\n"
            f"{'='*50}"
        )
        self.display_result(output)

    def display_result(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    app = PropAnalyzerApp()
    app.mainloop()
