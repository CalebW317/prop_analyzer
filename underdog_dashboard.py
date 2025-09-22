# full_underdog_dashboard_secure.py
import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import gspread
from google.oauth2.service_account import Credentials

# -------------------------------
# CONFIG
# -------------------------------
# Load secrets from environment variables
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT")

if not ODDS_API_KEY or not SERVICE_ACCOUNT_JSON:
    raise ValueError("ODDS_API_KEY or GOOGLE_SERVICE_ACCOUNT environment variable is missing!")

service_account_info = json.loads(SERVICE_ACCOUNT_JSON)

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
ODDS_REGION = 'us'
ODDS_MARKETS = 'totals'
ODDS_FORMAT = 'decimal'
STAKE = 1
UNDERDOG_ODDS_THRESHOLD = 1.9
MASTER_CSV = 'underdog_picks_master.csv'
START_DATE = datetime(2025, 9, 7)

SHEET_NAME = 'Underdog Picks'
WORKSHEET_NAME = 'Weekly Picks'

# -------------------------------
# STEP 1: Current Week
# -------------------------------
def get_current_week():
    today = datetime.today()
    week = ((today - START_DATE).days // 7) + 1
    return max(1, min(week, 18))

current_week = get_current_week()
print(f"Processing Week {current_week}")

# -------------------------------
# STEP 2: Fetch Player Stats
# -------------------------------
def fetch_player_stats():
    url = "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2/table/rushing/sort/rushYards/dir/desc"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    players = []
    table = soup.find('table')
    if not table:
        print("Player stats table not found.")
        return pd.DataFrame()
    
    rows = table.find_all('tr')[1:]
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 10:
            continue
        players.append({
            'player': cols[1].text.strip(),
            'team': cols[2].text.strip(),
            'rush_yds': float(cols[3].text.strip() or 0),
            'rec_yds': float(cols[6].text.strip() or 0),
            'tds': float(cols[9].text.strip() or 0),
            'snap_count': float(cols[4].text.strip() or 1),
        })
    return pd.DataFrame(players)

# -------------------------------
# STEP 3: Fetch Live Betting Odds
# -------------------------------
def fetch_live_odds():
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': ODDS_REGION,
        'markets': ODDS_MARKETS,
        'oddsFormat': ODDS_FORMAT
    }
    response = requests.get(ODDS_API_URL, params=params)
    data = response.json()
    
    odds_rows = []
    for game in data:
        home_team = game['home_team']
        away_team = game['away_team']
        for bookmaker in game.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                for outcome in market.get('outcomes', []):
                    odds_rows.append({
                        'player': outcome.get('name'),
                        'prop_line': outcome.get('point'),
                        'odds': outcome.get('price'),
                        'stat_type': market.get('key'),
                        'team': home_team if outcome.get('name').split()[0] in home_team else away_team,
                        'opponent': away_team if home_team in outcome.get('name') else home_team
                    })
    return pd.DataFrame(odds_rows)

# -------------------------------
# STEP 4: Fetch Opponent Defense Stats
# -------------------------------
def fetch_opponent_defense():
    url = "https://www.espn.com/nfl/stats/team/_/view/defense"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    defense_rows = []
    table = soup.find('table')
    if not table:
        print("Defense stats table not found.")
        return pd.DataFrame()
    
    rows = table.find_all('tr')[1:]
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 10:
            continue
        defense_rows.append({
            'team': cols[1].text.strip(),
            'rush_allowed': float(cols[3].text.strip() or 0),
            'rec_allowed': float(cols[6].text.strip() or 0),
            'td_allowed': float(cols[9].text.strip() or 0)
        })
    return pd.DataFrame(defense_rows)

# -------------------------------
# STEP 5: Fetch & Merge Data
# -------------------------------
player_stats_df = fetch_player_stats()
odds_df = fetch_live_odds()
defense_df = fetch_opponent_defense()

if player_stats_df.empty or odds_df.empty or defense_df.empty:
    print("Missing data. Exiting.")
    exit()

df = pd.merge(odds_df, player_stats_df, on=['player','team'], how='inner')
df = pd.merge(df, defense_df, left_on='opponent', right_on='team', how='left', suffixes=('','_def'))

# -------------------------------
# STEP 6: Feature Engineering
# -------------------------------
df['rush_avg_3'] = df.groupby('player')['rush_yds'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['rec_avg_3'] = df.groupby('player')['rec_yds'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['td_rate'] = df['tds'] / df['snap_count']

df['rush_vs_defense'] = df['rush_avg_3'] - df['rush_allowed']
df['rec_vs_defense'] = df['rec_avg_3'] - df['rec_allowed']
df['td_vs_defense'] = df['td_rate'] - (df['td_allowed'] / df['snap_count'])

df['rush_usage'] = df['rush_yds'] / df['snap_count']
df['rec_usage'] = df['rec_yds'] / df['snap_count']

# -------------------------------
# STEP 7: Train Model Per Stat Type & Calculate EV
# -------------------------------
master_rows = []

for stat in ['rush_yds','rec_yds','tds']:
    stat_map = {'rush_yds':'rush_avg_3','rec_yds':'rec_avg_3','tds':'td_rate'}
    df_stat = df[df['stat_type'].str.contains(stat.split('_')[0])].copy()
    if df_stat.empty:
        continue
    
    df_stat['over_hit'] = (df_stat[stat_map[stat]] > df_stat['prop_line']).astype(int)
    features = ['rush_avg_3','rec_avg_3','td_rate','rush_vs_defense','rec_vs_defense','td_vs_defense','rush_usage','rec_usage']
    
    X = df_stat[features]
    y = df_stat['over_hit']
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    df_stat['prob_over'] = model.predict_proba(X)[:,1]
    df_stat['ev'] = (df_stat['prob_over'] * df_stat['odds']) - ((1 - df_stat['prob_over']) * STAKE)
    
    underdogs = df_stat[df_stat['odds'] > UNDERDOG_ODDS_THRESHOLD].copy()
    underdogs['stat_type'] = stat
    underdogs['week'] = current_week
    master_rows.append(underdogs[['week','player','team','stat_type','prop_line','odds','prob_over','ev']])

# -------------------------------
# STEP 8: Append to Master CSV
# -------------------------------
if master_rows:
    weekly_df = pd.concat(master_rows, ignore_index=True)
    master_path = Path(MASTER_CSV)
    if master_path.exists():
        master_df = pd.read_csv(MASTER_CSV)
        master_df = pd.concat([master_df, weekly_df], ignore_index=True)
    else:
        master_df = weekly_df
    
    master_df.to_csv(MASTER_CSV, index=False)
    print(f"Week {current_week} underdog picks appended to CSV")
else:
    print("No underdog picks generated this week.")
    exit()

# -------------------------------
# STEP 9: Update Google Sheets
# -------------------------------
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
gc = gspread.authorize(credentials)

# Open or create sheet
try:
    sh = gc.open(SHEET_NAME)
except gspread.SpreadsheetNotFound:
    sh = gc.create(SHEET_NAME)

try:
    ws = sh.worksheet(WORKSHEET_NAME)
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(title=WORKSHEET_NAME, rows="1000", cols="20")

# Clear previous data & write new data
ws.clear()
ws.update([master_df.columns.values.tolist()] + master_df.values.tolist())
print(f"Google Sheets dashboard updated: {SHEET_NAME} → {WORKSHEET_NAME}")
