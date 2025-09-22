# underdog_dashboard.py
import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import gspread
from google.oauth2.service_account import Credentials

# -------------------------------
# CONFIG
# -------------------------------
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT")

if not ODDS_API_KEY or not SERVICE_ACCOUNT_JSON:
    raise ValueError("ODDS_API_KEY or GOOGLE_SERVICE_ACCOUNT environment variable is missing!")

service_account_info = json.loads(SERVICE_ACCOUNT_JSON)

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
ODDS_REGION = 'us'
ODDS_MARKETS = 'totals'
ODDS_FORMAT = 'decimal'
STAKE = float(os.environ.get("STAKE", 1))
UNDERDOG_ODDS_THRESHOLD = float(os.environ.get("UNDERDOG_ODDS_THRESHOLD", 1.9))
MASTER_CSV = 'underdog_picks_master.csv'
START_DATE = datetime(2025, 9, 7)

SHEET_NAME = 'Underdog Picks'
WORKSHEET_NAME = 'Weekly Picks'

# -------------------------------
# UTILITIES
# -------------------------------
def safe_float(val, default=0):
    try:
        return float(str(val).replace(',',''))
    except:
        return default

def get_current_week():
    today = datetime.today()
    delta = (today - START_DATE).days
    week = (delta // 7) + 1
    week = max(1, min(week, 18))
    week_start = START_DATE + timedelta(weeks=week-1)
    week_end = week_start + timedelta(days=6)
    print(f"Processing Week {week}: {week_start.date()} → {week_end.date()}")
    return week

current_week = get_current_week()

# -------------------------------
# STEP 1: Fetch Player Stats
# -------------------------------
def fetch_player_stats():
    url = "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2/table/rushing/sort/rushYards/dir/desc"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        table = soup.find('table')
        if not table:
            print("Player stats table not found.")
            return pd.DataFrame()
        
        players = []
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 10:
                continue
            players.append({
                'player': cols[1].get_text(strip=True),
                'team': cols[2].get_text(strip=True),
                'rush_yds': safe_float(cols[3].get_text()),
                'rec_yds': safe_float(cols[6].get_text()),
                'tds': safe_float(cols[9].get_text()),
                'snap_count': max(safe_float(cols[4].get_text()), 1)
            })
        return pd.DataFrame(players)
    except Exception as e:
        print("Error fetching player stats:", e)
        return pd.DataFrame()

# -------------------------------
# STEP 2: Fetch Opponent Defense
# -------------------------------
def fetch_opponent_defense():
    url = "https://www.espn.com/nfl/stats/team/_/view/defense"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        table = soup.find('table')
        if not table:
            print("Defense stats table not found.")
            return pd.DataFrame()
        
        defense_rows = []
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 10:
                continue
            defense_rows.append({
                'team': cols[1].get_text(strip=True),
                'rush_allowed': safe_float(cols[3].get_text()),
                'rec_allowed': safe_float(cols[6].get_text()),
                'td_allowed': safe_float(cols[9].get_text())
            })
        return pd.DataFrame(defense_rows)
    except Exception as e:
        print("Error fetching defense stats:", e)
        return pd.DataFrame()

# -------------------------------
# STEP 3: Fetch Live Odds
# -------------------------------
def fetch_live_odds():
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': ODDS_REGION,
        'markets': ODDS_MARKETS,
        'oddsFormat': ODDS_FORMAT
    }
    try:
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
                            'prop_line': safe_float(outcome.get('point')),
                            'odds': safe_float(outcome.get('price')),
                            'stat_type': market.get('key'),
                            'team': home_team if outcome.get('name').split()[0] in home_team else away_team,
                            'opponent': away_team if home_team in outcome.get('name') else home_team
                        })
        return pd.DataFrame(odds_rows)
    except Exception as e:
        print("Error fetching odds:", e)
        return pd.DataFrame()

# -------------------------------
# STEP 4: Merge Data
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
# STEP 5: Feature Engineering
# -------------------------------
df['rush_avg_3'] = df.groupby('player')['rush_yds'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['rec_avg_3'] = df.groupby('player')['rec_yds'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['td_rate'] = df['tds'] / df['snap_count']

df['rush_vs_defense'] = df['rush_avg_3'] - df['rush_allowed']
df['rec_vs_defense'] = df['rec_avg_3'] - df['rec_allowed']
df['td_vs_defense'] = df['td_rate'] - (df['td_allowed'] / df['snap_count'].replace(0,1))

df['rush_usage'] = df['rush_yds'] / df['snap_count']
df['rec_usage'] = df['rec_yds'] / df['snap_count']

# -------------------------------
# STEP 6: Train Model & Calculate EV
# -------------------------------
master_rows = []
for stat in ['rush_yds','rec_yds','tds']:
    stat_map = {'rush_yds':'rush_avg_3','rec_yds':'rec_avg_3','tds':'td_rate'}
    df_stat = df[df['stat_type'].str.contains(stat.split('_')[0], case=False)].copy()
    if df_stat.empty or len(df_stat) < 5:
        continue

    df_stat['over_hit'] = (df_stat[stat_map[stat]] > df_stat['prop_line']).astype(int)
    features = ['rush_avg_3','rec_avg_3','td_rate','rush_vs_defense','rec_vs_defense','td_vs_defense','rush_usage','rec_usage']
    features = [f for f in features if f in df_stat.columns]

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
# STEP 7: Update Master CSV
# -------------------------------
if master_rows:
    weekly_df = pd.concat(master_rows, ignore_index=True)
    master_path = Path(MASTER_CSV)
    if master_path.exists():
        master_df = pd.read_csv(MASTER_CSV)
        master_df = pd.concat([master_df, weekly_df], ignore_index=True)
        master_df = master_df.drop_duplicates(subset=['week','player','stat_type'])
    else:
        master_df = weekly_df

    master_df.to_csv(MASTER_CSV, index=False)
    print(f"Week {current_week} underdog picks appended to CSV")
else:
    print("No underdog picks generated this week.")
    exit()

# -------------------------------
# STEP 8: Update Google Sheets
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
