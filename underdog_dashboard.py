# underdog_dashboard.py
import os
import re
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import gspread
from google.oauth2.service_account import Credentials
import traceback

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

REQUEST_TIMEOUT = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UnderdogBot/1.0)"}

# -------------------------------
# UTILITIES
# -------------------------------
def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        # Some APIs return nested structures; handle that gracefully
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        s = s.replace(',', '')
        # remove trailing non-numeric characters
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if m:
            return float(m.group(0))
    except Exception:
        pass
    return default

def normalize_name(s: str):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.replace('.', '')
    s = re.sub(r'[^a-z0-9\'\-\s]', ' ', s)  # keep letters, numbers, apostrophes, hyphens
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_player_name(outcome_name: str):
    """
    Extract the player name from an outcome label, e.g.:
    "D. Adams Over 65.5 Receiving Yards" -> "D. Adams"
    This uses simple heuristics: split on ' over ', ' under ', ' - ', '(' etc.
    """
    if not outcome_name:
        return None
    # split on common separators (case-insensitive)
    parts = re.split(r'\s+(?:over|under)\b|\(|\s+-\s+|\/|-', outcome_name, flags=re.I)
    name = parts[0].strip()
    # trim trailing numbers if they got included
    name = re.sub(r'\d+$', '', name).strip()
    return name

def infer_player_team_from_stats(outcome_name: str, player_team_map: dict):
    """
    Try to infer the player's team from the outcome text using the
    player->team map from the player stats table.
    Returns team or None.
    """
    if not outcome_name:
        return None
    extracted = extract_player_name(outcome_name)
    if not extracted:
        return None
    n_extracted = normalize_name(extracted)
    # exact match
    if n_extracted in player_team_map:
        return player_team_map[n_extracted]
    # try last name matching
    last = n_extracted.split()[-1]
    candidates = [team for pname, team in player_team_map.items() if pname.split()[-1] == last]
    if len(candidates) == 1:
        return candidates[0]
    # fallback: try substring match
    for pname, team in player_team_map.items():
        if last in pname:
            return team
    return None

def get_current_week():
    today = datetime.today()
    delta = (today - START_DATE).days
    week = (delta // 7) + 1
    week = max(1, min(week, 18))
    week_start = START_DATE + timedelta(weeks=week-1)
    week_end = week_start + timedelta(days=6)
    print(f"[DEBUG] Processing Week {week}: {week_start.date()} → {week_end.date()}")
    return week

current_week = get_current_week()

# -------------------------------
# STEP 1: Fetch Player Stats (ESPN) with debug saving
# -------------------------------
def fetch_player_stats():
    url = "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2/table/rushing/sort/rushYards/dir/desc"
    print(f"[DEBUG] Fetching ESPN player stats from {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        print(f"[DEBUG] ESPN player stats HTTP {resp.status_code}")
        # Save debug HTML for inspection in Actions artifacts
        with open("espn_player_debug.html", "w", encoding="utf-8") as f:
            f.write(resp.text)
        soup = BeautifulSoup(resp.content, 'lxml')
        table = soup.find('table')
        if not table:
            print("[ERROR] Player stats table not found on ESPN page. Saved espn_player_debug.html for inspection.")
            return pd.DataFrame()

        rows = table.find_all('tr')[1:]
        players = []
        for row in rows:
            cols = row.find_all('td')
            # ESPN tables sometimes have different layouts; guard by columns
            if len(cols) < 6:
                continue
            try:
                player_name = cols[1].get_text(strip=True)
                team = cols[2].get_text(strip=True)
                rush_yds = safe_float(cols[3].get_text())
                snap_count = safe_float(cols[4].get_text()) or 1.0
                # rec columns might shift depending on table; guard with try
                rec_yds = safe_float(cols[6].get_text()) if len(cols) > 6 else 0.0
                tds = safe_float(cols[9].get_text()) if len(cols) > 9 else 0.0
                players.append({
                    "player": player_name,
                    "team": team,
                    "rush_yds": rush_yds,
                    "rec_yds": rec_yds,
                    "tds": tds,
                    "snap_count": max(snap_count, 1.0)
                })
            except Exception as e:
                print("[WARN] Skipping a player row due to parse error:", e)
                continue

        df = pd.DataFrame(players)
        print(f"[DEBUG] player_stats_df shape: {df.shape}")
        if not df.empty:
            print(df.head())
        return df
    except Exception as e:
        print("[ERROR] Exception fetching player stats:", e)
        traceback.print_exc()
        # save partial content if available
        try:
            with open("espn_player_debug_error.html", "w", encoding="utf-8") as f:
                f.write(str(e))
        except:
            pass
        return pd.DataFrame()

# -------------------------------
# STEP 2: Fetch Opponent Defense (ESPN) with debug saving
# -------------------------------
def fetch_opponent_defense():
    url = "https://www.espn.com/nfl/stats/team/_/view/defense"
    print(f"[DEBUG] Fetching ESPN defense stats from {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        print(f"[DEBUG] ESPN defense HTTP {resp.status_code}")
        with open("espn_defense_debug.html", "w", encoding="utf-8") as f:
            f.write(resp.text)
        soup = BeautifulSoup(resp.content, 'lxml')
        table = soup.find('table')
        if not table:
            print("[ERROR] Defense stats table not found on ESPN page. Saved espn_defense_debug.html for inspection.")
            return pd.DataFrame()
        defense_rows = []
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 6:
                continue
            try:
                team = cols[1].get_text(strip=True)
                rush_allowed = safe_float(cols[3].get_text())
                rec_allowed = safe_float(cols[6].get_text()) if len(cols) > 6 else 0.0
                td_allowed = safe_float(cols[9].get_text()) if len(cols) > 9 else 0.0
                defense_rows.append({
                    "team": team,
                    "rush_allowed": rush_allowed,
                    "rec_allowed": rec_allowed,
                    "td_allowed": td_allowed
                })
            except Exception as e:
                print("[WARN] Skipping a defense row due to parse error:", e)
                continue
        df = pd.DataFrame(defense_rows)
        print(f"[DEBUG] defense_df shape: {df.shape}")
        if not df.empty:
            print(df.head())
        return df
    except Exception as e:
        print("[ERROR] Exception fetching defense stats:", e)
        traceback.print_exc()
        try:
            with open("espn_defense_debug_error.html", "w", encoding="utf-8") as f:
                f.write(str(e))
        except:
            pass
        return pd.DataFrame()

# -------------------------------
# STEP 3: Fetch Live Odds (the-odds-api) with debug saving
# -------------------------------
def fetch_live_odds(player_team_map):
    print("[DEBUG] Fetching live odds from Odds API")
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': ODDS_REGION,
        'markets': ODDS_MARKETS,
        'oddsFormat': ODDS_FORMAT
    }
    try:
        resp = requests.get(ODDS_API_URL, params=params, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        print(f"[DEBUG] Odds API HTTP {resp.status_code}")
        # Save raw JSON for debugging
        try:
            with open("odds_api_debug.json", "w", encoding="utf-8") as f:
                f.write(resp.text)
        except Exception as e:
            print("[WARN] Failed to write odds_api_debug.json:", e)

        data = resp.json() if resp.status_code == 200 else []
        odds_rows = []
        for game in data:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    key = market.get('key')  # e.g., totals
                    for outcome in market.get('outcomes', []):
                        outcome_name = outcome.get('name')  # string containing player + description
                        prop_line = safe_float(outcome.get('point'))
                        odds_val = safe_float(outcome.get('price'))
                        # extract a cleaned player name
                        player_name_raw = extract_player_name(outcome_name)
                        player_name = player_name_raw.strip() if player_name_raw else outcome_name
                        # try to infer team via player name/lookup
                        team = infer_player_team_from_stats(outcome_name, player_team_map)
                        # fallback heuristics: if team still None, try to check if last name in home_team or away_team strings
                        if not team:
                            # check if home or away contains last name
                            last = normalize_name(player_name).split()[-1] if player_name else ''
                            if last and last in normalize_name(home_team):
                                team = home_team
                            elif last and last in normalize_name(away_team):
                                team = away_team
                            else:
                                # fallback: set team as None (will be evident during merge)
                                team = None
                        opponent = away_team if team == home_team else home_team
                        odds_rows.append({
                            "player": player_name,
                            "player_raw": outcome_name,
                            "prop_line": prop_line,
                            "odds": odds_val,
                            "stat_type": key,
                            "team": team,
                            "opponent": opponent
                        })
        df = pd.DataFrame(odds_rows)
        print(f"[DEBUG] odds_df shape: {df.shape}")
        if not df.empty:
            print(df.head())
        return df
    except Exception as e:
        print("[ERROR] Exception fetching odds:", e)
        traceback.print_exc()
        return pd.DataFrame()

# -------------------------------
# RUN FETCHES
# -------------------------------
player_stats_df = fetch_player_stats()
# build player->team map for inference
player_team_map = {}
if not player_stats_df.empty:
    for _, r in player_stats_df.iterrows():
        key = normalize_name(r['player'])
        player_team_map[key] = r['team']

odds_df = fetch_live_odds(player_team_map)
defense_df = fetch_opponent_defense()

print(f"[DEBUG] Shapes -> player_stats: {player_stats_df.shape}, odds: {odds_df.shape}, defense: {defense_df.shape}")

# If any are empty, bail with explanation (and saved debug files)
if player_stats_df.empty or odds_df.empty or defense_df.empty:
    print("[ERROR] One or more data sources are empty. Details above. Exiting without updating Google Sheets.")
    # Show which ones are empty:
    if player_stats_df.empty:
        print("[ERROR] player_stats_df is empty. Check espn_player_debug.html")
    if odds_df.empty:
        print("[ERROR] odds_df is empty. Check odds_api_debug.json and ODDS_API_KEY.")
    if defense_df.empty:
        print("[ERROR] defense_df is empty. Check espn_defense_debug.html")
    exit(1)

# -------------------------------
# CLEAN & MERGE
# -------------------------------
# Normalize player names for merging
player_stats_df['player_norm'] = player_stats_df['player'].apply(lambda x: normalize_name(x))
odds_df['player_norm'] = odds_df['player'].apply(lambda x: normalize_name(x) if isinstance(x, str) else '')

# Show unmatched players for debugging before merge
odds_players = set(odds_df['player_norm'].unique())
stat_players = set(player_stats_df['player_norm'].unique())
unmatched = sorted(list(odds_players - stat_players))
print(f"[DEBUG] Number of odds players: {len(odds_players)}, stat players: {len(stat_players)}, unmatched odds players sample (up to 20): {unmatched[:20]}")

# Merge - try first on normalized player name + team if team known, else fallback to player_norm only
df = pd.merge(odds_df, player_stats_df, left_on=['player_norm','team'], right_on=['player_norm','team'], how='inner')
if df.empty:
    # fallback: merge on player_norm only (teamless)
    df = pd.merge(odds_df, player_stats_df, on='player_norm', how='inner', suffixes=('','_ps'))
print(f"[DEBUG] Post-merge shape: {df.shape}")
if df.empty:
    print("[ERROR] After merge there are 0 rows. Cannot continue. Dumping samples for debugging.")
    print("odds_df sample:")
    print(odds_df.head(20).to_dict(orient='records'))
    print("player_stats_df sample:")
    print(player_stats_df.head(20).to_dict(orient='records'))
    exit(1)

# prefer consistent column names
if 'player_x' in df.columns and 'player_y' in df.columns:
    df['player'] = df['player_y']  # prefer player name from stats
elif 'player' not in df.columns and 'player_y' in df.columns:
    df['player'] = df['player_y']

# Keep necessary columns and avoid duplicate team columns
if 'team_ps' in df.columns and 'team' in df.columns:
    # team from player stats might be team_ps, normalize accordingly
    df.rename(columns={'team_ps': 'team_stats'}, inplace=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
# ensure numeric columns exist
for col in ['rush_yds','rec_yds','tds','snap_count','rush_allowed','rec_allowed','td_allowed','prop_line','odds']:
    if col not in df.columns:
        df[col] = 0.0

df['snap_count'] = df['snap_count'].replace(0, 1.0)

df['rush_avg_3'] = df.groupby('player')['rush_yds'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['rec_avg_3'] = df.groupby('player')['rec_yds'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['td_rate'] = df['tds'] / df['snap_count'].replace(0, 1.0)

df['rush_vs_defense'] = df['rush_avg_3'] - df['rush_allowed']
df['rec_vs_defense'] = df['rec_avg_3'] - df['rec_allowed']
df['td_vs_defense'] = df['td_rate'] - (df['td_allowed'] / df['snap_count'].replace(0,1.0))

df['rush_usage'] = df['rush_yds'] / df['snap_count'].replace(0,1.0)
df['rec_usage'] = df['rec_yds'] / df['snap_count'].replace(0,1.0)

print(f"[DEBUG] Feature engineering complete. Sample:")
print(df[['player','team','prop_line','odds','rush_avg_3','rec_avg_3','td_rate']].head())

# -------------------------------
# MODEL TRAINING & EV
# -------------------------------
master_rows = []

for stat in ['rush_yds', 'rec_yds', 'tds']:
    stat_map = {'rush_yds':'rush_avg_3','rec_yds':'rec_avg_3','tds':'td_rate'}
    keyword = stat.split('_')[0]  # e.g., 'rush'
    # guard against missing stat_type
    df['stat_type'] = df.get('stat_type', '').astype(str)
    try:
        df_stat = df[df['stat_type'].str.contains(keyword, case=False, na=False)].copy()
    except Exception:
        df_stat = df.copy()

    print(f"[DEBUG] Preparing model for stat '{stat}' -> df_stat size: {df_stat.shape}")
    if df_stat.empty or len(df_stat) < 6:
        print(f"[WARN] Not enough rows for stat {stat} (need >=6). Skipping.")
        continue

    # target: whether the player's recent measure > prop_line
    df_stat['over_hit'] = (df_stat[stat_map[stat]].fillna(0) > df_stat['prop_line'].fillna(0)).astype(int)
    features = ['rush_avg_3','rec_avg_3','td_rate','rush_vs_defense','rec_vs_defense','td_vs_defense','rush_usage','rec_usage']
    features = [f for f in features if f in df_stat.columns]

    X = df_stat[features].fillna(0)
    y = df_stat['over_hit']

    # If y has only one class, use its mean as probability
    prob_over = None
    if y.nunique() == 1:
        prob_over = float(y.mean())
        df_stat['prob_over'] = prob_over
        print(f"[WARN] Only one class present in target for {stat}. Using prob={prob_over:.3f} for all rows.")
    else:
        try:
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X, y)
            df_stat['prob_over'] = model.predict_proba(X)[:, 1]
        except Exception as e:
            print("[ERROR] Model training/predict failed for stat", stat, e)
            traceback.print_exc()
            continue

    # EV formula for decimal odds with stake=STAKE: EV = prob * odds - (1 - prob) * 1
    df_stat['ev'] = (df_stat['prob_over'] * df_stat['odds']) - ((1 - df_stat['prob_over']) * STAKE)

    underdogs = df_stat[df_stat['odds'] > UNDERDOG_ODDS_THRESHOLD].copy()
    if underdogs.empty:
        print(f"[DEBUG] No underdogs found for stat {stat}.")
        continue

    underdogs['stat_type'] = stat
    underdogs['week'] = current_week
    master_rows.append(underdogs[['week','player','team','stat_type','prop_line','odds','prob_over','ev']])

# -------------------------------
# APPEND CURRENT WEEK TO MASTER CSV
# -------------------------------
if master_rows:
    weekly_df = pd.concat(master_rows, ignore_index=True)
    # append to CSV
    header = not Path(MASTER_CSV).exists()
    weekly_df.to_csv(MASTER_CSV, mode='a', index=False, header=header)
    print(f"[DEBUG] Week {current_week} underdog picks appended to {MASTER_CSV} (rows added: {len(weekly_df)})")
else:
    print("[WARN] No underdog picks were generated this week. Exiting.")
    exit(0)

# -------------------------------
# READ & LOG MASTER CSV
# -------------------------------
if Path(MASTER_CSV).exists():
    master_df = pd.read_csv(MASTER_CSV)
    print(f"[DEBUG] master CSV now shape: {master_df.shape}")
    if not master_df.empty:
        print(master_df.tail(10))
else:
    print("[ERROR] Master CSV does not exist after writing. Exiting.")
    exit(1)

# -------------------------------
# UPDATE GOOGLE SHEETS
# -------------------------------
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
try:
    credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    gc = gspread.authorize(credentials)
    print("[DEBUG] Authorized to Google Sheets via service account.")
except Exception as e:
    print("[ERROR] Google Sheets authorization failed:", e)
    traceback.print_exc()
    exit(1)

try:
    try:
        sh = gc.open(SHEET_NAME)
        print(f"[DEBUG] Opened spreadsheet: {SHEET_NAME}")
    except gspread.SpreadsheetNotFound:
        # Try to create and then share (note: sharing requires Drive API or manual share)
        print(f"[INFO] Spreadsheet '{SHEET_NAME}' not found. Creating it now.")
        sh = gc.create(SHEET_NAME)
        print(f"[INFO] Created spreadsheet: {SHEET_NAME}. NOTE: You may need to share it with the service account email if you want to see it in Google Drive UI.")
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        print(f"[INFO] Worksheet '{WORKSHEET_NAME}' not found. Adding worksheet.")
        ws = sh.add_worksheet(title=WORKSHEET_NAME, rows="1000", cols="20")

    # Write the whole master CSV into the worksheet (replace contents)
    data_to_write = [master_df.columns.values.tolist()] + master_df.values.tolist()
    ws.clear()
    ws.update(data_to_write)
    print(f"[DEBUG] Google Sheets dashboard updated: {SHEET_NAME} → {WORKSHEET_NAME}")
except Exception as e:
    print("[ERROR] Failed to update Google Sheets:", e)
    traceback.print_exc()
    print("[INFO] Saved debug files: espn_player_debug.html, espn_defense_debug.html, odds_api_debug.json")
    exit(1)

print("[SUCCESS] Script completed.")
