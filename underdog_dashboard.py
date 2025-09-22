# underdog_dashboard.py
import os
import re
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import gspread
from google.oauth2.service_account import Credentials
import traceback
from difflib import get_close_matches

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
        if val is None: return default
        if isinstance(val, (int, float)): return float(val)
        s = str(val).strip().replace(',', '')
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if m: return float(m.group(0))
    except Exception:
        pass
    return default

def normalize_name(s: str):
    if not isinstance(s, str): return ""
    s = s.lower().replace('.', '')
    s = re.sub(r'[^a-z0-9\'\-\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_team(s: str):
    if not s: return ''
    s = s.lower().replace('.', '').replace('*','').strip()
    return s

def extract_player_name(outcome_name: str):
    if not outcome_name: return None
    parts = re.split(r'\s+(?:over|under)\b|\(|\s+-\s+|\/|-', outcome_name, flags=re.I)
    name = parts[0].strip()
    name = re.sub(r'\d+$', '', name).strip()
    return name

def flatten_columns(df):
    """Flatten MultiIndex columns into single strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(i) for i in col if i]).strip() for col in df.columns.values]
    else:
        df.columns = [str(col) for col in df.columns]
    return df

def detect_stat_column(df, keywords):
    """Return the first column name containing any of the keywords (case-insensitive)."""
    for col in df.columns:
        col_str = str(col)
        for kw in keywords:
            if kw.lower() in col_str.lower():
                return col_str
    return None

def detect_column(df, keywords):
    """Return first column name containing any of the keywords (case-insensitive)."""
    for col in df.columns:
        col_str = str(col)
        for kw in keywords:
            if kw.lower() in col_str.lower():
                return col_str
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
# FETCH PLAYER STATS
# -------------------------------
def fetch_player_stats_pfr():
    try:
        # ---- RUSHING ----
        rush_url = "https://www.pro-football-reference.com/years/2025/rushing.htm"
        df_rush_list = pd.read_html(rush_url)
        df_rush = df_rush_list[0].fillna(0)
        df_rush = flatten_columns(df_rush)

        player_col = detect_column(df_rush, ['Player', 'player'])
        team_col = detect_column(df_rush, ['Tm', 'Team', 'team'])
        rush_yds_col = detect_stat_column(df_rush, ['Yds','Rushing Yds'])
        rush_td_col = detect_stat_column(df_rush, ['TD','Rushing TD'])

        if not all([player_col, team_col, rush_yds_col, rush_td_col]):
            raise ValueError(f"Could not detect rushing columns: {df_rush.columns}")

        df_rush = df_rush[df_rush[player_col] != player_col]
        df_rush = df_rush[[player_col, team_col, rush_yds_col, rush_td_col]].copy()
        df_rush.rename(columns={player_col:'Player', team_col:'Tm',
                                rush_yds_col:'rush_yds', rush_td_col:'rush_tds'}, inplace=True)

        # ---- RECEIVING ----
        rec_url = "https://www.pro-football-reference.com/years/2025/receiving.htm"
        df_rec_list = pd.read_html(rec_url)
        df_rec = df_rec_list[0].fillna(0)
        df_rec = flatten_columns(df_rec)

        player_col = detect_column(df_rec, ['Player', 'player'])
        team_col = detect_column(df_rec, ['Tm', 'Team', 'team'])
        rec_yds_col = detect_stat_column(df_rec, ['Yds','Receiving Yds'])
        rec_td_col = detect_stat_column(df_rec, ['TD','Receiving TD'])

        if not all([player_col, team_col, rec_yds_col, rec_td_col]):
            raise ValueError(f"Could not detect receiving columns: {df_rec.columns}")

        df_rec = df_rec[df_rec[player_col] != player_col]
        df_rec = df_rec[[player_col, team_col, rec_yds_col, rec_td_col]].copy()
        df_rec.rename(columns={player_col:'Player', team_col:'Tm',
                               rec_yds_col:'rec_yds', rec_td_col:'rec_tds'}, inplace=True)

        # ---- MERGE ----
        df = pd.merge(df_rush, df_rec, on=['Player','Tm'], how='outer').fillna(0)

        players = []
        for _, r in df.iterrows():
            players.append({
                "player": r['Player'],
                "team": r['Tm'],
                "rush_yds": safe_float(r['rush_yds']),
                "rec_yds": safe_float(r['rec_yds']),
                "tds": safe_float(r['rush_tds'] + r['rec_tds']),
                "snap_count": 1.0
            })

        df_final = pd.DataFrame(players)
        print(f"[DEBUG] player_stats_df shape: {df_final.shape}")
        return df_final

    except Exception as e:
        print("[ERROR] Exception fetching PFR player stats:", e)
        traceback.print_exc()
        return pd.DataFrame()

# -------------------------------
# FETCH DEFENSE
# -------------------------------
def fetch_opponent_defense_pfr():
    url = "https://www.pro-football-reference.com/years/2025/opp.htm"
    print(f"[DEBUG] Fetching PFR opponent defense stats from {url}")
    try:
        df_list = pd.read_html(url)
        df_def = df_list[0].fillna(0)
        df_def = flatten_columns(df_def)

        team_col = detect_column(df_def, ['Team', 'Tm', 'team'])
        rush_allowed_col = detect_stat_column(df_def, ['Rushing Yds','Yds.1','Rush Yds'])
        pass_allowed_col = detect_stat_column(df_def, ['Passing Yds','Yds.2','Pass Yds'])
        rush_td_col = detect_stat_column(df_def, ['Rushing TD','TD.1'])
        pass_td_col = detect_stat_column(df_def, ['Passing TD','TD.2'])

        if not all([team_col, rush_allowed_col, pass_allowed_col, rush_td_col, pass_td_col]):
            raise ValueError(f"Could not detect all defense columns: {df_def.columns}")

        df_def = df_def[df_def[team_col] != team_col]

        defense_rows = []
        for _, r in df_def.iterrows():
            defense_rows.append({
                "team": r[team_col],
                "rush_allowed": safe_float(r.get(rush_allowed_col, 0)),
                "rec_allowed": safe_float(r.get(pass_allowed_col, 0)),
                "td_allowed": safe_float(r.get(rush_td_col, 0)),
                "td_pass_allowed": safe_float(r.get(pass_td_col, 0))
            })

        df_final = pd.DataFrame(defense_rows)
        print(f"[DEBUG] defense_df shape: {df_final.shape}")
        return df_final

    except Exception as e:
        print("[ERROR] Exception fetching PFR defense stats:", e)
        traceback.print_exc()
        return pd.DataFrame()

# -------------------------------
# FETCH ODDS
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
        with open("odds_api_debug.json", "w", encoding="utf-8") as f:
            f.write(resp.text)

        data = resp.json() if resp.status_code == 200 else []
        odds_rows = []

        for game in data:
            home_team = game.get('home_team')
            away_team = game.get('away_team')

            for bookmaker in game.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    for outcome in market.get('outcomes', []):
                        outcome_name = outcome.get('name') or ''
                        prop_line = safe_float(outcome.get('point'))
                        odds_val = safe_float(outcome.get('price'))
                        player_name_raw = extract_player_name(outcome_name)
                        player_name = player_name_raw.strip() if player_name_raw else outcome_name

                        # Determine stat_type from outcome name
                        outcome_lower = outcome_name.lower()
                        if 'rush' in outcome_lower or 'rushing' in outcome_lower:
                            stat_type = 'rush_yds'
                        elif 'rec' in outcome_lower or 'receive' in outcome_lower or 'receiving' in outcome_lower:
                            stat_type = 'rec_yds'
                        elif 'td' in outcome_lower or 'touchdown' in outcome_lower:
                            stat_type = 'tds'
                        else:
                            stat_type = 'other'

                        # Infer team if missing
                        team = None
                        last_name = normalize_name(player_name).split()[-1] if player_name else ''
                        hnorm = normalize_team(home_team)
                        anorm = normalize_team(away_team)
                        if last_name in hnorm: team = home_team
                        elif last_name in anorm: team = away_team
                        opponent = away_team if team == home_team else home_team

                        odds_rows.append({
                            "player": player_name,
                            "player_raw": outcome_name,
                            "prop_line": prop_line,
                            "odds": odds_val,
                            "stat_type": stat_type,
                            "team": team,
                            "opponent": opponent
                        })

        df = pd.DataFrame(odds_rows)
        print(f"[DEBUG] odds_df shape: {df.shape}")
        print(f"[DEBUG] Unique stat_type values in odds_df:\n{df['stat_type'].unique()}")
        return df

    except Exception as e:
        print("[ERROR] Exception fetching odds:", e)
        traceback.print_exc()
        return pd.DataFrame()

# -------------------------------
# RUN FETCHES
# -------------------------------
player_stats_df = fetch_player_stats_pfr()
player_team_map = {normalize_name(r['player']): r['team'] for _, r in player_stats_df.iterrows()} if not player_stats_df.empty else {}
odds_df = fetch_live_odds(player_team_map)
defense_df = fetch_opponent_defense_pfr()

print(f"[DEBUG] Shapes -> player_stats: {player_stats_df.shape}, odds: {odds_df.shape}, defense: {defense_df.shape}")

# Save debug CSVs if empty
if player_stats_df.empty: Path("player_stats_debug.csv").write_text(player_stats_df.to_csv(index=False))
if odds_df.empty: Path("odds_debug.csv").write_text(odds_df.to_csv(index=False))
if defense_df.empty: Path("defense_debug.csv").write_text(defense_df.to_csv(index=False))

if player_stats_df.empty or odds_df.empty or defense_df.empty:
    print("[ERROR] One or more data sources are empty. Exiting gracefully.")
    exit(0)

# -------------------------------
# NORMALIZE NAMES & MERGE
# -------------------------------
def normalize_pfr_name(s):
    if not isinstance(s,str): return ''
    s = s.lower().replace('.', '').replace('*','')
    s = re.sub(r' jr$', '', s)
    s = re.sub(r' sr$', '', s)
    s = re.sub(r'[^a-z0-9\'\-\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

player_stats_df['player_norm'] = player_stats_df['player'].apply(normalize_pfr_name)
odds_df['player_norm'] = odds_df['player'].apply(normalize_pfr_name)

# Fuzzy team inference
def merge_odds_stats(odds_df, player_stats_df):
    if odds_df.empty or player_stats_df.empty: return pd.DataFrame()
    player_team_map = {r['player_norm']: r['team'] for _, r in player_stats_df.iterrows()}
    def infer_team(row):
        if row['team']: return row['team']
        matches = get_close_matches(row['player_norm'], player_team_map.keys(), n=1, cutoff=0.8)
        if matches: return player_team_map[matches[0]]
        return None
    odds_df['team'] = odds_df.apply(infer_team, axis=1)
    df = pd.merge(odds_df, player_stats_df, on=['player_norm','team'], how='inner', suffixes=('','_ps'))
    if df.empty:
        df = pd.merge(odds_df, player_stats_df, on='player_norm', how='inner', suffixes=('','_ps'))
    if 'player_y' in df.columns: df['player'] = df['player_y']
    return df

df = merge_odds_stats(odds_df, player_stats_df)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
for col in ['rush_yds','rec_yds','tds','snap_count','rush_allowed','rec_allowed','td_allowed','td_pass_allowed','prop_line','odds']:
    if col not in df.columns: df[col] = 0.0
df['snap_count'] = df['snap_count'].replace(0,1.0)
df['rush_avg_3'] = df.groupby('player')['rush_yds'].rolling(3,min_periods=1).mean().reset_index(0,drop=True)
df['rec_avg_3'] = df.groupby('player')['rec_yds'].rolling(3,min_periods=1).mean().reset_index(0,drop=True)
df['td_rate'] = df['tds'] / df['snap_count']
df = df.merge(defense_df, how='left', left_on='opponent', right_on='team', suffixes=('','_def'))
df['rush_vs_defense'] = df['rush_avg_3'] - df['rush_allowed'].fillna(0)
df['rec_vs_defense'] = df['rec_avg_3'] - df['rec_allowed'].fillna(0)
df['td_vs_defense'] = df['td_rate'] - ((df['td_allowed'].fillna(0) + df['td_pass_allowed'].fillna(0)) / df['snap_count'])
df['rush_usage'] = df['rush_yds'] / df['snap_count']
df['rec_usage'] = df['rec_yds'] / df['snap_count']

# -------------------------------
# MODEL & EV
# -------------------------------
master_rows = []
for stat in ['rush_yds','rec_yds','tds']:
    stat_map = {'rush_yds':'rush_avg_3','rec_yds':'rec_avg_3','tds':'td_rate'}
    keyword = stat.split('_')[0]
    df['stat_type'] = df.get('stat_type','').astype(str)
    df_stat = df[df['stat_type'].str.contains(keyword,case=False,na=False)].copy()
    if df_stat.empty or len(df_stat) < 6: continue
    df_stat['over_hit'] = (df_stat[stat_map[stat]].fillna(0) > df_stat['prop_line'].fillna(0)).astype(int)
    features = [f for f in ['rush_avg_3','rec_avg_3','td_rate','rush_vs_defense','rec_vs_defense','td_vs_defense','rush_usage','rec_usage'] if f in df_stat.columns]
    X = df_stat[features].fillna(0)
    y = df_stat['over_hit']
    if y.nunique() == 1:
        prob_over = float(y.mean())
        df_stat['prob_over'] = prob_over
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X,y)
        df_stat['prob_over'] = model.predict_proba(X)[:,1]
    df_stat['ev'] = (df_stat['prob_over']*df_stat['odds']) - ((1-df_stat['prob_over'])*STAKE)
    underdogs = df_stat[df_stat['odds'] > UNDERDOG_ODDS_THRESHOLD].copy()
    if not underdogs.empty:
        underdogs['stat_type'] = stat
        underdogs['week'] = current_week
        master_rows.append(underdogs[['week','player','team','stat_type','prop_line','odds','prob_over','ev']])

# -------------------------------
# APPEND TO MASTER CSV
# -------------------------------
if master_rows:
    weekly_df = pd.concat(master_rows, ignore_index=True)
    header = not Path(MASTER_CSV).exists()
    weekly_df.to_csv(MASTER_CSV, mode='a', index=False, header=header)
else:
    print("[WARN] No underdog picks generated. Exiting.")
    exit(0)

# -------------------------------
# UPDATE GOOGLE SHEETS
# -------------------------------
master_df = pd.read_csv(MASTER_CSV)
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
gc = gspread.authorize(credentials)
try:
    sh = gc.open(SHEET_NAME)
except gspread.SpreadsheetNotFound:
    sh = gc.create(SHEET_NAME)
try:
    ws = sh.worksheet(WORKSHEET_NAME)
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(title=WORKSHEET_NAME, rows="1000", cols="20")
data_to_write = [master_df.columns.values.tolist()] + master_df.values.tolist()
ws.clear()
ws.update(data_to_write)

print("[SUCCESS] Script completed.")





