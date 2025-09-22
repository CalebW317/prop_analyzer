# prop_analyzer_dashboard_final.py
import streamlit as st
import pandas as pd
import nfl_data_py as nfl

# -------------------------------
# CONFIG
# -------------------------------
YEARS_TO_FETCH = [2022, 2023, 2024]

# -------------------------------
# DATA LOADING WITH STREAMLIT CACHING
# -------------------------------

@st.cache_data(ttl="24h") # Cache data for 24 hours
def get_nfl_data() -> pd.DataFrame:
    """
    Fetches and caches weekly game data.
    """
    print("Fetching fresh data...") # This will print in the terminal on a cache miss
    try:
        columns_to_get = [
            'player_id', 'player_display_name', 'season', 'week', 'receptions', 
            'receiving_yards', 'receiving_tds', 'carries', 'rushing_yards', 
            'rushing_tds', 'completions', 'passing_yards', 'passing_tds'
        ]
        weekly_df = nfl.import_weekly_data(years=YEARS_TO_FETCH, columns=columns_to_get)
        
        weekly_df.rename(columns={
            'player_display_name': 'Player', 'season': 'Year', 'week': 'Week',
            'receptions': 'Rec', 'receiving_yards': 'Rec_Yds', 'receiving_tds': 'Rec_TD',
            'carries': 'Rush_Att', 'rushing_yards': 'Rush_Yds', 'rushing_tds': 'Rush_TD',
            'completions': 'Pass_Cmp', 'passing_yards': 'Pass_Yds', 'passing_tds': 'Pass_TD'
        }, inplace=True)

        weekly_df['TD'] = weekly_df['Rec_TD'].fillna(0) + weekly_df['Rush_TD'].fillna(0) + weekly_df['Pass_TD'].fillna(0)
        
        return weekly_df
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

# -------------------------------
# WEB UI LAYOUT
# -------------------------------

st.set_page_config(page_title="NFL Prop Analyzer", layout="wide")

st.title("🏈 NFL Player Prop Analyzer")

# --- Load Data ---
with st.spinner("Loading historical NFL data... This may take a moment on first run."):
    weekly_df = get_nfl_data()

if weekly_df.empty:
    st.error("Data could not be loaded. Please check your internet connection and restart.")
else:
    # --- User Inputs ---
    stat_map = {
        "Receiving Yards": "Rec_Yds",
        "Rushing Yards": "Rush_Yds",
        "Passing Yards": "Pass_Yds",
        "Receptions": "Rec",
        "Total Touchdowns": "TD",
    }
    
    st.sidebar.header("Analyze a Prop")
    player_name = st.sidebar.text_input("Player Name:", placeholder="e.g., C. Olave")
    stat_desc = st.sidebar.selectbox("Stat Type:", options=list(stat_map.keys()))
    prop_line = st.sidebar.number_input("Prop Line:", min_value=0.0, step=0.5, format="%.1f")
    
    analyze_button = st.sidebar.button("Analyze", type="primary", use_container_width=True)
    
    if not analyze_button and 'player_df' not in st.session_state:
        st.info("Enter a player and prop on the left to get started.")
    
    if analyze_button:
        if not player_name:
            st.warning("Please enter a player name.")
        else:
            stat_col = stat_map[stat_desc]
            
            player_df = weekly_df[weekly_df['Player'].str.contains(player_name, case=False, na=False)].copy()

            if player_df.empty:
                st.error(f"No data found for a player containing '{player_name}'. Please check spelling.")
                # Clear previous results if a new search finds nothing
                if 'player_df' in st.session_state:
                    del st.session_state.player_df
            else:
                st.session_state.player_df = player_df
                st.session_state.player_name_actual = player_df['Player'].iloc[0]
                st.session_state.stat_col = stat_col
                st.session_state.stat_desc = stat_desc
                st.session_state.prop_line = prop_line

    # --- Display Results (if they exist in session state) ---
    if 'player_df' in st.session_state:
        player_df = st.session_state.player_df
        player_name_actual = st.session_state.player_name_actual
        stat_col = st.session_state.stat_col
        stat_desc = st.session_state.stat_desc
        prop_line = st.session_state.prop_line

        player_df[stat_col] = pd.to_numeric(player_df[stat_col], errors='coerce').fillna(0)
        
        # --- Calculations ---
        games_played = len(player_df)
        average = player_df[stat_col].mean()
        median = player_df[stat_col].median()
        games_over = (player_df[stat_col] > prop_line).sum()
        hit_rate = (games_over / games_played) * 100 if games_played > 0 else 0
        recent_df = player_df.sort_values(by=['Year', 'Week']).tail(5)
        recent_average = recent_df[stat_col].mean() if not recent_df.empty else 0

        score = 0
        if average > prop_line: score += 1
        if median > prop_line: score += 1
        if recent_average > prop_line: score += 1
        if hit_rate > 55: score += 1
        recommendation = "Higher" if score >= 3 else "Lower"

        # --- Display Layout ---
        st.header(f"Analysis for {player_name_actual}")
        st.subheader(f"{stat_desc} | Prop Line: {prop_line}")
        st.divider()

        # --- Key Metrics ---
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Recommendation", recommendation)
        metric_col2.metric("Hit Rate", f"{hit_rate:.1f}%")
        metric_col3.metric("Career Average", f"{average:.2f}")
        metric_col4.metric("Recent Avg (L5)", f"{recent_average:.2f}")

        # --- Performance Chart ---
        st.subheader("Game-by-Game Performance")
        chart_df = player_df.copy()
        chart_df['Game'] = chart_df['Year'].astype(str) + " - W" + chart_df['Week'].astype(str)
        chart_df['Prop Line'] = prop_line
        st.line_chart(chart_df, x='Game', y=[stat_col, 'Prop Line'], color=["#0068c9", "#ff4b4b"])
        
        # --- Recent Games Table ---
        st.subheader("Recent Games Log")
        st.dataframe(recent_df[['Year', 'Week', 'Rec', 'Rec_Yds', 'Rush_Yds', 'TD']].sort_values(by=['Year', 'Week'], ascending=False))