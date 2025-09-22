# prop_analyzer_dashboard_final_v4.py
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
    Fetches weekly stats and merges it with player data to get ESPN IDs for headshots.
    """
    print("Fetching fresh data...")
    try:
        weekly_df = nfl.import_weekly_data(years=YEARS_TO_FETCH)
        player_df = nfl.import_players()
        
        player_id_map = player_df[['gsis_id', 'espn_id']]
        
        merged_df = pd.merge(
            weekly_df, 
            player_id_map, 
            left_on='player_id', 
            right_on='gsis_id', 
            how='left'
        )
        
        merged_df.rename(columns={
            'player_display_name': 'Player', 'season': 'Year', 'week': 'Week',
            'receptions': 'Rec', 'receiving_yards': 'Rec_Yds', 'receiving_tds': 'Rec_TD',
            'carries': 'Rush_Att', 'rushing_yards': 'Rush_Yds', 'rushing_tds': 'Rush_TD',
            'completions': 'Pass_Cmp', 'passing_yards': 'Pass_Yds', 'passing_tds': 'Pass_TD'
        }, inplace=True)

        merged_df['TD'] = merged_df['Rec_TD'].fillna(0) + merged_df['Rush_TD'].fillna(0) + merged_df['Pass_TD'].fillna(0)
        
        return merged_df
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

# -------------------------------
# WEB UI LAYOUT
# -------------------------------

st.set_page_config(page_title="NFL Prop Analyzer", layout="wide")
st.title("🏈 NFL Player Prop Analyzer")

with st.spinner("Loading historical NFL data... This may take a moment on first run."):
    full_df = get_nfl_data()

if full_df.empty:
    st.error("Data could not be loaded. Please check your internet connection and restart.")
else:
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
            player_df = full_df[full_df['Player'].str.contains(player_name, case=False, na=False)].copy()

            if player_df.empty:
                st.error(f"No data found for a player containing '{player_name}'. Please check spelling.")
                if 'player_df' in st.session_state:
                    del st.session_state.player_df
            else:
                st.session_state.player_df = player_df
                st.session_state.player_name_actual = player_df['Player'].iloc[0]
                st.session_state.stat_col = stat_col
                st.session_state.stat_desc = stat_desc
                st.session_state.prop_line = prop_line

    if 'player_df' in st.session_state:
        player_df = st.session_state.player_df
        player_name_actual = st.session_state.player_name_actual
        stat_col = st.session_state.stat_col
        stat_desc = st.session_state.stat_desc
        prop_line = st.session_state.prop_line

        player_df[stat_col] = pd.to_numeric(player_df[stat_col], errors='coerce').fillna(0)
        
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

        # CORRECTED LAYOUT: Adjusted column ratio for better spacing
        header_col1, header_col2 = st.columns([1, 5]) 
        with header_col1:
            espn_id = player_df['espn_id'].iloc[0]
            if pd.notna(espn_id):
                headshot_url = f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(espn_id)}.png"
                # CORRECTED IMAGE CALL: Removed deprecated parameter
                st.image(headshot_url, width=300)

        with header_col2:
            st.header(f"Analysis for {player_name_actual}")
            st.subheader(f"{stat_desc} | Prop Line: {prop_line}")

        st.divider()

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Recommendation", recommendation)
        metric_col2.metric("Hit Rate", f"{hit_rate:.1f}%")
        metric_col3.metric("Career Average", f"{average:.2f}")
        metric_col4.metric("Recent Avg (L5)", f"{recent_average:.2f}")

        st.subheader("Game-by-Game Performance")
        chart_df = player_df.copy()
        chart_df['Game'] = chart_df['Year'].astype(str) + " - W" + chart_df['Week'].astype(str)
        chart_df['Prop Line'] = prop_line
        st.line_chart(chart_df, x='Game', y=[stat_col, 'Prop Line'], color=["#0068c9", "#ff4b4b"])
        
        st.subheader("Recent Games Log")
        display_cols = ['Year', 'Week', 'Rec', 'Rec_Yds', 'Rush_Yds', 'TD']
        st.dataframe(recent_df[display_cols].sort_values(by=['Year', 'Week'], ascending=False))
