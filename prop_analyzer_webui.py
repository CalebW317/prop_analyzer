# prop_analyzer_dashboard_final_v27_final_v8.py
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl 

# -------------------------------
# CONFIG
# -------------------------------
YEARS_TO_FETCH = [2022, 2023, 2024, 2025]

OFFENSE_POS = ['QB', 'RB', 'WR', 'TE', 'FB']
DEFENSE_POS = ['DE', 'DT', 'NT', 'DL', 'EDGE', 'LB', 'CB', 'SAF', 'DB']
SPECIAL_TEAMS_POS = ['K', 'P', 'LS']

# -------------------------------
# DATA LOADING
# -------------------------------
@st.cache_data(ttl=86400)
def get_nfl_data() -> pd.DataFrame:
    """
    Fetches weekly player stats for specified years directly from nflverse GitHub CSVs.
    This ensures all columns (offense, defense, special teams) are consistently available.
    """
    merged_df_list = []
    
    for year in YEARS_TO_FETCH:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"
        try:
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip()
            df['Year'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)
            df['Week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
            merged_df_list.append(df)
            st.write(f"Successfully loaded {year} data...") # Progress indicator
        except Exception as e:
            st.warning(f"Could not fetch {year} data from GitHub: {e}")

    if not merged_df_list:
        st.error("No NFL data could be loaded.")
        return pd.DataFrame()

    full_df = pd.concat(merged_df_list, ignore_index=True)
    
    full_df['position'] = full_df['position'].fillna('UNK')
    full_df['position'] = full_df['position'].replace(['ILB', 'OLB', 'MLB'], 'LB')
    full_df['position'] = full_df['position'].replace(['S', 'FS', 'SS'], 'SAF')

    try:
        player_df = nfl.import_players()
        player_id_map = player_df[['gsis_id', 'espn_id']]
        full_df = pd.merge(full_df, player_id_map, left_on='player_id', right_on='gsis_id', how='left')
    except Exception as e:
        st.warning(f"Could not fetch player mapping for ESPN IDs: {e}")

    conditions = [
        full_df['position'].isin(OFFENSE_POS),
        full_df['position'].isin(DEFENSE_POS),
        full_df['position'].isin(SPECIAL_TEAMS_POS)
    ]
    choices = ['Offense', 'Defense', 'Special Teams']
    full_df['prop_position_group'] = np.select(conditions, choices, default='Other')
    
    rename_map = {
        'player_display_name': 'Player', 'receptions': 'Rec', 'receiving_yards': 'Rec_Yds',
        'receiving_tds': 'Rec_TD', 'carries': 'Rush_Att', 'rushing_yards': 'Rush_Yds',
        'rushing_tds': 'Rush_TD', 'completions': 'Pass_Cmp', 'passing_yards': 'Pass_Yds',
        'passing_tds': 'Pass_TD', 'def_tackles_solo': 'Tackles_Solo', 'def_tackles_with_assist': 'Tackles_Assists',
        'def_tds': 'Defensive_TD', 'def_sacks': 'Sacks', 'def_interceptions': 'Interceptions',
        'def_fumbles_forced': 'Fumbles_Forced', 'fumble_recovery_opp': 'Fumbles_Recovered',
        'def_interception_yards': 'Int_Yds', 'def_fumble_recovery_tds': 'Fumble_Rec_TDs',
        'fg_made': 'FGM', 'pat_made': 'Extra Points Made'
    }
    full_df.rename(columns=rename_map, inplace=True)

    numeric_cols_to_check = [
        'Rec', 'Rec_Yds', 'Rec_TD', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Pass_Cmp', 'Pass_Yds', 'Pass_TD',
        'Tackles_Solo', 'Tackles_Assists', 'Tackles_Total', 'Sacks', 'Interceptions', 'Fumbles_Forced',
        'Fumbles_Recovered', 'Defensive_TD', 'FGM', 'Extra Points Made'
    ]
    for col in numeric_cols_to_check:
        if col not in full_df.columns:
            full_df[col] = 0
    full_df[numeric_cols_to_check] = full_df[numeric_cols_to_check].apply(pd.to_numeric, errors='coerce').fillna(0)

    full_df['TD'] = full_df['Rec_TD'] + full_df['Rush_TD'] + full_df['Pass_TD']
    full_df['Tackles_Total'] = full_df['Tackles_Solo'] + full_df['Tackles_Assists']
    full_df['Defensive_TD'] = full_df.get('Defensive_TD', 0) + full_df.get('Fumble_Rec_TDs', 0)

    return full_df

# -------------------------------
# LAYOUT
# -------------------------------
st.set_page_config(page_title="NFL Prop Analyzer", layout="wide")
st.title("🏈 NFL Player Prop Analyzer")

# Check if data is already loaded in session state to avoid re-running the spinner logic
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading historical NFL data... This may take a moment."):
        full_df = get_nfl_data()
        if not full_df.empty:
            st.session_state.full_df = full_df
            st.session_state.data_loaded = True
            st.rerun() 
else:
    full_df = st.session_state.full_df


if 'data_loaded' in st.session_state:
    st.sidebar.header("Analyze Props")

    position_groups = ['Offense', 'Defense', 'Special Teams']
    selected_position_group = st.sidebar.selectbox("Position Group:", options=position_groups, key="pos_group")

    offense_stats = {"Receiving Yards": "Rec_Yds", "Rushing Yards": "Rush_Yds", "Passing Yards": "Pass_Yds", "Receptions": "Rec", "Receiving TDs": "Rec_TD", "Rushing TDs": "Rush_TD", "Passing TDs": "Pass_TD", "Total Touchdowns": "TD", "Completions": "Pass_Cmp"}
    defense_stats = {"Solo Tackles": "Tackles_Solo", "Assists": "Tackles_Assists", "Tackles + Assists": "Tackles_Total", "Sacks": "Sacks", "Interceptions": "Interceptions", "Forced Fumbles": "Fumbles_Forced", "Fumble Recoveries": "Fumbles_Recovered", "Defensive TDs": "Defensive_TD"}
    special_teams_stats = {"Field Goals Made": "FGM", "Extra Points Made": "Extra Points Made"}

    group_stat_map = offense_stats
    if selected_position_group == "Defense":
        group_stat_map = defense_stats
    elif selected_position_group == "Special Teams":
        group_stat_map = special_teams_stats

    player_options = sorted(full_df[full_df['prop_position_group'] == selected_position_group]['Player'].dropna().unique())
    selected_players = st.sidebar.multiselect("Select Player(s):", options=player_options, key=f"players_{selected_position_group}")
    stat_desc = st.sidebar.selectbox("Stat Type:", options=list(group_stat_map.keys()), key=f"stat_{selected_position_group}")
    prop_line = st.sidebar.number_input("Prop Line:", min_value=0.0, step=0.5, format="%.1f")
    recent_n = st.sidebar.slider("Recent Games to Consider", min_value=1, max_value=10, value=5)
    year_options = sorted(full_df['Year'].dropna().astype(int).unique(), reverse=True)
    years = st.sidebar.multiselect("Years", options=year_options, default=year_options)
    analyze_button = st.sidebar.button("Analyze", type="primary")

    if analyze_button:
        if not selected_players:
            st.warning("Select at least one player.")
        else:
            filtered_df = full_df[(full_df['Player'].isin(selected_players)) & (full_df['Year'].isin([int(y) for y in years])) & (full_df['prop_position_group'] == selected_position_group)]
            st.session_state.filtered_df = filtered_df
            st.session_state.stat_col = group_stat_map[stat_desc]
            st.session_state.stat_desc = stat_desc
            st.session_state.prop_line = prop_line
            st.session_state.recent_n = recent_n

    def format_stat(col_name, value):
        if col_name == 'Sacks':
            rounded_val = round(value, 1)
            if rounded_val == int(rounded_val):
                return int(rounded_val)
            return rounded_val
        else:
            return int(round(value, 0))

    if 'filtered_df' in st.session_state:
        filtered_df = st.session_state.filtered_df
        stat_col = st.session_state.stat_col
        stat_desc = st.session_state.stat_desc
        prop_line = st.session_state.prop_line
        recent_n = st.session_state.recent_n

        if filtered_df.empty:
            st.info("No data available for the selected player(s), position, and year(s).")
        else:
            unique_players = filtered_df['Player'].unique()
            player_cols = st.columns(len(unique_players))

            for i, player_name in enumerate(unique_players):
                with player_cols[i]:
                    player_df = filtered_df[filtered_df['Player'] == player_name].copy()
                    
                    espn_id = player_df['espn_id'].dropna().iloc[0] if not player_df['espn_id'].dropna().empty else None
                    if espn_id:
                        st.image(f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(espn_id)}.png", width=250)
                    st.subheader(player_name)
                    
                    games_played = len(player_df)
                    average = player_df[stat_col].mean() if games_played > 0 else 0
                    median = player_df[stat_col].median() if games_played > 0 else 0
                    std_dev = player_df[stat_col].std() if games_played > 0 else 0
                    recent_df = player_df.sort_values(by=['Year', 'Week'], ascending=False).head(recent_n)
                    recent_average = recent_df[stat_col].mean() if not recent_df.empty else 0
                    last_game = recent_df[stat_col].iloc[0] if not recent_df.empty else 0
                    
                    avg_disp = format_stat(stat_col, average)
                    recent_avg_disp = format_stat(stat_col, recent_average)
                    std_dev_disp = format_stat(stat_col, std_dev)
                    projected = format_stat(stat_col, 0.5 * average + 0.3 * recent_average + 0.2 * last_game)
                    
                    hit_rate = ((player_df[stat_col] > prop_line).sum() / games_played) * 100 if games_played > 0 else 0
                    score = sum([average > prop_line, median > prop_line, recent_average > prop_line, hit_rate > 55])
                    recommendation = "Higher" if score >= 3 else "Lower"
                    
                    metrics_row = st.columns(3)
                    metrics_row[0].metric("Recommendation", recommendation)
                    metrics_row[1].metric("Hit Rate", f"{hit_rate:.1f}%")
                    metrics_row[2].metric("Career Avg", avg_disp)
                    
                    metrics_row_2 = st.columns(3)
                    metrics_row_2[0].metric(f"Recent Avg (L{recent_n})", recent_avg_disp)
                    metrics_row_2[1].metric("Std Dev", std_dev_disp)
                    metrics_row_2[2].metric("Projected", projected)

                    if not player_df.empty:
                        chart_df = player_df.copy()
                        chart_df['Game'] = chart_df['Year'].astype(str) + " - W" + chart_df['Week'].astype(str)
                        chart_df['Prop Line'] = prop_line
                        st.line_chart(chart_df.set_index('Game')[[stat_col, 'Prop Line']], use_container_width=True)

                        recent_df_display = recent_df[['Year', 'Week', stat_col]].copy()
                        recent_df_display.rename(columns={stat_col: stat_desc}, inplace=True)

                        def highlight_over_under(val):
                            return f'background-color: {"#2E8B57" if val > prop_line else "#B22222"}; color: white;'
                        
                        styler = recent_df_display.style.applymap(highlight_over_under, subset=[stat_desc])

                        if stat_col == 'Sacks':
                            styler.format({stat_desc: lambda val: f"{int(val)}" if val == int(val) else f"{val:.1f}"})
                        else:
                            styler.format({stat_desc: "{:.0f}"})
                        
                        # FIX: Use the older, backward-compatible command to hide the index
                        styler.hide_index()

                        st.dataframe(styler, use_container_width=True)

elif 'data_loaded' not in st.session_state:
    st.error("Data could not be loaded. Please check your connection and try again.")
