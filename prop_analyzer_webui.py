# prop_analyzer_dashboard_final_v18.py
import streamlit as st
import pandas as pd
import nfl_data_py as nfl

# -------------------------------
# CONFIG
# -------------------------------
YEARS_TO_FETCH = [2022, 2023, 2024, 2025]

# -------------------------------
# DATA LOADING
# -------------------------------
@st.cache_data(ttl=86400)
def fetch_2025_github_data() -> pd.DataFrame:
    """Fetch 2025 weekly player stats directly from NFLverse GitHub."""
    url_2025 = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2025.csv"
    try:
        df = pd.read_csv(url_2025)
        df.columns = df.columns.str.strip()  # Remove any whitespace
        df['Year'] = pd.to_numeric(df['season'], errors='coerce').fillna(0).astype(int)
        df['Week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
        return df
    except Exception as e:
        st.warning(f"Could not fetch 2025 data from GitHub: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_nfl_data() -> pd.DataFrame:
    merged_df_list = []

    # Load weekly data for 2022–2024 via nfl_data_py
    try:
        weekly_df = nfl.import_weekly_data(years=[2022, 2023, 2024])
        weekly_df.columns = weekly_df.columns.str.strip()
        weekly_df['Year'] = pd.to_numeric(weekly_df['season'], errors='coerce').fillna(0).astype(int)
        weekly_df['Week'] = pd.to_numeric(weekly_df['week'], errors='coerce').fillna(0).astype(int)
        merged_df_list.append(weekly_df)
    except Exception as e:
        st.warning(f"Could not fetch 2022–2024 data: {e}")

    # Load 2025 weekly data from GitHub
    if 2025 in YEARS_TO_FETCH:
        df_2025 = fetch_2025_github_data()
        if not df_2025.empty:
            merged_df_list.append(df_2025)

    if not merged_df_list:
        st.error("No NFL data could be loaded.")
        return pd.DataFrame()

    # Concatenate all years
    full_df = pd.concat(merged_df_list, ignore_index=True)

    # Drop duplicate 'season' column if it exists
    if 'season' in full_df.columns:
        full_df.drop(columns=['season'], inplace=True)

    # Load player info and merge
    try:
        player_df = nfl.import_players()
        player_id_map = player_df[['gsis_id', 'espn_id']]
        full_df = pd.merge(
            full_df,
            player_id_map,
            left_on='player_id',
            right_on='gsis_id',
            how='left'
        )
    except Exception as e:
        st.warning(f"Could not fetch player mapping: {e}")

    # Rename columns for consistency
    rename_map = {
        'player_display_name': 'Player',
        'receptions': 'Rec',
        'receiving_yards': 'Rec_Yds',
        'receiving_tds': 'Rec_TD',
        'carries': 'Rush_Att',
        'rushing_yards': 'Rush_Yds',
        'rushing_tds': 'Rush_TD',
        'completions': 'Pass_Cmp',
        'passing_yards': 'Pass_Yds',
        'passing_tds': 'Pass_TD'
    }
    for k, v in rename_map.items():
        if k in full_df.columns:
            full_df.rename(columns={k: v}, inplace=True)

    # Ensure numeric stats and fill NaNs
    stats_cols = ['Rec', 'Rec_Yds', 'Rec_TD', 'Rush_Att', 'Rush_Yds', 'Rush_TD',
                  'Pass_Cmp', 'Pass_Yds', 'Pass_TD']
    for col in stats_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    # Compute total touchdowns
    full_df['TD'] = full_df.get('Rec_TD', 0) + full_df.get('Rush_TD', 0) + full_df.get('Pass_TD', 0)

    return full_df

# -------------------------------
# LAYOUT
# -------------------------------
st.set_page_config(page_title="NFL Prop Analyzer", layout="wide")
st.title("🏈 NFL Player Prop Analyzer")

with st.spinner("Loading historical NFL data..."):
    full_df = get_nfl_data()

if full_df.empty:
    st.error("Data could not be loaded.")
else:
    stat_map = {
        "Receiving Yards": "Rec_Yds",
        "Rushing Yards": "Rush_Yds",
        "Passing Yards": "Pass_Yds",
        "Receptions": "Rec",
        "Total Touchdowns": "TD",
    }

    # -------------------------------
    # SIDEBAR
    # -------------------------------
    st.sidebar.header("Analyze Props")
    player_options = sorted(full_df['Player'].dropna().unique())
    selected_players = st.sidebar.multiselect("Select Player(s):", options=player_options, default=None)
    stat_desc = st.sidebar.selectbox("Stat Type:", options=list(stat_map.keys()))
    prop_line = st.sidebar.number_input("Prop Line:", min_value=0.0, step=0.5, format="%.1f")
    recent_n = st.sidebar.slider("Recent Games to Consider", min_value=1, max_value=10, value=5)

    # Years filtering
    year_options = sorted(full_df['Year'].dropna().astype(int).unique())
    years = st.sidebar.multiselect("Years", options=year_options, default=YEARS_TO_FETCH)
    analyze_button = st.sidebar.button("Analyze", type="primary")

    # -------------------------------
    # FILTER DATA
    # -------------------------------
    if analyze_button:
        if not selected_players:
            st.warning("Select at least one player.")
        else:
            stat_col = stat_map[stat_desc]

            filtered_df = full_df[
                (full_df['Player'].isin(selected_players)) &
                (full_df['Year'].isin([int(y) for y in years]))
            ]

            st.session_state.filtered_df = filtered_df
            st.session_state.stat_col = stat_col
            st.session_state.stat_desc = stat_desc
            st.session_state.prop_line = prop_line
            st.session_state.recent_n = recent_n

    # -------------------------------
    # DISPLAY ANALYSIS
    # -------------------------------
    if 'filtered_df' in st.session_state:
        filtered_df = st.session_state.filtered_df
        stat_col = st.session_state.stat_col
        prop_line = st.session_state.prop_line
        recent_n = st.session_state.recent_n

        if filtered_df.empty:
            st.info("No data available for the selected year(s). 2025 may not have published stats yet.")
        else:
            num_players = len(filtered_df['Player'].unique())
            player_cols = st.columns(num_players)

            for i, player_name in enumerate(filtered_df['Player'].unique()):
                player_df = filtered_df[filtered_df['Player'] == player_name].copy()
                player_df[stat_col] = pd.to_numeric(player_df[stat_col], errors='coerce').fillna(0)

                # Metrics calculation
                games_played = len(player_df)
                average = player_df[stat_col].mean() if games_played > 0 else 0
                median = player_df[stat_col].median() if games_played > 0 else 0
                std_dev = player_df[stat_col].std() if games_played > 0 else 0
                games_over = (player_df[stat_col] > prop_line).sum()
                hit_rate = (games_over / games_played) * 100 if games_played > 0 else 0
                recent_df = player_df.sort_values(by=['Year', 'Week']).tail(recent_n)
                recent_average = recent_df[stat_col].mean() if not recent_df.empty else 0
                last_game = recent_df[stat_col].iloc[-1] if not recent_df.empty else 0
                projected = 0.5*average + 0.3*recent_average + 0.2*last_game
                score = sum([average > prop_line, median > prop_line, recent_average > prop_line, hit_rate > 55])
                recommendation = "Higher" if score >= 3 else "Lower"

                col = player_cols[i]
                with col:
                    # Player headshot
                    espn_id = player_df['espn_id'].dropna().iloc[0] if not player_df['espn_id'].dropna().empty else None
                    if espn_id:
                        headshot_url = f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(espn_id)}.png"
                        st.image(headshot_url, width=300)
                    st.subheader(player_name)

                    # Horizontal metrics
                    metrics_row = st.columns(6)
                    metrics_row[0].metric("Recommendation", recommendation)
                    metrics_row[1].metric("Hit Rate", f"{hit_rate:.1f}%")
                    metrics_row[2].metric("Career Avg", f"{average:.2f}")
                    metrics_row[3].metric(f"Recent Avg (L{recent_n})", f"{recent_average:.2f}")
                    metrics_row[4].metric("Std Dev", f"{std_dev:.2f}")
                    metrics_row[5].metric("Projected", f"{projected:.2f}")

                    # Game-by-game line chart
                    if not player_df.empty:
                        player_df['Game'] = player_df['Year'].astype(str) + " - W" + player_df['Week'].astype(str)
                        chart_df = player_df.copy()
                        chart_df['Prop Line'] = prop_line
                        st.line_chart(chart_df.set_index('Game')[[stat_col, 'Prop Line']], use_container_width=True)

                        # Recent games table
                        display_cols = ['Year', 'Week', 'Rec', 'Rec_Yds', 'Rush_Yds', 'TD']
                        if stat_col not in display_cols:
                            display_cols.append(stat_col)

                        recent_df_display = recent_df[display_cols].copy()
                        numeric_cols = recent_df_display.select_dtypes(include='number').columns

                        for col_name in numeric_cols:
                            recent_df_display[col_name] = recent_df_display[col_name].apply(
                                lambda x: round(x, 1) if not pd.isna(x) and x % 1 != 0 else int(x)
                            )

                        def highlight_over_under(val):
                            try:
                                color = 'green' if val > prop_line else 'red'
                                return f'background-color: {color}'
                            except:
                                return ''

                        st.table(
                            recent_df_display.reset_index(drop=True)
                            .style.applymap(highlight_over_under, subset=[stat_col])
                        )
                    else:
                        st.info("No game data available yet for this player in the selected year(s).")
