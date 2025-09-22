# prop_analyzer_dashboard_final_v15.py
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
@st.cache_data(ttl="24h")
def get_nfl_data() -> pd.DataFrame:
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
    player_options = sorted(full_df['Player'].unique())
    selected_players = st.sidebar.multiselect("Select Player(s):", options=player_options, default=None)
    stat_desc = st.sidebar.selectbox("Stat Type:", options=list(stat_map.keys()))
    prop_line = st.sidebar.number_input("Prop Line:", min_value=0.0, step=0.5, format="%.1f")
    recent_n = st.sidebar.slider("Recent Games to Consider", min_value=1, max_value=10, value=5)

    # Force all YEARS_TO_FETCH into the sidebar filter
    year_options = sorted(set(full_df['Year'].unique()).union(YEARS_TO_FETCH))
    years = st.sidebar.multiselect(
        "Years",
        options=year_options,
        default=YEARS_TO_FETCH
    )

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
                (full_df['Year'].isin(years))
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

                        # Round to 1 decimal only if not a whole number
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
