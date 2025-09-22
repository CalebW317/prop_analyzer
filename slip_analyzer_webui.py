# prop_analyzer_parlay_v23_pos_filter.py
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from scipy.stats import poisson, norm
import altair as alt

# -------------------------------
# CONFIG
# -------------------------------
YEARS_TO_FETCH = [2022, 2023, 2024]
CURRENT_YEAR = 2025
OL_POSITIONS = ['C', 'G', 'OG', 'T', 'OT', 'LS']

# **RESTORED**: Position group definitions
OFFENSE_POS = ['QB', 'RB', 'WR', 'TE', 'FB']
DEFENSE_POS = ['DE', 'DT', 'NT', 'DL', 'EDGE', 'LB', 'CB', 'SAF', 'DB']
SPECIAL_TEAMS_POS = ['K', 'P']

# -------------------------------
# DATA LOADING & PROCESSING
# -------------------------------

@st.cache_data(ttl=86400)
def get_team_defense_stats(years):
    """
    Calculates avg yards allowed per team, creates multipliers, and ranks defenses.
    """
    weekly_df = nfl.import_weekly_data(years=years)

    game_totals_allowed = weekly_df.groupby(['season', 'opponent_team', 'week']).agg(
        total_pass_yds_allowed_in_game=('passing_yards', 'sum'),
        total_rush_yds_allowed_in_game=('rushing_yards', 'sum')
    ).reset_index()
    game_totals_allowed.rename(columns={'opponent_team': 'defteam'}, inplace=True)

    team_defense_stats = game_totals_allowed.groupby(['defteam', 'season']).agg(
        avg_pass_yds_allowed=('total_pass_yds_allowed_in_game', 'mean'),
        avg_rush_yds_allowed=('total_rush_yds_allowed_in_game', 'mean')
    ).reset_index()

    last_full_season = max(years)
    final_def_stats = team_defense_stats[team_defense_stats['season'] == last_full_season].copy()
    
    final_def_stats['avg_total_yds_allowed'] = final_def_stats['avg_pass_yds_allowed'] + final_def_stats['avg_rush_yds_allowed']
    
    league_avg_pass_yds = final_def_stats['avg_pass_yds_allowed'].mean()
    league_avg_rush_yds = final_def_stats['avg_rush_yds_allowed'].mean()
    league_avg_total_yds = final_def_stats['avg_total_yds_allowed'].mean()

    final_def_stats['pass_def_multiplier'] = final_def_stats['avg_pass_yds_allowed'] / league_avg_pass_yds
    final_def_stats['rush_def_multiplier'] = final_def_stats['avg_rush_yds_allowed'] / league_avg_rush_yds
    final_def_stats['total_def_multiplier'] = final_def_stats['avg_total_yds_allowed'] / league_avg_total_yds
    
    final_def_stats['pass_def_rank'] = final_def_stats['avg_pass_yds_allowed'].rank(method='min').astype(int)
    final_def_stats['rush_def_rank'] = final_def_stats['avg_rush_yds_allowed'].rank(method='min').astype(int)
    final_def_stats['total_def_rank'] = final_def_stats['avg_total_yds_allowed'].rank(method='min').astype(int)
    
    return final_def_stats

@st.cache_data(ttl=86400)
def get_nfl_data():
    """
    Fetches all player, team, and schedule data and creates combo stats.
    """
    team_defense_df = get_team_defense_stats(YEARS_TO_FETCH)

    merged_df_list = []
    years_for_players = YEARS_TO_FETCH + [CURRENT_YEAR]
    for year in years_for_players:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"
        try:
            df = pd.read_csv(url)
            merged_df_list.append(df)
        except:
            st.warning(f"Could not fetch {year} player data.")

    full_df = pd.concat(merged_df_list, ignore_index=True)
    full_df.columns = full_df.columns.str.strip()
    
    try:
        player_info = nfl.import_players()
        player_id_map = player_info[['gsis_id', 'espn_id']]
        full_df = pd.merge(full_df, player_id_map, left_on='player_id', right_on='gsis_id', how='left')
    except Exception as e:
        st.warning(f"Could not fetch player mapping for ESPN IDs: {e}")

    rename_map = {
        'player_display_name': 'Player', 'team': 'team_abbr', 'opponent_team': 'Opponent', 
        'season': 'Year', 'week': 'Week', 'receptions': 'Rec', 'receiving_yards': 'Rec_Yds', 
        'rushing_yards': 'Rush_Yds', 'passing_yards': 'Pass_Yds', 'passing_tds': 'Pass_TD', 
        'rushing_tds': 'Rush_TD', 'receiving_tds': 'Rec_TD', 'completions': 'Pass_Cmp',
        'attempts': 'Pass_Att', 'passing_interceptions': 'Pass_Int', 'fantasy_points_ppr': 'Fantasy_Pts_PPR',
        'fg_made': 'FGM', 'pat_made': 'XPM'
    }
    full_df.rename(columns=rename_map, inplace=True)
    
    calc_cols = ['Rec_TD', 'Rush_TD', 'Rush_Yds', 'Rec_Yds', 'Pass_Yds', 'Pass_Cmp', 'Pass_Att', 'FGM', 'XPM']
    for col in calc_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].fillna(0)
        else:
            full_df[col] = 0

    # **RESTORED**: Clean up positions and create prop_position_group
    full_df['position'] = full_df['position'].fillna('UNK')
    full_df['position'] = full_df['position'].replace(['ILB', 'OLB', 'MLB'], 'LB')
    full_df['position'] = full_df['position'].replace(['S', 'FS', 'SS'], 'SAF')
    
    conditions = [
        full_df['position'].isin(OFFENSE_POS),
        full_df['position'].isin(DEFENSE_POS),
        full_df['position'].isin(SPECIAL_TEAMS_POS)
    ]
    choices = ['Offense', 'Defense', 'Special Teams']
    full_df['prop_position_group'] = np.select(conditions, choices, default='Other')

    full_df['Rush+Rec_Yds'] = full_df['Rush_Yds'] + full_df['Rec_Yds']
    full_df['Pass+Rush_Yds'] = full_df['Pass_Yds'] + full_df['Rush_Yds']
    full_df['Rush+Rec_TDs'] = full_df['Rush_TD'] + full_df['Rec_TD']
    full_df['Comp_Pct'] = 100 * (full_df['Pass_Cmp'] / full_df['Pass_Att']).replace([np.inf, -np.inf], 0).fillna(0)
    full_df['Kicking_Pts'] = (full_df['FGM'] * 3) + full_df['XPM']
    
    team_info = nfl.import_team_desc()
    team_map = team_info[['team_abbr', 'team_name', 'team_logo_espn']]

    return full_df, team_defense_df, team_map

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
# (Helper functions are unchanged)
def remove_leg(index):
    if 0 <= index < len(st.session_state.parlay_slip):
        st.session_state.parlay_slip.pop(index)

def calculate_probability(player_df, stat_col, prop_line, matchup_multiplier=1.0):
    if player_df.empty or stat_col not in player_df.columns or stat_col not in player_df:
        return 0.5, 0, 0

    baseline_avg = player_df[stat_col].mean()
    adjusted_avg = baseline_avg * matchup_multiplier

    if stat_col in ['Rush+Rec_TDs', 'Rec', 'Pass_Att', 'Pass_Int']:
        if adjusted_avg <= 0: return 0.0, baseline_avg, adjusted_avg
        prob_at_or_under = poisson.cdf(k=int(prop_line), mu=adjusted_avg)
        return 1 - prob_at_or_under, baseline_avg, adjusted_avg

    elif any(s in stat_col for s in ['Yds', 'Pct', 'Pts']):
        player_std = player_df[stat_col].std()
        if pd.isna(player_std) or player_std == 0: return 0.5, baseline_avg, adjusted_avg
        prob_at_or_under = norm.cdf(x=prop_line, loc=adjusted_avg, scale=player_std)
        return 1 - prob_at_or_under, baseline_avg, adjusted_avg

    return 0.5, baseline_avg, adjusted_avg

def add_to_parlay(player, stat_desc, stat_col, line, over_under, player_df, opponent, defense_df, team_map):
    if not player or not opponent:
        st.toast("⚠️ Player and Opponent must be selected.", icon="⚠️")
        return

    opponent_abbr = team_map.loc[team_map['team_name'] == opponent, 'team_abbr'].iloc[0]

    multiplier = 1.0
    if stat_col in ['Rush+Rec_Yds', 'Pass+Rush_Yds']:
        multiplier_series = defense_df.loc[defense_df['defteam'] == opponent_abbr, 'total_def_multiplier']
    elif 'Rush' in stat_col:
        multiplier_series = defense_df.loc[defense_df['defteam'] == opponent_abbr, 'rush_def_multiplier']
    elif any(s in stat_col for s in ['Rec', 'Pass']):
        multiplier_series = defense_df.loc[defense_df['defteam'] == opponent_abbr, 'pass_def_multiplier']
    else:
        multiplier_series = None
        
    if multiplier_series is not None and not multiplier_series.empty:
        multiplier = multiplier_series.iloc[0]

    prob_over, _, _ = calculate_probability(player_df, stat_col, line, multiplier)
    prob = prob_over if over_under == "Over" else 1 - prob_over

    st.session_state.parlay_slip.append({
        "player": player, "stat_desc": f"{stat_desc} (vs {opponent_abbr})", "line": line,
        "position": over_under, "prob": prob
    })
    st.toast(f"✅ Added {player} {over_under} to slip!", icon="✅")
    st.session_state.switch_to_parlay = True

# -------------------------------
# MAIN APP LAYOUT
# -------------------------------
st.set_page_config(page_title="NFL Prop Analyzer", layout="wide")
st.title("🏈 NFL Prop & Parlay Analyzer")

if 'data_loaded' not in st.session_state:
    with st.spinner("Loading All NFL Data (Players, Teams, Matchups)..."):
        full_df, defense_df, team_map = get_nfl_data()
        st.session_state.full_df = full_df
        st.session_state.defense_df = defense_df
        st.session_state.team_map = team_map
        st.session_state.data_loaded = True
else:
    full_df = st.session_state.full_df
    defense_df = st.session_state.defense_df
    team_map = st.session_state.team_map

if 'parlay_slip' not in st.session_state:
    st.session_state.parlay_slip = []
if st.session_state.get("switch_to_parlay", False):
    st.session_state.app_mode = "Parlay Calculator"
    del st.session_state.switch_to_parlay

if 'data_loaded' in st.session_state:
    app_mode = st.sidebar.radio("Select Tool", ("Single Prop Analysis", "Parlay Calculator"), key="app_mode")
    st.sidebar.markdown("---")

    if app_mode == "Single Prop Analysis":
        st.sidebar.header("Analyze a Prop")
        
        # **MODIFIED**: Add Position Group selector
        position_groups = ['Offense', 'Defense', 'Special Teams']
        selected_position_group = st.sidebar.selectbox("Position Group", options=position_groups, key="pos_group_single")
        
        # **MODIFIED**: Filter player list based on both OL positions and selected position group
        filtered_df = full_df[
            (~full_df['position'].isin(OL_POSITIONS)) &
            (full_df['prop_position_group'] == selected_position_group)
        ]
        player_options = sorted(filtered_df[filtered_df['Player'].notna()]['Player'].unique())
        
        selected_player = st.sidebar.selectbox("Select Player", options=player_options, key="player_single", index=None, placeholder=f"Select a {selected_position_group} player...")

        stat_map = {
            "Receiving Yards": "Rec_Yds", "Rushing Yards": "Rush_Yds", "Passing Yards": "Pass_Yds",
            "Rush + Rec Yards": "Rush+Rec_Yds", "Pass + Rush Yards": "Pass+Rush_Yds", "Receptions": "Rec",
            "Passing Attempts": "Pass_Att", "Completion %": "Comp_Pct", "Interceptions Thrown": "Pass_Int",
            "Rush + Rec TDs": "Rush+Rec_TDs", "Fantasy Points (PPR)": "Fantasy_Pts_PPR", "Kicking Points": "Kicking_Pts"
        }
        stat_desc = st.sidebar.selectbox("Stat Type", options=list(stat_map.keys()), key="stat_single")

        opponent_list = sorted(team_map['team_name'].unique())
        selected_opponent = st.sidebar.selectbox("Select Opponent", options=opponent_list, key="opponent_single", index=None, placeholder="Select an opponent...")

        prop_line = st.sidebar.number_input("Prop Line", min_value=0.0, step=0.5, format="%.1f", key="prop_single")
        recent_n = st.sidebar.slider("Games for Recent Table", min_value=1, max_value=15, value=5, key="recent_n_single")

        if st.sidebar.button("Analyze Prop", type="primary", use_container_width=True):
            if not selected_player or not selected_opponent:
                st.warning("Please select a Player and an Opponent.")
            else:
                st.session_state.analysis_player = selected_player
                st.session_state.analysis_opponent = selected_opponent
                st.session_state.analysis_data = full_df[full_df['Player'] == selected_player].copy()
        
        # Analysis display logic remains the same...
        if 'analysis_player' in st.session_state and st.session_state.analysis_player == selected_player:
            player_df_single = st.session_state.analysis_data
            stat_col = stat_map.get(stat_desc)
            if player_df_single.empty or not stat_col or stat_col not in player_df_single.columns:
                st.info("No data for selected player or stat.")
            else:
                vis_col, analysis_col = st.columns([1, 2.5])

                with vis_col:
                    espn_id = player_df_single['espn_id'].dropna().iloc[0] if not player_df_single['espn_id'].dropna().empty else None
                    if espn_id:
                        st.image(f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(espn_id)}.png", width=220)
                    opponent_abbr_for_logo = team_map.loc[team_map['team_name'] == st.session_state.analysis_opponent, 'team_abbr'].iloc[0]
                    logo_url = team_map.loc[team_map['team_abbr'] == opponent_abbr_for_logo, 'team_logo_espn'].iloc[0]
                    if logo_url:
                        st.image(logo_url, width=160)

                with analysis_col:
                    st.subheader(f"{selected_player} ({stat_desc}) vs. {st.session_state.analysis_opponent}")
                    opponent_abbr = team_map.loc[team_map['team_name'] == st.session_state.analysis_opponent, 'team_abbr'].iloc[0]
                    matchup_multiplier, opp_rank, opp_avg_allowed = 1.0, "N/A", 0
                    
                    d_rank_label, d_avg_label, show_defense_kpis = "D-Rank", "Avg Yds Allowed", True
                    if stat_col in ['Rush+Rec_Yds', 'Pass+Rush_Yds']:
                        cols_to_check = ['total_def_rank', 'avg_total_yds_allowed', 'total_def_multiplier']
                        d_rank_label, d_avg_label = "Total D-Rank", "Avg Total Yds Allowed"
                    elif 'Rush' in stat_col:
                        cols_to_check = ['rush_def_rank', 'avg_rush_yds_allowed', 'rush_def_multiplier']
                        d_rank_label, d_avg_label = "Rush D-Rank", "Avg Rush Yds Allowed"
                    elif any(s in stat_col for s in ['Rec', 'Pass']):
                        cols_to_check = ['pass_def_rank', 'avg_pass_yds_allowed', 'pass_def_multiplier']
                        d_rank_label, d_avg_label = "Pass D-Rank", "Avg Pass Yds Allowed"
                    else:
                        cols_to_check, show_defense_kpis = None, False

                    if cols_to_check:
                        opp_stats = defense_df[defense_df['defteam'] == opponent_abbr]
                        if not opp_stats.empty and all(col in opp_stats.columns for col in cols_to_check):
                            opp_rank = opp_stats[cols_to_check[0]].iloc[0]
                            opp_avg_allowed = opp_stats[cols_to_check[1]].iloc[0]
                            matchup_multiplier = opp_stats[cols_to_check[2]].iloc[0]
                        else:
                            show_defense_kpis = False
                    
                    prob_over, baseline_avg, adjusted_avg = calculate_probability(player_df_single, stat_col, prop_line, matchup_multiplier)
                    
                    hit_rate_over, hit_rate_under = 0, 0
                    games_played = player_df_single[stat_col].count()
                    if games_played > 0:
                        hit_rate_over = (player_df_single[stat_col] > prop_line).sum() / games_played * 100
                        hit_rate_under = (player_df_single[stat_col] < prop_line).sum() / games_played * 100
                    
                    st.markdown("##### Matchup-Adjusted Analysis")
                    kpi_cols = st.columns(4)
                    kpi_cols[0].metric("Adjusted Projection", f"{adjusted_avg:.1f}")
                    kpi_cols[1].metric("Model Prob (Over)", f"{prob_over:.1%}")
                    kpi_cols[2].metric("Hit Rate (Over)", f"{hit_rate_over:.1f}%")
                    kpi_cols[3].metric("Hit Rate (Under)", f"{hit_rate_under:.1f}%")
                    
                    st.markdown("##### Player & Opponent Baselines")
                    base_cols = st.columns(3 if show_defense_kpis else 1)
                    base_cols[0].metric("Player Career Avg", f"{baseline_avg:.1f}")
                    if show_defense_kpis:
                        base_cols[1].metric(f"{opponent_abbr} {d_rank_label}", f"#{opp_rank}", help=f"Rank in yards allowed. #1 is best defense, #32 is worst.")
                        base_cols[2].metric(f"{opponent_abbr} {d_avg_label}", f"{opp_avg_allowed:.1f}")

                add_col1, add_col2 = st.columns(2)
                with add_col1:
                    add_col1.button(f"Add **Over** to Parlay (Prob: {prob_over:.1%})",
                                    on_click=add_to_parlay,
                                    args=(selected_player, stat_desc, stat_col, prop_line, "Over", player_df_single, st.session_state.analysis_opponent, defense_df, team_map),
                                    type="primary", use_container_width=True)
                with add_col2:
                    add_col2.button(f"Add **Under** to Parlay (Prob: {1-prob_over:.1%})",
                                    on_click=add_to_parlay,
                                    args=(selected_player, stat_desc, stat_col, prop_line, "Under", player_df_single, st.session_state.analysis_opponent, defense_df, team_map),
                                    type="secondary", use_container_width=True)
                
                st.markdown("---")
                st.subheader("Performance vs. Prop Line & Opponent Average")
                chart_df = player_df_single.copy().dropna(subset=[stat_col])
                chart_df['Game'] = chart_df['Year'].astype(str) + " - W" + chart_df['Week'].astype(str)
                
                opp_avg_label_chart = f'Opponent {d_avg_label} ({opponent_abbr})'
                chart_df['Prop Line'] = prop_line
                if show_defense_kpis:
                    chart_df[opp_avg_label_chart] = opp_avg_allowed

                source = chart_df.melt(id_vars=['Game'], value_vars=[stat_col], var_name='Stat', value_name='Value')
                source['Stat'] = stat_desc
                
                line = alt.Chart(source).mark_line(point=True, size=3).encode(
                    x=alt.X('Game:N', sort=None, title="Game"),
                    y=alt.Y('Value:Q', title=stat_desc, scale=alt.Scale(zero=False)),
                    color=alt.Color('Stat:N', legend=alt.Legend(title=None)),
                    tooltip=['Game', 'Value']
                ).properties(height=350)
                
                prop_line_rule = alt.Chart(pd.DataFrame({'y': [prop_line]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
                prop_line_text = prop_line_rule.mark_text(align='left', dx=5, dy=-10, text=f'Prop Line: {prop_line}', color='white').encode(y='y')
                
                final_chart = line + prop_line_rule + prop_line_text
                if show_defense_kpis:
                    opp_avg_rule = alt.Chart(pd.DataFrame({'y': [opp_avg_allowed]})).mark_rule(color='#757575').encode(y='y')
                    opp_avg_text = opp_avg_rule.mark_text(align='left', dx=5, dy=10, text=f'{opp_avg_label_chart}: {opp_avg_allowed:.1f}', color='white').encode(y='y')
                    final_chart += opp_avg_rule + opp_avg_text
                    
                st.altair_chart(final_chart, use_container_width=True)

                st.subheader(f"Recent {recent_n} Games")
                recent_df_display = player_df_single.sort_values(by=['Year', 'Week'], ascending=False)[['Year', 'Week', 'Opponent', stat_col]].head(recent_n)
                recent_df_display.rename(columns={stat_col: stat_desc}, inplace=True)
                
                def highlight_over_under(val):
                    color = "#2E8B57" if val > prop_line else "#B22222"
                    return f'background-color: {color}; color: white;'

                st.dataframe(recent_df_display.style.applymap(highlight_over_under, subset=[stat_desc]).format({stat_desc: "{:.0f}"}),
                             hide_index=True, use_container_width=True)

    elif app_mode == "Parlay Calculator":
        st.sidebar.header("Add a Prop to Parlay")
        
        # **MODIFIED**: Add Position Group selector
        position_groups_parlay = ['Offense', 'Defense', 'Special Teams']
        selected_position_group_parlay = st.sidebar.selectbox("Position Group", options=position_groups_parlay, key="pos_group_parlay")

        # **MODIFIED**: Filter player list based on both OL positions and selected position group
        skill_players_df_parlay = full_df[
            (~full_df['position'].isin(OL_POSITIONS)) &
            (full_df['prop_position_group'] == selected_position_group_parlay)
        ]
        parlay_player_options = sorted(skill_players_df_parlay[skill_players_df_parlay['Player'].notna()]['Player'].unique())
        
        parlay_player = st.sidebar.selectbox("Player", options=parlay_player_options, key="player_parlay", index=None, placeholder=f"Select a {selected_position_group_parlay} player...")
        
        parlay_stat_map = {
            "Receiving Yards": "Rec_Yds", "Rushing Yards": "Rush_Yds", "Passing Yards": "Pass_Yds",
            "Rush + Rec Yards": "Rush+Rec_Yds", "Pass + Rush Yards": "Pass+Rush_Yds", "Receptions": "Rec",
            "Passing Attempts": "Pass_Att", "Completion %": "Comp_Pct", "Interceptions Thrown": "Pass_Int",
            "Rush + Rec TDs": "Rush+Rec_TDs", "Fantasy Points (PPR)": "Fantasy_Pts_PPR", "Kicking Points": "Kicking_Pts"
        }
        parlay_stat_desc = st.sidebar.selectbox("Stat", options=list(parlay_stat_map.keys()), key="stat_parlay")
        
        parlay_opponent_list = sorted(team_map['team_name'].unique())
        parlay_opponent = st.sidebar.selectbox("Opponent", options=parlay_opponent_list, key="opponent_parlay", index=None)
        parlay_line = st.sidebar.number_input("Prop Line", min_value=0.0, step=0.5, value=50.5, format="%.1f", key="line_parlay")
        parlay_over_under = st.sidebar.radio("Over/Under", ["Over", "Under"], horizontal=True, key="ou_parlay")

        if st.sidebar.button("Add to Parlay Slip", type="primary"):
            player_df = full_df[full_df['Player'] == parlay_player]
            add_to_parlay(parlay_player, parlay_stat_desc, parlay_stat_map.get(parlay_stat_desc), parlay_line, parlay_over_under, player_df, parlay_opponent, defense_df, team_map)
            
        st.header("Current Parlay Slip")
        if not st.session_state.parlay_slip:
            st.info("Add a prop to begin.")
        else:
            total_prob = 1.0
            for i, leg in enumerate(st.session_state.parlay_slip):
                total_prob *= leg['prob']
                col1, col2, col3 = st.columns([5, 2, 1])
                with col1:
                    st.markdown(f"**{i+1}. {leg['player']}** | {leg['stat_desc']} **{leg['position']} {leg['line']}**")
                with col2:
                    st.markdown(f"**Prob: `{leg['prob']:.1%}`**")
                with col3:
                    st.button("Remove", key=f"remove_btn_{i}", on_click=remove_leg, args=(i,))
            st.markdown("---")
            implied_odds_decimal = (1 / total_prob) if total_prob > 0 else 0
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Combined Parlay Probability", f"{total_prob:.2%}")
            if implied_odds_decimal >= 2.0:
                american_odds = f"+{100 * (implied_odds_decimal - 1):.0f}"
            elif implied_odds_decimal > 1.0:
                 american_odds = f"{-100 / (implied_odds_decimal - 1):.0f}"
            else:
                american_odds = "N/A"
            p_col2.metric("Implied American Odds", american_odds)
            if st.button("Clear Parlay Slip"):
                st.session_state.parlay_slip = []
                st.rerun()# prop_analyzer_parlay_v23_pos_filter.py
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from scipy.stats import poisson, norm
import altair as alt

# -------------------------------
# CONFIG
# -------------------------------
YEARS_TO_FETCH = [2022, 2023, 2024]
CURRENT_YEAR = 2025
OL_POSITIONS = ['C', 'G', 'OG', 'T', 'OT', 'LS']

# **RESTORED**: Position group definitions
OFFENSE_POS = ['QB', 'RB', 'WR', 'TE', 'FB']
DEFENSE_POS = ['DE', 'DT', 'NT', 'DL', 'EDGE', 'LB', 'CB', 'SAF', 'DB']
SPECIAL_TEAMS_POS = ['K', 'P']

# -------------------------------
# DATA LOADING & PROCESSING
# -------------------------------

@st.cache_data(ttl=86400)
def get_team_defense_stats(years):
    """
    Calculates avg yards allowed per team, creates multipliers, and ranks defenses.
    """
    weekly_df = nfl.import_weekly_data(years=years)

    game_totals_allowed = weekly_df.groupby(['season', 'opponent_team', 'week']).agg(
        total_pass_yds_allowed_in_game=('passing_yards', 'sum'),
        total_rush_yds_allowed_in_game=('rushing_yards', 'sum')
    ).reset_index()
    game_totals_allowed.rename(columns={'opponent_team': 'defteam'}, inplace=True)

    team_defense_stats = game_totals_allowed.groupby(['defteam', 'season']).agg(
        avg_pass_yds_allowed=('total_pass_yds_allowed_in_game', 'mean'),
        avg_rush_yds_allowed=('total_rush_yds_allowed_in_game', 'mean')
    ).reset_index()

    last_full_season = max(years)
    final_def_stats = team_defense_stats[team_defense_stats['season'] == last_full_season].copy()
    
    final_def_stats['avg_total_yds_allowed'] = final_def_stats['avg_pass_yds_allowed'] + final_def_stats['avg_rush_yds_allowed']
    
    league_avg_pass_yds = final_def_stats['avg_pass_yds_allowed'].mean()
    league_avg_rush_yds = final_def_stats['avg_rush_yds_allowed'].mean()
    league_avg_total_yds = final_def_stats['avg_total_yds_allowed'].mean()

    final_def_stats['pass_def_multiplier'] = final_def_stats['avg_pass_yds_allowed'] / league_avg_pass_yds
    final_def_stats['rush_def_multiplier'] = final_def_stats['avg_rush_yds_allowed'] / league_avg_rush_yds
    final_def_stats['total_def_multiplier'] = final_def_stats['avg_total_yds_allowed'] / league_avg_total_yds
    
    final_def_stats['pass_def_rank'] = final_def_stats['avg_pass_yds_allowed'].rank(method='min').astype(int)
    final_def_stats['rush_def_rank'] = final_def_stats['avg_rush_yds_allowed'].rank(method='min').astype(int)
    final_def_stats['total_def_rank'] = final_def_stats['avg_total_yds_allowed'].rank(method='min').astype(int)
    
    return final_def_stats

@st.cache_data(ttl=86400)
def get_nfl_data():
    """
    Fetches all player, team, and schedule data and creates combo stats.
    """
    team_defense_df = get_team_defense_stats(YEARS_TO_FETCH)

    merged_df_list = []
    years_for_players = YEARS_TO_FETCH + [CURRENT_YEAR]
    for year in years_for_players:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"
        try:
            df = pd.read_csv(url)
            merged_df_list.append(df)
        except:
            st.warning(f"Could not fetch {year} player data.")

    full_df = pd.concat(merged_df_list, ignore_index=True)
    full_df.columns = full_df.columns.str.strip()
    
    try:
        player_info = nfl.import_players()
        player_id_map = player_info[['gsis_id', 'espn_id']]
        full_df = pd.merge(full_df, player_id_map, left_on='player_id', right_on='gsis_id', how='left')
    except Exception as e:
        st.warning(f"Could not fetch player mapping for ESPN IDs: {e}")

    rename_map = {
        'player_display_name': 'Player', 'team': 'team_abbr', 'opponent_team': 'Opponent', 
        'season': 'Year', 'week': 'Week', 'receptions': 'Rec', 'receiving_yards': 'Rec_Yds', 
        'rushing_yards': 'Rush_Yds', 'passing_yards': 'Pass_Yds', 'passing_tds': 'Pass_TD', 
        'rushing_tds': 'Rush_TD', 'receiving_tds': 'Rec_TD', 'completions': 'Pass_Cmp',
        'attempts': 'Pass_Att', 'passing_interceptions': 'Pass_Int', 'fantasy_points_ppr': 'Fantasy_Pts_PPR',
        'fg_made': 'FGM', 'pat_made': 'XPM'
    }
    full_df.rename(columns=rename_map, inplace=True)
    
    calc_cols = ['Rec_TD', 'Rush_TD', 'Rush_Yds', 'Rec_Yds', 'Pass_Yds', 'Pass_Cmp', 'Pass_Att', 'FGM', 'XPM']
    for col in calc_cols:
        if col in full_df.columns:
            full_df[col] = full_df[col].fillna(0)
        else:
            full_df[col] = 0

    # **RESTORED**: Clean up positions and create prop_position_group
    full_df['position'] = full_df['position'].fillna('UNK')
    full_df['position'] = full_df['position'].replace(['ILB', 'OLB', 'MLB'], 'LB')
    full_df['position'] = full_df['position'].replace(['S', 'FS', 'SS'], 'SAF')
    
    conditions = [
        full_df['position'].isin(OFFENSE_POS),
        full_df['position'].isin(DEFENSE_POS),
        full_df['position'].isin(SPECIAL_TEAMS_POS)
    ]
    choices = ['Offense', 'Defense', 'Special Teams']
    full_df['prop_position_group'] = np.select(conditions, choices, default='Other')

    full_df['Rush+Rec_Yds'] = full_df['Rush_Yds'] + full_df['Rec_Yds']
    full_df['Pass+Rush_Yds'] = full_df['Pass_Yds'] + full_df['Rush_Yds']
    full_df['Rush+Rec_TDs'] = full_df['Rush_TD'] + full_df['Rec_TD']
    full_df['Comp_Pct'] = 100 * (full_df['Pass_Cmp'] / full_df['Pass_Att']).replace([np.inf, -np.inf], 0).fillna(0)
    full_df['Kicking_Pts'] = (full_df['FGM'] * 3) + full_df['XPM']
    
    team_info = nfl.import_team_desc()
    team_map = team_info[['team_abbr', 'team_name', 'team_logo_espn']]

    return full_df, team_defense_df, team_map

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
# (Helper functions are unchanged)
def remove_leg(index):
    if 0 <= index < len(st.session_state.parlay_slip):
        st.session_state.parlay_slip.pop(index)

def calculate_probability(player_df, stat_col, prop_line, matchup_multiplier=1.0):
    if player_df.empty or stat_col not in player_df.columns or stat_col not in player_df:
        return 0.5, 0, 0

    baseline_avg = player_df[stat_col].mean()
    adjusted_avg = baseline_avg * matchup_multiplier

    if stat_col in ['Rush+Rec_TDs', 'Rec', 'Pass_Att', 'Pass_Int']:
        if adjusted_avg <= 0: return 0.0, baseline_avg, adjusted_avg
        prob_at_or_under = poisson.cdf(k=int(prop_line), mu=adjusted_avg)
        return 1 - prob_at_or_under, baseline_avg, adjusted_avg

    elif any(s in stat_col for s in ['Yds', 'Pct', 'Pts']):
        player_std = player_df[stat_col].std()
        if pd.isna(player_std) or player_std == 0: return 0.5, baseline_avg, adjusted_avg
        prob_at_or_under = norm.cdf(x=prop_line, loc=adjusted_avg, scale=player_std)
        return 1 - prob_at_or_under, baseline_avg, adjusted_avg

    return 0.5, baseline_avg, adjusted_avg

def add_to_parlay(player, stat_desc, stat_col, line, over_under, player_df, opponent, defense_df, team_map):
    if not player or not opponent:
        st.toast("⚠️ Player and Opponent must be selected.", icon="⚠️")
        return

    opponent_abbr = team_map.loc[team_map['team_name'] == opponent, 'team_abbr'].iloc[0]

    multiplier = 1.0
    if stat_col in ['Rush+Rec_Yds', 'Pass+Rush_Yds']:
        multiplier_series = defense_df.loc[defense_df['defteam'] == opponent_abbr, 'total_def_multiplier']
    elif 'Rush' in stat_col:
        multiplier_series = defense_df.loc[defense_df['defteam'] == opponent_abbr, 'rush_def_multiplier']
    elif any(s in stat_col for s in ['Rec', 'Pass']):
        multiplier_series = defense_df.loc[defense_df['defteam'] == opponent_abbr, 'pass_def_multiplier']
    else:
        multiplier_series = None
        
    if multiplier_series is not None and not multiplier_series.empty:
        multiplier = multiplier_series.iloc[0]

    prob_over, _, _ = calculate_probability(player_df, stat_col, line, multiplier)
    prob = prob_over if over_under == "Over" else 1 - prob_over

    st.session_state.parlay_slip.append({
        "player": player, "stat_desc": f"{stat_desc} (vs {opponent_abbr})", "line": line,
        "position": over_under, "prob": prob
    })
    st.toast(f"✅ Added {player} {over_under} to slip!", icon="✅")
    st.session_state.switch_to_parlay = True

# -------------------------------
# MAIN APP LAYOUT
# -------------------------------
st.set_page_config(page_title="NFL Prop Analyzer", layout="wide")
st.title("🏈 NFL Prop & Parlay Analyzer")

if 'data_loaded' not in st.session_state:
    with st.spinner("Loading All NFL Data (Players, Teams, Matchups)..."):
        full_df, defense_df, team_map = get_nfl_data()
        st.session_state.full_df = full_df
        st.session_state.defense_df = defense_df
        st.session_state.team_map = team_map
        st.session_state.data_loaded = True
else:
    full_df = st.session_state.full_df
    defense_df = st.session_state.defense_df
    team_map = st.session_state.team_map

if 'parlay_slip' not in st.session_state:
    st.session_state.parlay_slip = []
if st.session_state.get("switch_to_parlay", False):
    st.session_state.app_mode = "Parlay Calculator"
    del st.session_state.switch_to_parlay

if 'data_loaded' in st.session_state:
    app_mode = st.sidebar.radio("Select Tool", ("Single Prop Analysis", "Parlay Calculator"), key="app_mode")
    st.sidebar.markdown("---")

    if app_mode == "Single Prop Analysis":
        st.sidebar.header("Analyze a Prop")
        
        # **MODIFIED**: Add Position Group selector
        position_groups = ['Offense', 'Defense', 'Special Teams']
        selected_position_group = st.sidebar.selectbox("Position Group", options=position_groups, key="pos_group_single")
        
        # **MODIFIED**: Filter player list based on both OL positions and selected position group
        filtered_df = full_df[
            (~full_df['position'].isin(OL_POSITIONS)) &
            (full_df['prop_position_group'] == selected_position_group)
        ]
        player_options = sorted(filtered_df[filtered_df['Player'].notna()]['Player'].unique())
        
        selected_player = st.sidebar.selectbox("Select Player", options=player_options, key="player_single", index=None, placeholder=f"Select a {selected_position_group} player...")

        stat_map = {
            "Receiving Yards": "Rec_Yds", "Rushing Yards": "Rush_Yds", "Passing Yards": "Pass_Yds",
            "Rush + Rec Yards": "Rush+Rec_Yds", "Pass + Rush Yards": "Pass+Rush_Yds", "Receptions": "Rec",
            "Passing Attempts": "Pass_Att", "Completion %": "Comp_Pct", "Interceptions Thrown": "Pass_Int",
            "Rush + Rec TDs": "Rush+Rec_TDs", "Fantasy Points (PPR)": "Fantasy_Pts_PPR", "Kicking Points": "Kicking_Pts"
        }
        stat_desc = st.sidebar.selectbox("Stat Type", options=list(stat_map.keys()), key="stat_single")

        opponent_list = sorted(team_map['team_name'].unique())
        selected_opponent = st.sidebar.selectbox("Select Opponent", options=opponent_list, key="opponent_single", index=None, placeholder="Select an opponent...")

        prop_line = st.sidebar.number_input("Prop Line", min_value=0.0, step=0.5, format="%.1f", key="prop_single")
        recent_n = st.sidebar.slider("Games for Recent Table", min_value=1, max_value=15, value=5, key="recent_n_single")

        if st.sidebar.button("Analyze Prop", type="primary", use_container_width=True):
            if not selected_player or not selected_opponent:
                st.warning("Please select a Player and an Opponent.")
            else:
                st.session_state.analysis_player = selected_player
                st.session_state.analysis_opponent = selected_opponent
                st.session_state.analysis_data = full_df[full_df['Player'] == selected_player].copy()
        
        # Analysis display logic remains the same...
        if 'analysis_player' in st.session_state and st.session_state.analysis_player == selected_player:
            player_df_single = st.session_state.analysis_data
            stat_col = stat_map.get(stat_desc)
            if player_df_single.empty or not stat_col or stat_col not in player_df_single.columns:
                st.info("No data for selected player or stat.")
            else:
                vis_col, analysis_col = st.columns([1, 2.5])

                with vis_col:
                    espn_id = player_df_single['espn_id'].dropna().iloc[0] if not player_df_single['espn_id'].dropna().empty else None
                    if espn_id:
                        st.image(f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(espn_id)}.png", width=220)
                    opponent_abbr_for_logo = team_map.loc[team_map['team_name'] == st.session_state.analysis_opponent, 'team_abbr'].iloc[0]
                    logo_url = team_map.loc[team_map['team_abbr'] == opponent_abbr_for_logo, 'team_logo_espn'].iloc[0]
                    if logo_url:
                        st.image(logo_url, width=160)

                with analysis_col:
                    st.subheader(f"{selected_player} ({stat_desc}) vs. {st.session_state.analysis_opponent}")
                    opponent_abbr = team_map.loc[team_map['team_name'] == st.session_state.analysis_opponent, 'team_abbr'].iloc[0]
                    matchup_multiplier, opp_rank, opp_avg_allowed = 1.0, "N/A", 0
                    
                    d_rank_label, d_avg_label, show_defense_kpis = "D-Rank", "Avg Yds Allowed", True
                    if stat_col in ['Rush+Rec_Yds', 'Pass+Rush_Yds']:
                        cols_to_check = ['total_def_rank', 'avg_total_yds_allowed', 'total_def_multiplier']
                        d_rank_label, d_avg_label = "Total D-Rank", "Avg Total Yds Allowed"
                    elif 'Rush' in stat_col:
                        cols_to_check = ['rush_def_rank', 'avg_rush_yds_allowed', 'rush_def_multiplier']
                        d_rank_label, d_avg_label = "Rush D-Rank", "Avg Rush Yds Allowed"
                    elif any(s in stat_col for s in ['Rec', 'Pass']):
                        cols_to_check = ['pass_def_rank', 'avg_pass_yds_allowed', 'pass_def_multiplier']
                        d_rank_label, d_avg_label = "Pass D-Rank", "Avg Pass Yds Allowed"
                    else:
                        cols_to_check, show_defense_kpis = None, False

                    if cols_to_check:
                        opp_stats = defense_df[defense_df['defteam'] == opponent_abbr]
                        if not opp_stats.empty and all(col in opp_stats.columns for col in cols_to_check):
                            opp_rank = opp_stats[cols_to_check[0]].iloc[0]
                            opp_avg_allowed = opp_stats[cols_to_check[1]].iloc[0]
                            matchup_multiplier = opp_stats[cols_to_check[2]].iloc[0]
                        else:
                            show_defense_kpis = False
                    
                    prob_over, baseline_avg, adjusted_avg = calculate_probability(player_df_single, stat_col, prop_line, matchup_multiplier)
                    
                    hit_rate_over, hit_rate_under = 0, 0
                    games_played = player_df_single[stat_col].count()
                    if games_played > 0:
                        hit_rate_over = (player_df_single[stat_col] > prop_line).sum() / games_played * 100
                        hit_rate_under = (player_df_single[stat_col] < prop_line).sum() / games_played * 100
                    
                    st.markdown("##### Matchup-Adjusted Analysis")
                    kpi_cols = st.columns(4)
                    kpi_cols[0].metric("Adjusted Projection", f"{adjusted_avg:.1f}")
                    kpi_cols[1].metric("Model Prob (Over)", f"{prob_over:.1%}")
                    kpi_cols[2].metric("Hit Rate (Over)", f"{hit_rate_over:.1f}%")
                    kpi_cols[3].metric("Hit Rate (Under)", f"{hit_rate_under:.1f}%")
                    
                    st.markdown("##### Player & Opponent Baselines")
                    base_cols = st.columns(3 if show_defense_kpis else 1)
                    base_cols[0].metric("Player Career Avg", f"{baseline_avg:.1f}")
                    if show_defense_kpis:
                        base_cols[1].metric(f"{opponent_abbr} {d_rank_label}", f"#{opp_rank}", help=f"Rank in yards allowed. #1 is best defense, #32 is worst.")
                        base_cols[2].metric(f"{opponent_abbr} {d_avg_label}", f"{opp_avg_allowed:.1f}")

                add_col1, add_col2 = st.columns(2)
                with add_col1:
                    add_col1.button(f"Add **Over** to Parlay (Prob: {prob_over:.1%})",
                                    on_click=add_to_parlay,
                                    args=(selected_player, stat_desc, stat_col, prop_line, "Over", player_df_single, st.session_state.analysis_opponent, defense_df, team_map),
                                    type="primary", use_container_width=True)
                with add_col2:
                    add_col2.button(f"Add **Under** to Parlay (Prob: {1-prob_over:.1%})",
                                    on_click=add_to_parlay,
                                    args=(selected_player, stat_desc, stat_col, prop_line, "Under", player_df_single, st.session_state.analysis_opponent, defense_df, team_map),
                                    type="secondary", use_container_width=True)
                
                st.markdown("---")
                st.subheader("Performance vs. Prop Line & Opponent Average")
                chart_df = player_df_single.copy().dropna(subset=[stat_col])
                chart_df['Game'] = chart_df['Year'].astype(str) + " - W" + chart_df['Week'].astype(str)
                
                opp_avg_label_chart = f'Opponent {d_avg_label} ({opponent_abbr})'
                chart_df['Prop Line'] = prop_line
                if show_defense_kpis:
                    chart_df[opp_avg_label_chart] = opp_avg_allowed

                source = chart_df.melt(id_vars=['Game'], value_vars=[stat_col], var_name='Stat', value_name='Value')
                source['Stat'] = stat_desc
                
                line = alt.Chart(source).mark_line(point=True, size=3).encode(
                    x=alt.X('Game:N', sort=None, title="Game"),
                    y=alt.Y('Value:Q', title=stat_desc, scale=alt.Scale(zero=False)),
                    color=alt.Color('Stat:N', legend=alt.Legend(title=None)),
                    tooltip=['Game', 'Value']
                ).properties(height=350)
                
                prop_line_rule = alt.Chart(pd.DataFrame({'y': [prop_line]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
                prop_line_text = prop_line_rule.mark_text(align='left', dx=5, dy=-10, text=f'Prop Line: {prop_line}', color='white').encode(y='y')
                
                final_chart = line + prop_line_rule + prop_line_text
                if show_defense_kpis:
                    opp_avg_rule = alt.Chart(pd.DataFrame({'y': [opp_avg_allowed]})).mark_rule(color='#757575').encode(y='y')
                    opp_avg_text = opp_avg_rule.mark_text(align='left', dx=5, dy=10, text=f'{opp_avg_label_chart}: {opp_avg_allowed:.1f}', color='white').encode(y='y')
                    final_chart += opp_avg_rule + opp_avg_text
                    
                st.altair_chart(final_chart, use_container_width=True)

                st.subheader(f"Recent {recent_n} Games")
                recent_df_display = player_df_single.sort_values(by=['Year', 'Week'], ascending=False)[['Year', 'Week', 'Opponent', stat_col]].head(recent_n)
                recent_df_display.rename(columns={stat_col: stat_desc}, inplace=True)
                
                def highlight_over_under(val):
                    color = "#2E8B57" if val > prop_line else "#B22222"
                    return f'background-color: {color}; color: white;'

                st.dataframe(recent_df_display.style.applymap(highlight_over_under, subset=[stat_desc]).format({stat_desc: "{:.0f}"}),
                             hide_index=True, use_container_width=True)

    elif app_mode == "Parlay Calculator":
        st.sidebar.header("Add a Prop to Parlay")
        
        # **MODIFIED**: Add Position Group selector
        position_groups_parlay = ['Offense', 'Defense', 'Special Teams']
        selected_position_group_parlay = st.sidebar.selectbox("Position Group", options=position_groups_parlay, key="pos_group_parlay")

        # **MODIFIED**: Filter player list based on both OL positions and selected position group
        skill_players_df_parlay = full_df[
            (~full_df['position'].isin(OL_POSITIONS)) &
            (full_df['prop_position_group'] == selected_position_group_parlay)
        ]
        parlay_player_options = sorted(skill_players_df_parlay[skill_players_df_parlay['Player'].notna()]['Player'].unique())
        
        parlay_player = st.sidebar.selectbox("Player", options=parlay_player_options, key="player_parlay", index=None, placeholder=f"Select a {selected_position_group_parlay} player...")
        
        parlay_stat_map = {
            "Receiving Yards": "Rec_Yds", "Rushing Yards": "Rush_Yds", "Passing Yards": "Pass_Yds",
            "Rush + Rec Yards": "Rush+Rec_Yds", "Pass + Rush Yards": "Pass+Rush_Yds", "Receptions": "Rec",
            "Passing Attempts": "Pass_Att", "Completion %": "Comp_Pct", "Interceptions Thrown": "Pass_Int",
            "Rush + Rec TDs": "Rush+Rec_TDs", "Fantasy Points (PPR)": "Fantasy_Pts_PPR", "Kicking Points": "Kicking_Pts"
        }
        parlay_stat_desc = st.sidebar.selectbox("Stat", options=list(parlay_stat_map.keys()), key="stat_parlay")
        
        parlay_opponent_list = sorted(team_map['team_name'].unique())
        parlay_opponent = st.sidebar.selectbox("Opponent", options=parlay_opponent_list, key="opponent_parlay", index=None)
        parlay_line = st.sidebar.number_input("Prop Line", min_value=0.0, step=0.5, value=50.5, format="%.1f", key="line_parlay")
        parlay_over_under = st.sidebar.radio("Over/Under", ["Over", "Under"], horizontal=True, key="ou_parlay")

        if st.sidebar.button("Add to Parlay Slip", type="primary"):
            player_df = full_df[full_df['Player'] == parlay_player]
            add_to_parlay(parlay_player, parlay_stat_desc, parlay_stat_map.get(parlay_stat_desc), parlay_line, parlay_over_under, player_df, parlay_opponent, defense_df, team_map)
            
        st.header("Current Parlay Slip")
        if not st.session_state.parlay_slip:
            st.info("Add a prop to begin.")
        else:
            total_prob = 1.0
            for i, leg in enumerate(st.session_state.parlay_slip):
                total_prob *= leg['prob']
                col1, col2, col3 = st.columns([5, 2, 1])
                with col1:
                    st.markdown(f"**{i+1}. {leg['player']}** | {leg['stat_desc']} **{leg['position']} {leg['line']}**")
                with col2:
                    st.markdown(f"**Prob: `{leg['prob']:.1%}`**")
                with col3:
                    st.button("Remove", key=f"remove_btn_{i}", on_click=remove_leg, args=(i,))
            st.markdown("---")
            implied_odds_decimal = (1 / total_prob) if total_prob > 0 else 0
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Combined Parlay Probability", f"{total_prob:.2%}")
            if implied_odds_decimal >= 2.0:
                american_odds = f"+{100 * (implied_odds_decimal - 1):.0f}"
            elif implied_odds_decimal > 1.0:
                 american_odds = f"{-100 / (implied_odds_decimal - 1):.0f}"
            else:
                american_odds = "N/A"
            p_col2.metric("Implied American Odds", american_odds)
            if st.button("Clear Parlay Slip"):
                st.session_state.parlay_slip = []
                st.rerun()
