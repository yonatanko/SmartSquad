import streamlit as st
import pandas as pd
from data_collection.fpl_api_collection import (get_league_table, get_current_gw, get_fixt_dfs, get_bootstrap_data, get_player_data, get_player_id_dict, get_name_to_dicts)
import json
import os
import warnings
import altair as alt
from streamlit_extras.row import row
import streamlit as st
import google.generativeai as genai

gemini_model = genai.GenerativeModel('gemini-pro')
genai.configure(api_key="AIzaSyCGmtRzgyzi7uiHkVFWkr2ccO37L9Ydmwc")

warnings.filterwarnings("ignore")

base_url = 'https://fantasy.premierleague.com/api/'

colors = {
    "Arsenal": "#EF0107",
    "Aston Villa": "#95BFE5",
    "Bournemouth": "#DA291C",
    "Brentford": "#FFDB00",
    "Brighton": "#0057B8",
    "Burnley": "#6C1D45",
    "Chelsea": "#034694",
    "Crystal Palace": "#1B458F",
    "Everton": "#003399",
    "Fulham": "#000000",
    "Leicester": "#003090",
    "Liverpool": "#C8102E",
    "Luton": "#FF5000",
    "Man City": "#6CABDD",
    "Man Utd": "#DA291C",
    "Newcastle": "#241F20",
    "Nott'm Forest": "#FFCC00",
    "Sheffield Utd": "#EE2737",
    "Spurs": "#102257",
    "West Ham": "#7A263A",
    "Wolves": "#FDB913",
}

if "selected_stats" not in st.session_state:
    st.session_state.selected_stats = ['Wins', 'Draws', 'Losses', 'Goals', 'Assists', 'Yellow Cards', 'Red Cards', 'Won At Home %', 'Won Away %']

league_df = get_league_table()

team_fdr_df, team_fixt_df, team_ga_df, team_gf_df = get_fixt_dfs()
team_fixt_df.to_csv("team_fixt_df.csv")

ct_gw = get_current_gw()

new_fixt_df = team_fixt_df.loc[:, ct_gw:(ct_gw+2)]
new_fixt_cols = ['GW' + str(col) for col in new_fixt_df.columns.tolist()]
new_fixt_df.columns = new_fixt_cols

new_fdr_df = team_fdr_df.loc[:, ct_gw:(ct_gw+2)]

league_df = league_df.join(new_fixt_df)

float_cols = league_df.select_dtypes(include='float64').columns.values

league_df = league_df.reset_index()
league_df.rename(columns={'team': 'Team'}, inplace=True)
league_df.index += 1

league_df['GD'] = league_df['GD'].map('{:+}'.format)

teams_df = pd.DataFrame(get_bootstrap_data()['teams'])

team1_stats = {
        "home":{
            "goals": [],
            "yellow_cards": [],
            "red_cards": [],
            "assists": [],
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "season": []
        }, 
        "away":{
            "goals": [],
            "yellow_cards": [],
            "red_cards": [],
            "assists": [],
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "season": []
        }
    }

team2_stats = {
    "home":{
        "goals": [],
        "yellow_cards": [],
        "red_cards": [],
        "assists": [],
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "season": []
    }, 
    "away":{
        "goals": [],
        "yellow_cards": [],
        "red_cards": [],
        "assists": [],
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "season": []
    }
}


def get_home_away_str_dict():
    new_fdr_df.columns = new_fixt_cols
    result_dict = {}
    for col in new_fdr_df.columns:
        values = list(new_fdr_df[col])
        max_length = new_fixt_df[col].str.len().max()
        if max_length > 7:
            new_fixt_df.loc[new_fixt_df[col].str.len() <= 7, col] = new_fixt_df[col].str.pad(width=max_length+9, side='both', fillchar=' ')
        strings = list(new_fixt_df[col])
        value_dict = {}
        for value, string in zip(values, strings):
            if value not in value_dict:
                value_dict[value] = []
            value_dict[value].append(string)
        result_dict[col] = value_dict
    
    merged_dict = {}
    for k, dict1 in result_dict.items():
        for key, value in dict1.items():
            if key in merged_dict:
                merged_dict[key].extend(value)
            else:
                merged_dict[key] = value
    for k, v in merged_dict.items():
        decoupled_list = list(set(v))
        merged_dict[k] = decoupled_list
    for i in range(1,6):
        if i not in merged_dict:
            merged_dict[i] = []
    return merged_dict


home_away_dict = get_home_away_str_dict()


def color_fixtures(val):
    bg_color = 'background-color: '
    font_color = 'color: '
    if val in home_away_dict[1]:
        bg_color += '#008000' # green
    elif val in home_away_dict[2]:
        bg_color += '#90EE90' # light green
    elif val in home_away_dict[3]:
        bg_color += '#808080' # grey
    elif val in home_away_dict[4]:
        bg_color += '#FF6347' # red
        font_color += 'white'
    elif val in home_away_dict[5]:
        bg_color += '#FF0000' # dark red
        font_color += 'white'
    else:
        bg_color += ''
    style = bg_color + '; ' + font_color
    return style

for col in new_fixt_cols:
    if league_df[col].dtype == 'O':
        max_length = league_df[col].str.len().max()
        if max_length > 7:
            league_df.loc[league_df[col].str.len() <= 7, col] = league_df[col].str.pad(width=max_length+9, side='both', fillchar=' ')

# league_df['GW7'] = ' ' * 10 + league_df['GW7'] + ' ' * 10
league_df.loc[league_df['Team'] == 'EVE', 'Team'] = 'EVE*'

def match_team_to_season_id(team_name, teams_df):
    for _,team in teams_df.iterrows():
        if team['name'] == team_name:
            return team['id']
        
def match_team_to_season_name(team_id, teams_df):
    for _,team in teams_df.iterrows():
        if team['id'] == team_id:
            return team['name']
        
def build_all_seasons_df(team_1_name, team_2_name):
    data_dir = os.path.join('Fantasy-Premier-Leaguue', 'data')
    seasons = os.listdir(data_dir)
    seasons = sorted(seasons, key=lambda x: int(x.split('-')[0]))[3:]
    all_seasons_df = pd.DataFrame()
    for season in seasons:
        data_path = os.path.join(data_dir, season, 'fixtures.csv')
        teams_path = os.path.join(data_dir, season, 'teams.csv') 
        df = pd.read_csv(data_path)
        teams_df = pd.read_csv(teams_path)
        team_1 = match_team_to_season_id(team_1_name, teams_df)
        team_2 = match_team_to_season_id(team_2_name, teams_df)
        
        team_matches = df[((df['team_h'] == team_1) & (df['team_a'] == team_2)) | ((df['team_h'] == team_2) & (df['team_a'] == team_1))]
        # filter out the matches that are not played yet
        team_matches = team_matches[team_matches['finished'] == True]
        # add a column with the season
        team_matches['season'] = season
        # add team names
        team_matches['team_h_name'] = team_matches['team_h'].apply(lambda x: match_team_to_season_name(x, teams_df))
        team_matches['team_a_name'] = team_matches['team_a'].apply(lambda x: match_team_to_season_name(x, teams_df))

        all_seasons_df = pd.concat([all_seasons_df, team_matches])
    return all_seasons_df

def extract_key_stats_updated(stats_str):
    stats = json.loads(stats_str.replace("'", "\""))
    extracted_stats = {
        'home_goals': 0,  
        'away_goals': 0,
        'home_yellow_cards': 0,
        'away_yellow_cards': 0,
        'home_red_cards': 0,
        'away_red_cards': 0,
        'home_assists': 0,
        'away_assists': 0
    }

    for stat in stats:
        if stat['identifier'] == 'goals_scored':
            extracted_stats['home_goals'] += sum(item['value'] for item in stat['h'])
            extracted_stats['away_goals'] += sum(item['value'] for item in stat['a'])
        elif stat['identifier'] == 'own_goals':
            extracted_stats['home_goals'] += sum(item['value'] for item in stat['a'])
            extracted_stats['away_goals'] += sum(item['value'] for item in stat['h'])
        elif stat['identifier'] == 'yellow_cards':
            extracted_stats['home_yellow_cards'] = sum(item['value'] for item in stat['h'])
            extracted_stats['away_yellow_cards'] = sum(item['value'] for item in stat['a'])
        elif stat['identifier'] == 'red_cards':
            extracted_stats['home_red_cards'] = sum(item['value'] for item in stat['h'])
            extracted_stats['away_red_cards'] = sum(item['value'] for item in stat['a'])
        elif stat['identifier'] == 'assists':
            extracted_stats['home_assists'] = sum(item['value'] for item in stat['h'])
            extracted_stats['away_assists'] = sum(item['value'] for item in stat['a'])

    return extracted_stats

def create_head_to_head_stats(team1, team2, df):
    selected_stats = st.session_state["selected_stats"]
    if team1 != team2 and team1 != None and team2 != None:
        if not df.empty:
            for _, row in df.iterrows():
                match_stats = extract_key_stats_updated(row['stats'])
                if row['team_h_name'] == team1:
                    team1_stats['home']['goals'].append(match_stats['home_goals'])
                    team1_stats['home']['yellow_cards'].append(match_stats['home_yellow_cards'])
                    team1_stats['home']['red_cards'].append(match_stats['home_red_cards'])
                    team1_stats['home']['season'].append(row['season'])
                    team1_stats['home']['assists'].append(match_stats['home_assists'])
                    team2_stats['away']['goals'].append(match_stats['away_goals'])
                    team2_stats['away']['yellow_cards'].append(match_stats['away_yellow_cards'])
                    team2_stats['away']['red_cards'].append(match_stats['away_red_cards'])
                    team2_stats['away']['season'].append(row['season'])
                    team2_stats['away']['assists'].append(match_stats['away_assists'])

                    if row['team_h_score'] > row['team_a_score']:
                        team1_stats['home']['wins'] += 1
                        team2_stats['away']['losses'] += 1
                    elif row['team_h_score'] < row['team_a_score']:
                        team1_stats['home']['losses'] += 1
                        team2_stats['away']['wins'] += 1
                    else:
                        team1_stats['home']['draws'] += 1
                        team2_stats['away']['draws'] += 1
                else:
                    team2_stats['home']['goals'].append(match_stats['home_goals'])
                    team2_stats['home']['yellow_cards'].append(match_stats['home_yellow_cards'])
                    team2_stats['home']['red_cards'].append(match_stats['home_red_cards'])
                    team2_stats['home']['season'].append(row['season'])
                    team2_stats['home']['assists'].append(match_stats['home_assists'])
                    team1_stats['away']['goals'].append(match_stats['away_goals'])
                    team1_stats['away']['yellow_cards'].append(match_stats['away_yellow_cards'])
                    team1_stats['away']['red_cards'].append(match_stats['away_red_cards'])
                    team1_stats['away']['season'].append(row['season'])
                    team1_stats['away']['assists'].append(match_stats['away_assists'])

                    if row['team_h_score'] > row['team_a_score']:
                        team2_stats['home']['wins'] += 1
                        team1_stats['away']['losses'] += 1
                    elif row['team_h_score'] < row['team_a_score']:
                        team2_stats['home']['losses'] += 1
                        team1_stats['away']['wins'] += 1
                    else:
                        team2_stats['home']['draws'] += 1
                        team1_stats['away']['draws'] += 1

            # create one df with 3 cols: team1, stat, team2.
            comparsion_df = pd.DataFrame(columns=['Team 1', 'Stat', 'Team 2'])
            for stat in ['wins','draws', 'losses', 'goals','assists', 'yellow_cards', 'red_cards']:
                if stat.replace('_', ' ').title() in selected_stats:
                    if stat in ['wins', 'draws', 'losses']:
                        team_1_total = team1_stats['home'][stat] + team1_stats['away'][stat]
                        team_2_total = team2_stats['home'][stat] + team2_stats['away'][stat]
                    else:
                        team_1_total = sum(team1_stats['home'][stat]) + sum(team1_stats['away'][stat])
                        team_2_total = sum(team2_stats['home'][stat]) + sum(team2_stats['away'][stat])

                    comparsion_df = comparsion_df.append(pd.DataFrame({'Team 1': [team_1_total], 'Stat': [stat], 'Team 2': [team_2_total]}))

            # adding manual stats - won_at_home % and won_away %
            if team1_stats['home']['wins'] + team1_stats['home']['draws'] + team1_stats['home']['losses'] > 0 and team2_stats['home']['wins'] + team2_stats['home']['draws'] + team2_stats['home']['losses'] > 0:
                team_1_won_at_home = (team1_stats['home']['wins'] / (team1_stats['home']['wins'] + team1_stats['home']['draws'] + team1_stats['home']['losses'])) * 100
                team_2_won_at_home = (team2_stats['home']['wins'] / (team2_stats['home']['wins'] + team2_stats['home']['draws'] + team2_stats['home']['losses'])) * 100
                team_1_won_away = (team1_stats['away']['wins'] / (team1_stats['away']['wins'] + team1_stats['away']['draws'] + team1_stats['away']['losses'])) * 100
                team_2_won_away = (team2_stats['away']['wins'] / (team2_stats['away']['wins'] + team2_stats['away']['draws'] + team2_stats['away']['losses'])) * 100
            else:
                team_1_won_at_home = 0
                team_2_won_at_home = 0
                team_1_won_away = 0
                team_2_won_away = 0

            if 'Won At Home %' in selected_stats:
                comparsion_df = comparsion_df.append(pd.DataFrame({'Team 1': [int(team_1_won_at_home)], 'Stat': ['won_at_home %'], 'Team 2': [int(team_2_won_at_home)]}))
            if 'Won Away %' in selected_stats:
                comparsion_df = comparsion_df.append(pd.DataFrame({'Team 1': [int(team_1_won_away)], 'Stat': ['won_away %'], 'Team 2': [int(team_2_won_away)]}))
            
            # make df tighter
            comparsion_df = comparsion_df.reset_index(drop=True)
            comparsion_df = comparsion_df.T.reset_index(drop=True).T
            comparsion_df.columns = [team1, 'Stat', team2]
            comparsion_df['Stat'] = comparsion_df['Stat'].apply(lambda x: x.replace('_', ' ').title()) 
            
            
            with inner_col2:
                # s1 should be the header, with 3 colors: one to each column
                s1 = [
                    dict(selector='th.col_heading', props=[('text-align', 'center')]),
                    dict(selector=f'th.col_heading.level0.col0', props=[('background-color', colors[team1])]),
                    dict(selector=f'th.col_heading.level0.col1', props=[('background-color', "")]),
                    dict(selector=f'th.col_heading.level0.col2', props=[('background-color', colors[team2])])
                ]
                s2 = dict(selector='td', props=[('text-align', 'center')])
                
                # Initialize the Styler object for the dataframe
                styler = comparsion_df.style.set_table_styles([s1[0], s1[1], s1[2], s1[3], s2]).hide_index().hide(axis=0)

                # Apply color styles to the team columns
                styler = styler.applymap(lambda x: f'background-color: {colors[team1]}', subset=[team1])
                styler = styler.applymap(lambda x: f'background-color: {colors[team2]}', subset=[team2])

                # Convert to HTML
                table_html = styler.to_html()

                # Annotate and display the text and table
                st.write("##### 2019-20 to 2023-24 Head-to-Head")
                st.write(table_html, unsafe_allow_html=True)
                
        else:
            st.write("No head-to-head matches found between the selected teams.")
    else:
        if team1 == None or team2 == None:
            st.warning("You must pick two different teams.")
        else:
            st.warning("Please select two different teams.")

# Premier League Table - title in the center of the page
st.title('Premier League Table :trophy:')
st.write('Current Gameweek: ', str(ct_gw))

styled_df = league_df.style.applymap(color_fixtures, subset=new_fixt_cols) \
                            .format(subset=float_cols, formatter='{:.2f}')
                            
st.dataframe(styled_df, height=210, use_container_width=True)
st.text('*Everton received a 10 Point deduction on 17/11/2023 for breaching Financial Fair Play rules.')

st.markdown("---")

col1, col2 = st.columns([1.4,1], gap="Large")

with col1:
    # Head-to-Head Stats
    st.title('Teams Head-to-Head :vs:')
    empty_row = row(1)
    inner_col1, inner_col2 = st.columns([1.8, 2])

    # Function to get the team's name mapping (you would replace this with actual team names if available)
    team_names = teams_df["name"].to_list()

    with inner_col1:
        team1 = st.selectbox('Choose Team 1:', options=sorted(team_names), placeholder='Select Team 1', index=None, on_change=None, key="stat_team1")
        team2 = st.selectbox('Choose Team 2:', options=sorted(team_names), placeholder='Select Team 2', index=None, on_change=None, key="stat_team2")
        if team1 != team2 and team1 != None and team2 != None:
            df = build_all_seasons_df(team1, team2)
            # Here we define the stats options for the multiselect widget.
            stats_options = ['Wins', 'Draws', 'Losses', 'Goals', 'Assists', 'Yellow Cards', 'Red Cards', 'Won At Home %', 'Won Away %']
            st.session_state["selected_stats"] = st.multiselect('Select stats to show:', stats_options, default=stats_options, on_change=None)
            create_head_to_head_stats(team1, team2, df)

    def create_stats_bar(player_id, player_team, stat_to_show):
        stat_to_show = stat_to_show.lower().replace(" ", "_")
        data_df = pd.DataFrame()
        player_fixtures = get_player_data(player_id)["history"]
        last_10_matches = player_fixtures[-10:]
        # create bar chart with number of goals
        for match in last_10_matches:
            match["opponent_team"] = match_team_to_season_name(match["opponent_team"], teams_df)
            data_df = data_df.append(match, ignore_index=True)

        data_df = data_df.sort_values(by="kickoff_time", ascending=True).reset_index(drop=True)
        # make all columns numeric
        data_df[stat_to_show] = pd.to_numeric(data_df[stat_to_show], errors='coerce')
        # show bar char by the df order
        st.write(
            alt.Chart(data_df).mark_bar().encode(
                x=alt.X('opponent_team', sort=None, title='Opponent Team (Earliest (left) to Latest (right))'),
                y=alt.Y(stat_to_show),  # Define domain to prevent inversion and remove gaps
                color = alt.value(colors[player_team])
            ).properties(
                title=f'{stat_to_show.replace("_", " ").title()} in last 10 matches',
                width=450,
                height=450,
            ).configure_title(
                anchor='middle'
            )
        )


with col2:
    st.title('Player Form :zap:')
    row1 = row(0, gap="small")
    players_id_dict, players_teams_dict = get_name_to_dicts()
    players_stats = list(get_player_data(308)['history'][0].keys())
    players_stats = [x.replace("_", " ").title() for x in players_stats if x not in ["element", "fixture", "opponent_team", "kickoff_time","was_home", "team_h_score","team_a_score", "round"]]
    row1.col1, row1.col2 = st.columns([1, 1])
    with row1.col1:
        player_name = st.selectbox('Select Player:', options=sorted(players_id_dict.keys()), placeholder='Select Player', index=None, on_change=None, key="player_name")
    with row1.col2:
        stat_to_show = st.selectbox('Select Stat:', options=players_stats, placeholder='Select Stat', index=None, on_change=None, key="stat_to_show")
    if player_name and stat_to_show:
        player_id = players_id_dict[player_name]
        player_team = players_teams_dict[player_name]
        create_stats_bar(player_id, player_team, stat_to_show)

st.markdown('---')

col3, col4 = st.columns(2)
with col3:
    # use gemini model to show a picked by user team's form
    st.title('Team Form :chart_with_upwards_trend:')
    st.write('Pick a team you want to know its form in the last 5 matches.')
    team_name = st.selectbox('Select Team:', options=sorted(team_names), placeholder='Select Team', index=None, on_change=None, key="team_name")

with col4:
    if team_name:
        all_fixtures = pd.read_csv('Fantasy-Premier-Leaguue/data/2023-24/fixtures.csv')
        team_id = match_team_to_season_id(team_name, teams_df)
        team_fixtures = all_fixtures[(all_fixtures['team_h'] == team_id) | (all_fixtures['team_a'] == team_id)]
        # filter out fixtures that are not played yet
        team_fixtures = team_fixtures[team_fixtures['finished'] == True][['team_h', 'team_a', 'team_h_score', 'team_a_score']]
        # convert to team names
        team_fixtures['team_h'] = team_fixtures['team_h'].apply(lambda x: match_team_to_season_name(x, teams_df))
        team_fixtures['team_a'] = team_fixtures['team_a'].apply(lambda x: match_team_to_season_name(x, teams_df))
        # take last 5 fixtures
        team_fixtures = team_fixtures.tail(5) # last 5 fixturessa
        # transform to dict of: list of dicts: [{team_h: team_h_score, team_a: team_a_score}]
        # team_fixtures_str = team_fixtures.to_string(index=False)
        # transform to df with the following columns: opponent_team, was_home, won, lost, draw, goals
        new_fixtures = team_fixtures.copy()
        new_fixtures['won'] = new_fixtures.apply(lambda x: 1 if x['team_h'] == team_name and x['team_h_score'] > x['team_a_score'] or x['team_a'] == team_name and x['team_a_score'] > x['team_h_score'] else 0, axis=1)
        new_fixtures['lost'] = new_fixtures.apply(lambda x: 1 if x['team_h'] == team_name and x['team_h_score'] < x['team_a_score'] or x['team_a'] == team_name and x['team_a_score'] < x['team_h_score'] else 0, axis=1)
        new_fixtures['draw'] = new_fixtures.apply(lambda x: 1 if x['team_h_score'] == x['team_a_score'] else 0, axis=1)
        new_fixtures['goals scored'] = new_fixtures.apply(lambda x: x['team_h_score'] if x['team_h'] == team_name else x['team_a_score'], axis=1)
        new_fixtures['goals conceded'] = new_fixtures.apply(lambda x: x['team_a_score'] if x['team_h'] == team_name else x['team_h_score'], axis=1)
        new_fixtures['opponent_team'] = new_fixtures.apply(lambda x: x['team_h'] if x['team_h'] != team_name else x['team_a'], axis=1)
        new_fixtures = new_fixtures[['opponent_team', 'won', 'lost', 'draw', 'goals scored', 'goals conceded']]
        team_fixtures_str = new_fixtures.to_string(index=False)
        text_prompt = f"analyze the form of {team_name} in the last 5 matches based on this data only: {team_fixtures_str}. summarize it in a passage without headline"
        response = gemini_model.generate_content(text_prompt)
        text = response.text.replace('•', '*')
        st.title(" ")
        st.markdown(f":robot_face:: {text}")

        
