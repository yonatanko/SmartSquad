import streamlit as st
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import pandas as pd
from data_collection.fpl_api_collection import get_bootstrap_data, get_name_and_pos_and_team_dict, get_fixture_dfs
import google.generativeai as genai
import os
import json
import numpy as np

###############################################################################################
################################## Functions ##################################################
###############################################################################################

def extract_key_stats_updated(stats_str):
    """
    Extracts the key stats from the stats string
    Args:
        stats_str (String): stats string from the fixtures df

    Returns:
        Dict: extracted stats in a dictionary
    """
    stats = json.loads(stats_str.replace("'", "\""))
    
    # initialize the extracted stats
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

# Matching functions: name -> id and id -> name
def match_team_to_season_id(team_name, teams_df):
    for _,team in teams_df.iterrows():
        if team['name'] == team_name:
            return team['id']
        
def match_team_to_season_name(team_id, teams_df):
    for _,team in teams_df.iterrows():
        if team['id'] == team_id:
            return team['name']

def build_all_seasons_df(team_1_name, team_2_name):
    """
    Build a df with all the matches between team_1 and team_2 from 2019/20 season to current season
    Args:
        team_1_name (String): team 1 name
        team_2_name (String): team 2 name

    Returns:
        _type_: Pandas DataFrame with all the matches between team_1 and team_2 from 2019/20 season to current season
    """
    data_dir = os.path.join('Fantasy-Premier-Leaguue', 'data')
    seasons = os.listdir(data_dir)
    # drop "cleaned_merged_seasons.csv" from the list
    seasons = [season for season in seasons if season != "cleaned_merged_seasons.csv"]
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

def color_fixtures(val):
    """
    Color the fixtures based on the difficulty
    Args:
        val (int): Difficulty value from 1 to 5

    Returns:
        String: Color code in hex format
    """
    if val == 5:
        # strong red
        return "#FF0000"
    elif val == 4:
        # light red
        return "#FF6347"
    elif val == 3:
        # grey
        return "#808080"
    elif val == 2:
        # light green
        return "#90EE90"
    elif val == 1:
        # strong green
        return "#008000"
    
def color_to_name(color):
    """
    Convert the color to the difficulty name
    Args:
        color (String): Color code in hex format

    Returns:
        (String, String): Tuple with the color name and the difficulty description
    """
    if color == "#FF0000":
        return ("strong red", "Very Difficult")
    elif color == "#FF6347":
        return ("light red", "Difficult")
    elif color == "#808080":
        return ("grey", "Neutral")
    elif color == "#90EE90":
        return ("light green", "Easy")
    else:
        return ("strong green", "Very Easy")

def generate_position_mapping(num_defenders, num_midfielders, num_forwards, pitch_height=80):
    """
    Generate the mapping of positions to x, y coordinates on the pitch based on the number of players in each position

    Args:
        num_defenders (int): Number of defenders in the team 
        num_midfielders (int): Number of midfielders in the team
        num_forwards (int): Number of forwards in the team
        pitch_height (int, optional): height of the pitch. Defaults to 80.

    Returns:
        Dict: Dictionary with position as key and list of x, y coordinates as value
    """
    position_mapping = {
        "GK": (10, pitch_height // 2)
    }  # Goalkeeper's position is fixed at the center of the goal line

    # Function to generate y positions starting from the center
    def generate_y_positions(num_players):
        center = pitch_height / 2
        positions = []
        if num_players % 2 == 0:  # Even number of players
            step = pitch_height / (num_players + 1) + 5
            for i in range(num_players // 2):
                positions.extend([center - (i + 0.5) * step, center + (i + 0.5) * step])
        else:  # Odd number of players
            step = pitch_height / num_players
            positions.append(center)
            for i in range(1, num_players // 2 + 1):
                positions.extend([center - i * step, center + i * step])

        return sorted(positions)
    
    position_mapping["SUB"] = [(-10, y) for y in generate_y_positions(4)]
    position_mapping["DEF"] = [(34, y) for y in generate_y_positions(num_defenders)]
    position_mapping["MID"] = [(60, y) for y in generate_y_positions(num_midfielders)]
    position_mapping["FWD"] = [(92, y) for y in generate_y_positions(num_forwards)]

    return position_mapping

def draw_pitch_with_players(starting_11, subs, colors, selected_stats, player_stats):
    """
    Draw the pitch with the players on it.
    Args:
        starting_11 (List): The starting 11 players with their positions and clubs as a List of tuples: (position, player, club)
        subs (List): The Remaining 4 players (substitutes) with their positions and clubs as a List of tuples: (position, player, club)
        colors (Dict): Dictionary with club names as keys and color codes as values
        selected_stats (List): List of The stats that were picked by the user to display below the player's name
        player_stats (Dict): Dict of dicts with player names as keys and stats as values

    Returns:
        matplotlib.figure.Figure: The figure with the pitch and players
    """
    fig, ax = plt.subplots(figsize=(7, 14))
    pitch = VerticalPitch(pitch_color="grass", line_color="white", stripe=True, pad_bottom=25)
    pitch.draw(ax=ax) # Built-in function to draw the pitch

    used_positions = {"DEF": 0, "MID": 0, "FWD": 0, "SUB": 0} # Counter to keep track of the number of players in each position

    for position, player, club in starting_11+subs: # Iterate over each player and plot them on the pitch
        if position == "GK":
            x, y = position_mapping[position]
        else:
            x, y = position_mapping[position][used_positions[position]]
            used_positions[position] += 1

        color = colors.get(club, "grey") # Get the color of the players' club from the colors dictionary
        pitch.scatter(x, y, s=600, ax=ax, edgecolors="black", c=color, zorder=2) # Plot the player on the pitch
        plt.text(y, x + 6, player, fontsize=15, ha="center", va="center", fontdict={"fontweight": "bold"}) # Add the player's name Above the player

        # Format the stats to display below the player's name
        stat_values = [
            f'{player_stats[player].get(stat, "N/A")}' for stat in selected_stats
        ]

        formatted_stat_values = " | ".join(stat_values[:2])
        if len(stat_values) == 0:
            formatted_stat_values = ""

        # Get the color of the fixture difficulty for the next game from the team_fdr_df
        fixture_difficulty_color = color_fixtures(
            team_fdr_df.iloc[match_team_name_to_id(club)-1, st.session_state.selected_gameweek-1]
        )

        # if stats has next game, color the box with the fixture difficulty color else white
        if "Next Game" in selected_stats:
            bbox_props = dict(boxstyle="round,pad=0.6", fc=fixture_difficulty_color, ec="black", lw=1, alpha=0.5)
        else:
            bbox_props = dict(boxstyle="round,pad=0.6", fc="white", ec="black", lw=1, alpha=0.5)
        
        # Add the stats below the player's circle
        ax.text(
            y,
            x - 8,
            formatted_stat_values,
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            bbox=bbox_props,
        )

    return fig

def match_team_name_to_id(team_name): # Function to match the team name to the team id in the teams_df
    return teams_df[teams_df['name'] == team_name]['id'].values[0]

def hide_container():
    """Hide the initial container with The question in order to show the recommendation container"""
    st.session_state.show_container = False


def show_recommendation():
    """Show the recommendation container with the recommended transfers"""
    st.session_state.show_container = False
    st.session_state.show_transfer_recommendation = True
    with col3:
        create_recommendation()

def create_recommendation(num_gameweeks):
    """
    Create the transfer recommendation based on the expected points for the next gameweeks, where the user picks the number of gameweeks to consider
    Args:
        num_gameweeks (int): Number of gameweeks to consider for the transfer recommendation chosen by the user
    """
    if num_gameweeks: # If the user has selected the number of gameweeks
        # create a df with the expected points for the next gameweeks for the not selected players
        not_selected_points_df = not_selected_players_points_df.iloc[:, st.session_state.selected_gameweek-1:st.session_state.selected_gameweek+num_gameweeks-1]
        not_selected_points_df["Total"] = not_selected_points_df.sum(axis=1)
        # create a df with the expected points for the next gameweeks for the selected players
        selected_points_df = players_points_df.iloc[:, st.session_state.selected_gameweek-1:st.session_state.selected_gameweek+num_gameweeks-1]
        selected_points_df["Total"] = selected_points_df.sum(axis=1)
        potential_swaps = []
        # for each position, select the player with the least expected avg points
        for position in ["GK", "DEF", "MID", "FWD"]:
            # filter df
            selected_players_in_pos = selected_points_df[selected_points_df.index.isin([player[1] for player in players if player[0] == position])]
            # get the player with the least expected avg points in the selected players
            player_name = selected_players_in_pos["Total"].idxmin()
            player_avg = selected_players_in_pos["Total"].min()
            # get the player with the highest expected avg points in the not selected players
            not_selected_players_in_pos = not_selected_points_df[not_selected_points_df.index.isin([player[1] for player in not_selected_players if player[0] == position])]
            player_name_not_selected = not_selected_players_in_pos["Total"].idxmax()
            player_avg_not_selected = not_selected_players_in_pos["Total"].max()
            if player_avg_not_selected > player_avg:
                potential_swaps.append((player_name, player_name_not_selected, player_avg_not_selected - player_avg))

        if potential_swaps: # If there are potential swaps
            first_col, second_col, third_col, fourth_col = st.columns([1.5, 1.5, 0.0001, 1.5])
            # choose the swap that will give the best additional points
            best_swap = max(potential_swaps, key=lambda x: x[2])
            selected_player = best_swap[0]
            selected_player_next_games_and_points = selected_points_df.loc[selected_player]
            selected_player_next_games_and_points = pd.DataFrame(selected_player_next_games_and_points).rename(columns={selected_player: "Points"})
            not_selected_player = best_swap[1]
            not_selected_player_next_games_and_points = not_selected_points_df.loc[not_selected_player]
            not_selected_player_next_games_and_points = pd.DataFrame(not_selected_player_next_games_and_points).rename(columns={not_selected_player: "Points"})
            # change gameweek to the opponnet team name
            selected_player_team = name_to_team_dict[selected_player]
            not_selected_player_team = name_to_team_dict[not_selected_player]
            selected_p_team_id = match_team_name_to_id(selected_player_team)
            not_selected_p_team_id = match_team_name_to_id(not_selected_player_team)
            selected_p_next_games = team_fixt_df.iloc[selected_p_team_id-1, st.session_state.selected_gameweek-1:st.session_state.selected_gameweek+num_gameweeks-1]
            not_selected_p_next_games = team_fixt_df.iloc[not_selected_p_team_id-1, st.session_state.selected_gameweek-1:st.session_state.selected_gameweek+num_gameweeks-1]
            # add the team names as a column
            selected_player_next_games_and_points.insert(0, "Opponent", selected_p_next_games.to_list() + [" "])
            not_selected_player_next_games_and_points.insert(0, "Opponent", not_selected_p_next_games.to_list() + [" "])
            
            with first_col: # Display the picked player from the current squad

                # center aligned subheader
                st.markdown(f'## {selected_player} - {selected_player_team}')
                # display the table with the Opponent team rows colored by the fixture difficulty
                clubs = selected_player_next_games_and_points["Opponent"].to_list()[:-1]
                gws = selected_player_next_games_and_points.index.to_list()[:-1]
                gws = [int(gw[2:]) for gw in gws]
                # create color map for the clubs
                colors = [color_fixtures(team_fdr_df.iloc[match_team_name_to_id(selected_player_team)-1, gw-1]) for gw in gws]
                mapping = dict(zip(clubs, colors))
                # add the transparent color for the last row
                mapping[" "] = "rgba(0,0,0,0)"
                # round the points to 1 decimal
                selected_player_next_games_and_points["Points"] = selected_player_next_games_and_points["Points"].apply(lambda x: int(x))
                st.write(selected_player_next_games_and_points.style.applymap(lambda x: f"background-color: {mapping[x]}", subset=["Opponent"]))

            with second_col: # Display the header "Substitute with:"
                for i in range(8):
                    st.write("")
                st.markdown("#### Subsitute with:")

            with fourth_col: # Display the potential swap from the not selected players

                st.markdown(f'## {not_selected_player} - {not_selected_player_team}')
                # display the table with the Opponent team rows colored by the fixture difficulty
                clubs = not_selected_player_next_games_and_points["Opponent"].to_list()[:-1]
                gws = not_selected_player_next_games_and_points.index.to_list()[:-1]
                gws = [int(gw[2:]) for gw in gws]
                # create color map for the clubs
                colors = [color_fixtures(team_fdr_df.iloc[match_team_name_to_id(not_selected_player_team)-1, gw-1]) for gw in gws]
                mapping = dict(zip(clubs, colors))
                # add the transparent color for the last row
                mapping[" "] = "rgba(0,0,0,0)"
                # round the points to 1 decimal
                not_selected_player_next_games_and_points["Points"] = not_selected_player_next_games_and_points["Points"].apply(lambda x: int(x))
                st.write(not_selected_player_next_games_and_points.style.applymap(lambda x: f"background-color: {mapping[x]}", subset=["Opponent"]))

        else: # If there are no potential swaps
            st.write("No potential swaps found")

def create_head_to_head_stats_for_inference(team1, team2, df):
    """
        Args:
        team1 (string): first team name
        team2 (string): second team name
        df (pandas.DataFrame): DataFrame with all the matches between team_1 and team_2 from 2019/20 season to current season

    Returns:
        pandas.DataFrame: Pandas DataFrame with the stats for the head to head matches between team1 and team2
    """

    # initialize the Dicts to store the stats for each team
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
    
    if team1 != team2 and team1 != None and team2 != None: # If the teams are different and not None
        if not df.empty:
            for _, row in df.iterrows():
                match_stats = extract_key_stats_updated(row['stats']) # Extract the key stats for the match
                if row['team_h_name'] == team1: # If team1 is the home team
                    # Add the stats to the team1_stats and team2_stats dictionaries
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

                    # Check the result of the match and update the wins, draws and losses
                    if row['team_h_score'] > row['team_a_score']:
                        team1_stats['home']['wins'] += 1
                        team2_stats['away']['losses'] += 1
                    elif row['team_h_score'] < row['team_a_score']:
                        team1_stats['home']['losses'] += 1
                        team2_stats['away']['wins'] += 1
                    else:
                        team1_stats['home']['draws'] += 1
                        team2_stats['away']['draws'] += 1

                else: # If team1 is the away team
                    # Add the stats to the team1_stats and team2_stats dictionaries
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

                    # Check the result of the match and update the wins, draws and losses
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

            # add the manual stats to the df
            comparsion_df = comparsion_df.append(pd.DataFrame({'Team 1': [int(team_1_won_at_home)], 'Stat': ['won_at_home %'], 'Team 2': [int(team_2_won_at_home)]}))
            comparsion_df = comparsion_df.append(pd.DataFrame({'Team 1': [int(team_1_won_away)], 'Stat': ['won_away %'], 'Team 2': [int(team_2_won_away)]}))
            
            # make df tighter
            comparsion_df = comparsion_df.reset_index(drop=True)
            comparsion_df = comparsion_df.T.reset_index(drop=True).T
            comparsion_df.columns = [team1, 'Stat', team2]

            return comparsion_df

###############################################################################################
################################## Main #######################################################
###############################################################################################

# Initialize the GenerativeAI model
gemini_model = genai.GenerativeModel('gemini-pro')
genai.configure(api_key="AIzaSyCGmtRzgyzi7uiHkVFWkr2ccO37L9Ydmwc")

# Define the CSS for the margins
margins_css = """
    <style>
        .main > div {
            padding-top: 1.5rem;
            padding-left: 5rem;
        }

        button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
"""

st.markdown(margins_css, unsafe_allow_html=True)

st.write(":point_left: You can open the sidebar to navigate to other pages and to select the statistics to display on the pitch")

# Define the colors for the teams
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

# Definiton of the session state variables
if "show_container" not in st.session_state:
    st.session_state.show_container = True

if "selected_gameweek" not in st.session_state:
    st.session_state.selected_gameweek = 1

if "show_transfer_recommendation" not in st.session_state:
    st.session_state.show_transfer_recommendation = False

# Check if the user has selected 15 players
# Happens when the user directly navigates to this page before selecting the players
if len(st.session_state["players"]) != 15:
    st.warning("Please select 15 players in the Welcome page", icon="⚠️")
    st.stop()

# Modify the sidebar section to allow the selection of a statistic
st.sidebar.markdown("### Select a statistic to display:")
selected_stats = [
    layer
    for layer_name, layer in {
        "Expected score": "Expected score",
        "Next Game": "Next Game"
    }.items()
    if st.sidebar.checkbox(layer_name, False)
]
st.sidebar.markdown("The stats will be displayed below the player's name on the pitch as shown below:")
st.sidebar.markdown("Expected score | Next Game")

# extract number of players in each position from the st.session_state['players'] list
num_defenders = len([player for player in st.session_state["players"] if player.split(",")[0] == "DEF"]) -1
num_midfielders = len([player for player in st.session_state["players"] if player.split(",")[0] == "MID"]) -1
num_forwards = len([player for player in st.session_state["players"] if player.split(",")[0] == "FWD"]) -1

# Generate the position mapping based on the number of players in each position
position_mapping = generate_position_mapping(
    num_defenders, num_midfielders, num_forwards
)

teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
team_names = teams_df["name"].to_list()

# extract player, position and club from the st.session_state['players'] list
players = [player.split(",") for player in st.session_state["players"]]
num_to_string_pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
name_to_pos_dict, name_to_team_dict = get_name_and_pos_and_team_dict()

# load the players_points_df and not_selected_players_points_df
players_points_df = pd.read_csv("players_points.csv", index_col=0)
not_selected_players_points_df = pd.read_csv("not_selected_players_points.csv", index_col=0)

# Extract the players that were not selected by the user
all_players = [(num_to_string_pos[name_to_pos_dict[player]], player, name_to_team_dict[player]) for player in name_to_pos_dict.keys()]
not_selected_players = [player for player in all_players if player[1] not in players_points_df.index]

# calculate the best starting 11 based on players_points_df in a given gameweek and put the rest on the bench
subs_names = []
gameweek_df = players_points_df["GW " + str(st.session_state.selected_gameweek)]
gameweek_df = gameweek_df.sort_values(ascending=False)

# for each position, select the player with the least expected points and put him on the bench
for position in ["GK", "DEF", "MID", "FWD"]:
    # filter df
    position_df = gameweek_df[gameweek_df.index.isin([player[1] for player in players if player[0] == position])]
    # get the player with the least expected points
    player_name = position_df.idxmin()
    subs_names.append(player_name)

# split the players into starting 11 and subs
starting_11 = [player for player in players if player[1] not in subs_names]
subs = [player for player in players if player[1] in subs_names]
# adjust the position of the subs to be "SUB"
subs = [(f"SUB", player[1], player[2]) for player in subs]

# get the team ids of the selected players
start_11_teams_ids = [match_team_name_to_id(player[2]) for player in starting_11]
subs_teams_ids = [match_team_name_to_id(player[2]) for player in subs]

# Load the fixtures data and the fixture difficulty data*
_, team_fixt_df = get_fixture_dfs()
team_fdr_df = pd.read_csv("difficulties_df.csv", index_col=0)

# Create a dictionary to store the stats for each player in the starting 11
player_stats = {
    player[1]: {
        "Expected score": players_points_df.loc[player[1], "GW " + str(st.session_state.selected_gameweek)],
        "Next Game": team_fixt_df.iloc[start_11_teams_ids[starting_11.index(player)]-1, st.session_state.selected_gameweek-1 if st.session_state.selected_gameweek < 37 else 37],
    }
    for player in starting_11
}

# Create a dictionary to store the stats for each player in the subs
subs_stats = {
    player[1]: {
        "Expected score": players_points_df.loc[player[1], "GW " + str(st.session_state.selected_gameweek)],
        "Next Game": team_fixt_df.iloc[subs_teams_ids[subs.index(player)]-1, st.session_state.selected_gameweek-1 if st.session_state.selected_gameweek < 37 else 37],
    }
    for player in subs
}

# add the subs stats to the player_stats
player_stats.update(subs_stats)

# Define the columns for the layout
col1, col2, col3, col4 = st.columns([2, 0.3, 3, 0.05])

with col4: # Back button to reset the session state and go back to the initial main page
    if st.button(":back:"):
        st.session_state.show_container = True
        st.session_state.show_transfer_recommendation = False
        st.rerun()

# Draw the pitch in the first column, which will span across all rows on the left side.
with col1:
    # Gameweek selection
    inner_col1, inner_col2, inner_col3, inner_col4 = st.columns(4)
    with inner_col1:
        st.write("Choose GameWeek:") # Display the text "Choose GameWeek:" so the user can select the gameweek
    if inner_col2.button(":arrow_backward:", key="back_button"): # Button to go to the previous gameweek
        if st.session_state.selected_gameweek > 1:
            st.session_state.selected_gameweek -= 1
            st.session_state.show_container = True
            st.session_state.show_transfer_recommendation = False
            st.rerun()
        else:
            st.warning("You are already at the first gameweek", icon="⚠️")

    with inner_col3: # Display the selected gameweek in a rounded text area with white background
        st.markdown(
            f"""
            <div style="margin-bottom: 20px; margin-top: 8px; margin-right: 40px; border-radius: 15px; text-align: center; background-color: white; color: black;">
                <p style= "font-size": 20px >{"GW" + str(st.session_state.selected_gameweek)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if inner_col4.button(":arrow_forward:"): # Button to go to the next gameweek
        if st.session_state.selected_gameweek < 38:
            st.session_state.selected_gameweek += 1
            st.session_state.show_container = True
            st.session_state.show_transfer_recommendation = False
            st.rerun()
        else:
            st.warning("You are already at the last gameweek", icon="⚠️")

    st.markdown(f"#### :robot_face: GW{st.session_state.selected_gameweek} Recommended Starting 11") # Pitch title
    fig = draw_pitch_with_players(starting_11, subs, colors, selected_stats, player_stats) # Build the pitch with the players
    st.pyplot(fig) # Display the pitch

with col3: # The column that will have: The transfer recommendation container, the color legend, the H and A explanation and the difficulty of fixtures explanation with the gemini model
    if st.session_state.show_container: 
        with st.container(): # The container that will have the initial transfer recommendation text
            st.markdown(
                """
                <div style="box-shadow: 0px 0px 20px #ccc; padding: 20px; margin-bottom: 20px; margin-right: 130px; border-radius: 15px; text-align: center;">
                    Want a recommendation for a good transfer?
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Define the user interaction buttons - Yes (to show the transfer recommendation) and No (to hide the container)
            inside_col1, inside_col2 = st.columns(2)

            with inside_col1:
                if st.button("Yes 👍", key="yes"):
                    st.session_state.show_container = False
                    st.session_state.show_transfer_recommendation = True
                    st.rerun()

            with inside_col2:
                if st.button("No 👎", on_click=hide_container, key="No"):
                    # Perform action for No: make the content disappear\
                    pass
    
    # 1) Transfer recommendation
    if st.session_state.show_transfer_recommendation: # If the user has clicked on Yes to show the transfer recommendation
        num_gameweeks = st.selectbox("Select the number of gameweeks to consider", range(1, 38 - st.session_state.selected_gameweek + 1), index=None) # user selects the number of gameweeks to consider
        create_recommendation(num_gameweeks) # Create the transfer recommendation based on the selected number of gameweeks by the user

    # 2) Fixture Difficulty Rating Explanation
    if "Next Game" in selected_stats or st.session_state.show_transfer_recommendation: # If the user has selected the "Next Game" stat or has clicked on Yes to show the transfer recommendation (In both cases, the difficulty of fixtures will be present)
        # Build the color legend for the fixture difficulty. Each markdown is a colored box with the difficulty rating name
        st.markdown("---")
        st.markdown(f"### Fixture Difficulty Rating Color Legend")

        st.markdown(
            """
            <div style="margin-right: 100px; text-align: center; background-color: #FF0000; color: black;">
                <p style= "font-size": 20px >Very Difficult</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-right: 100px; text-align: center; background-color: #FF6347; color: black;">
                <p style= "font-size": 20px >Slightly difficult</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-right: 100px; text-align: center; background-color: #808080; color: black;">
                <p style= "font-size": 20px >Medium</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-right: 100px; text-align: center; background-color: #90EE90; color: black;">
                <p style= "font-size": 20px >Easy</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-right: 100px; text-align: center; background-color: #008000; color: black;">
                <p style= "font-size": 20px >Very Easy</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # 3) H (Home) and A (Away) Explanation
        st.markdown(f"##### H: The Game is played at Home")
        st.markdown(f"##### A: The Game is played Away (At the Opponent's Stadium)")
        st.markdown("---")

        # 4) Difficulty of Fixtures Explanation with the Gemini Model
        st.markdown(f'## Choose a matchup to get explanation of the color chosen')
        # Create a multiselect widget to select the teams for the matchup
        team1 = st.selectbox('Choose Team 1:', options=sorted(team_names), placeholder='Select Team 1', index=None, on_change=None, key="team1")
        team2 = st.selectbox('Choose Team 2:', options=sorted(team_names), placeholder='Select Team 2', index=None, on_change=None, key="team2")
        if team1 != team2 and team1 != None and team2 != None: # If the teams are different and not None
            matchups_df = build_all_seasons_df(team1, team2)
            head_to_head_df = create_head_to_head_stats_for_inference(team1, team2, matchups_df)
            # convert the df to json string to pass to the gemini model
            head_to_head_json = head_to_head_df.to_json(orient='records')
            chosen_color, difficulty_rating = color_to_name(color_fixtures(team_fdr_df.iloc[match_team_name_to_id(team1)-1, st.session_state.selected_gameweek-1]))
            # Define the prompt for the Gemini model
            text_prompt = f"{team1} will play against {team2} in the next gameweek. The fixture difficulty rating for this game is {difficulty_rating} which means it will be colored as {chosen_color}. explain why shortly based on their head-to-head stats: {head_to_head_json}"
            # Generate the content using the Gemini model and display it in a markdown
            response = gemini_model.generate_content(text_prompt)
            text = response.text.replace('•', '*')
            st.markdown(text)
