import streamlit as st
import time
import numpy as np
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
from streamlit_extras.row import row
import time
from SmartSquad import colors
import pandas as pd
from analytics.fpl_utils.fpl_api_collection import get_fixt_dfs, get_bootstrap_data, get_name_and_pos_and_team_dict

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


def color_fixtures(val):
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
    else:
        # string green
        return "#008000"

if "show_container" not in st.session_state:
    st.session_state.show_container = True

if "selected_gameweek" not in st.session_state:
    st.session_state.selected_gameweek = 1

if "show_transfer_recommendation" not in st.session_state:
    st.session_state.show_transfer_recommendation = False

if len(st.session_state["players"]) != 15:
    st.warning("Please select 15 players in the Welcome page", icon="⚠️")
    st.stop()


def generate_position_mapping(num_defenders, num_midfielders, num_forwards, pitch_height=80):
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


# extract number of players in each position from the st.session_state['players'] list
num_defenders = len(
    [player for player in st.session_state["players"] if player.split(",")[0] == "DEF"]
) -1
num_midfielders = len(
    [player for player in st.session_state["players"] if player.split(",")[0] == "MID"]
) -1
num_forwards = len(
    [player for player in st.session_state["players"] if player.split(",")[0] == "FWD"]
) -1

position_mapping = generate_position_mapping(
    num_defenders, num_midfielders, num_forwards
)

# Modify the sidebar section to allow the selection of a statistic
st.sidebar.markdown("### Select a statistic to display:")
selected_stats = [
    layer
    for layer_name, layer in {
        "Expected score": "Expected score",
        "Next Game": "Next Game",
        "Price": "Price",
        "% owned": "% owned",
    }.items()
    if st.sidebar.checkbox(layer_name, False)
]

def draw_pitch_with_players(starting_11, subs, colors, selected_stats, player_stats):
    fig, ax = plt.subplots(figsize=(7, 14))
    pitch = VerticalPitch(pitch_color="grass", line_color="white", stripe=True, pad_bottom=25)
    pitch.draw(ax=ax)

    used_positions = {"DEF": 0, "MID": 0, "FWD": 0, "SUB": 0}

    for position, player, club in starting_11+subs:
        if position == "GK":
            x, y = position_mapping[position]
        else:
            x, y = position_mapping[position][used_positions[position]]
            used_positions[position] += 1

        color = colors.get(club, "grey")
        pitch.scatter(x, y, s=600, ax=ax, edgecolors="black", c=color, zorder=2)
        plt.text(y, x + 6, player, fontsize=15, ha="center", va="center", fontdict={"fontweight": "bold"})

        # Iterate over each player and fetch & display the selected stats below the player's name
        stat_values = [
            f'{player_stats[player].get(stat, "N/A")}' for stat in selected_stats
        ]
        # Insert a newline character after every two stats
        formatted_stat_values = " | ".join(stat_values[:2])
        if len(stat_values) > 2:
            formatted_stat_values += "\n" + " | ".join(stat_values[2:])
        if len(stat_values) == 0:
            formatted_stat_values = ""
        
        fixture_difficulty_color = color_fixtures(
            team_fdr_df.iloc[match_team_name_to_id(club)-1, st.session_state.selected_gameweek-1]
        )
        # if stats has next game, display the fixture difficulty color, else white
        if "Next Game" in selected_stats:
            bbox_props = dict(boxstyle="round,pad=0.6", fc=fixture_difficulty_color, ec="black", lw=1, alpha=0.5)
        else:
            bbox_props = dict(boxstyle="round,pad=0.6", fc="white", ec="black", lw=1, alpha=0.5)
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

teams_df = pd.DataFrame(get_bootstrap_data()['teams'])
teams_df.to_csv("teams.csv")

def match_team_name_to_id(team_name):
    return teams_df[teams_df['name'] == team_name]['id'].values[0]

# extract player, position and club from the st.session_state['players'] list
players = [player.split(",") for player in st.session_state["players"]]
num_to_string_pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# generate dummy expected points for each player in each gameweek (there are 38 gameweeks in a season)
# make a df with the player names (selected in smartSquad page) as the index and the columns as the gameweeks
expected_points_selected = np.random.randint(1, 10, size=(15, 38))
players_points_df = pd.DataFrame(expected_points_selected, columns=[f"GW{i}" for i in range(1, 39)])
players_points_df.index = [player[1] for player in players]

name_to_pos_dict, name_to_team_dict = get_name_and_pos_and_team_dict()
all_players = [(num_to_string_pos[name_to_pos_dict[player]], player, name_to_team_dict[player]) for player in name_to_pos_dict.keys()]
not_selected_players = [player for player in all_players if player[1] not in players_points_df.index]

expected_points_not_selected = np.random.randint(1, 10, size=(len(not_selected_players), 38))
not_selected_players_points_df = pd.DataFrame(expected_points_not_selected, columns=[f"GW{i}" for i in range(1, 39)])
not_selected_players_points_df.index = [player[1] for player in not_selected_players]

# calculate the best starting 11 based on players_points_df in a given gameweek and put the rest on the bench
subs_names = []
gameweek_df = players_points_df["GW" + str(st.session_state.selected_gameweek)]
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

team_fdr_df, team_fixt_df, _, _ = get_fixt_dfs()
team_fdr_df.to_csv("team_fdr.csv")
print(f"gameweek: {st.session_state.selected_gameweek}")

# Create a dictionary to store the stats for each player
player_stats = {
    player[1]: {
        "Expected score": players_points_df.loc[player[1], "GW" + str(st.session_state.selected_gameweek)],
        "Next Game": team_fixt_df.iloc[start_11_teams_ids[starting_11.index(player)]-1, st.session_state.selected_gameweek-1 if st.session_state.selected_gameweek < 37 else 37],
        "Price": 5.0,  
        "% owned": "10%" 
    }
    for player in starting_11
}

# Create a dictionary to store the stats for each player
subs_stats = {
    player[1]: {
        "Expected score": players_points_df.loc[player[1], "GW" + str(st.session_state.selected_gameweek)],
        "Next Game": team_fixt_df.iloc[subs_teams_ids[subs.index(player)]-1, st.session_state.selected_gameweek-1 if st.session_state.selected_gameweek < 37 else 37],
        "Price": 5.0,  # Dummy value
        "% owned": 10,  # Dummy value
    }
    for player in subs
}

# Create a dictionary to store the stats for each player
player_stats.update(subs_stats)

# use the club colors from SmartSquad.py

col1, col2, col3, col4 = st.columns([2, 0.3, 3, 0.05])

with col4:
    if st.button(":back:"):
        st.session_state.show_container = True
        st.session_state.show_transfer_recommendation = False


def hide_container():
    st.session_state.show_container = False


def show_recommendation():
    st.session_state.show_container = False
    st.session_state.show_transfer_recommendation = True
    with col3:
        create_recommendation()

def create_recommendation():
    # create a selectbox to select the number of gameweeks to consider
    num_gameweeks = st.selectbox("Select the number of gameweeks to consider", range(1, 38 - st.session_state.selected_gameweek + 1), index=None) # consider the remaining gameweeks
    if num_gameweeks:
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

        if potential_swaps:
            first_col, second_col, third_col, fourth_col = st.columns([1.5, 1.5, 0.0001, 1.5])
            # choose the swap that will give the best additional points
            best_swap = max(potential_swaps, key=lambda x: x[2])
            selected_player = best_swap[0]
            selected_player_next_games_and_points = selected_points_df.loc[selected_player]
            selected_player_next_games_and_points["Total"] = selected_player_next_games_and_points.sum(axis=0)
            selected_player_next_games_and_points = pd.DataFrame(selected_player_next_games_and_points).rename(columns={selected_player: "Points"})
            not_selected_player = best_swap[1]
            not_selected_player_next_games_and_points = not_selected_points_df.loc[not_selected_player]
            not_selected_player_next_games_and_points["Total"] = not_selected_player_next_games_and_points.sum(axis=0)
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
            

            with first_col:
                # center aligned subheader
                st.markdown(f'## {selected_player}')
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
            with second_col:
                for i in range(8):
                    st.write("")
                st.markdown("#### Subsitute with:")
            with fourth_col:
                st.markdown(f'## {not_selected_player}')
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
        else:
            st.write("No potential swaps found")


# Draw the pitch in the first column, which will span across all rows on the left side.
with col1:
    inner_col1, inner_col2, inner_col3 = st.columns(3)
    # gameweek selection
    with inner_col1:
        if st.button(":arrow_backward:"):
            if st.session_state.selected_gameweek > 1:
                st.session_state.selected_gameweek -= 1
                st.session_state.show_container = True
                st.session_state.show_transfer_recommendation = False
                st.rerun()
            else:
                st.warning("You are already at the first gameweek", icon="⚠️")
    with inner_col2:
        # rounded text area with white background and the selected gameweek
        st.markdown(
            f"""
            <div style="margin-bottom: 20px; margin-top: 8px; margin-right: 20px; border-radius: 15px; text-align: center; background-color: white; color: black;">
                <p style= "font-size": 20px >{"GW" + str(st.session_state.selected_gameweek)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if inner_col3.button(":arrow_forward:"):
        if st.session_state.selected_gameweek < 38:
            st.session_state.selected_gameweek += 1
            st.session_state.show_container = True
            st.session_state.show_transfer_recommendation = False
            st.rerun()
        else:
            st.warning("You are already at the last gameweek", icon="⚠️")

    fig = draw_pitch_with_players(starting_11, subs, colors, selected_stats, player_stats)
    st.pyplot(fig)

with col3:
    if st.session_state.show_container:
        with st.container():
            st.markdown(
                """
                <div style="box-shadow: 0px 0px 20px #ccc; padding: 20px; margin-bottom: 20px; margin-right: 130px; border-radius: 15px; text-align: center;">
                    Want a recommendation for a good transfer?
                </div>
                """,
                unsafe_allow_html=True,
            )

            inside_col1, inside_col2 = st.columns(2)

            with inside_col1:
                if st.button("Yes 👍", on_click=show_recommendation):
                    # Perform action for Yes
                    pass

            with inside_col2:
                if st.button("No 👎", on_click=hide_container):
                    # Perform action for No: make the content disappear\
                    pass

    if st.session_state.show_transfer_recommendation:
        create_recommendation()

    if "Next Game" in selected_stats or st.session_state.show_transfer_recommendation:
        # explain the user what the colors mean by shoiwng the color legend
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