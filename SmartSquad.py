import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, show_pages, add_page_title
import pandas as pd
from annotated_text import annotated_text

st.set_page_config(
    page_title="SmartSquad",
    page_icon="⚽",
    initial_sidebar_state="collapsed",
    layout="wide",
)

margins_css = """
    <style>
        .main > div {
            padding-top: 2rem;
        }
    </style>
"""

st.markdown(margins_css, unsafe_allow_html=True)

show_pages(
    [
        Page("SmartSquad.py", "Welcome", ":wave:"),
        Page("pages/main_page.py", "Main Page", ":house:"),
        Page("pages/stats_page.py", "Stats Page", ":chart_with_upwards_trend:"),
    ]
)

# based on the teams in teams.csv, build colors dict
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

# Initialize the players list
players_df = pd.read_csv("players_raw.csv")
players_df = players_df[["web_name", "element_type", "team"]]

teams_df = pd.read_csv("teams.csv")
teams_df = teams_df[["id", "name"]].rename(columns={"id": "team"})
# map team id to team name
players_df = players_df.merge(teams_df, on="team")
players_df = players_df[["web_name", "element_type", "name"]]
players_df = players_df.rename(
    columns={"web_name": "Player", "element_type": "Position", "name": "Team"}
)
players_df["Position"] = players_df["Position"].map(
    {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
)
players_df["list_item"] = (
    players_df["Position"] + "," + players_df["Player"] + "," + players_df["Team"]
)

players = players_df["list_item"].tolist()

# Initialize session state variables
if "players" not in st.session_state:
    st.session_state["players"] = []

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

if "position_counter" not in st.session_state:
    st.session_state["position_counter"] = {
        "GK": 0,
        "DEF": 0,
        "MID": 0,
        "FWD": 0,
    }

if "clubs_counter" not in st.session_state:
    st.session_state["clubs_counter"] = {}

hide_img_fs = """
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
"""

st.markdown(hide_img_fs, unsafe_allow_html=True)


# Function to add a player
def add_player():
    player_name = st.session_state.selected_player
    if player_name not in st.session_state["players"]:
        # check positional constraints
        position = player_name.split(",")[0]
        if position == "GK" and st.session_state["position_counter"]["GK"] > 1:
            with col3:
                st.warning("Maximum 2 goalkeepers allowed", icon="⚠️")
                return
        elif position == "DEF" and st.session_state["position_counter"]["DEF"] > 5:
            with col3:
                st.warning("Maximum 6 defenders allowed", icon="⚠️")
                return
        elif position == "MID" and st.session_state["position_counter"]["MID"] > 5:
            with col3:
                st.warning("Maximum 6 midfielders allowed", icon="⚠️")
                return
        elif position == "FWD" and st.session_state["position_counter"]["FWD"] > 3:
            with col3:
                st.warning("Maximum 4 forwards allowed", icon="⚠️")
                return
            
        # check club constraints
        club = player_name.split(",")[2]
        if club in st.session_state["clubs_counter"] and st.session_state["clubs_counter"][club] > 2:
            with col3:
                st.warning("Maximum 3 players allowed from the same club", icon="⚠️")
                return
        
        st.session_state["players"].append(player_name)
        st.session_state["position_counter"][position] += 1
        
        if club in st.session_state["clubs_counter"]:
            st.session_state["clubs_counter"][club] += 1
        else:
            st.session_state["clubs_counter"][club] = 1

        check_count()  # Update the count and disabled state after adding
    else:
        with col3:
            st.warning("Player already added", icon="⚠️")

# Manage player addition logic based on the count
def check_count():
    if len(st.session_state["players"]) == 15:
        st.session_state["disabled"] = True
    else:
        st.session_state["disabled"] = False

# Function to remove a player
def remove_player(player_to_remove):
    st.session_state["players"].remove(player_to_remove)
    check_count()  # Update the count and disabled state after removal
    st.experimental_rerun()

# Split the layout into two columns
col1, col3 = st.columns([1.2, 1])
        

# Column 1: Player selection and "Add" button
with col1:
    (
        _,
        image_pos,
        _,
    ) = st.columns([1, 2, 1])
    with image_pos:
        st.image("app_image.png", width=250)

    st.write("""Welcome to SmartSquad!
         This app is designed to help you manage your fantasy football team.
         You can use it to track your players' performance, compare them to other players, and make informed decisions about your team.""")

    selected_player = st.selectbox(
        "Select player",
        players,
        key="selected_player",
        disabled=st.session_state["disabled"],
        placeholder="Select a player to add to your squad",
        index=None,
    )
    add_player_button = st.button(
        "Add Player",
        on_click=add_player,
        key="add_player",
        disabled=st.session_state["disabled"],
    )


# Column 2: List and remove players
with col3:
    st.write(f"{len(st.session_state['players'])} / 15 Players picked")

    # Display each player with a remove button
    for index, player in enumerate(st.session_state["players"], start=1):
        player_col, remove_col = st.columns([3, 1])
        with player_col:
            # drop the team name from the list item
            player_and_pos = player.split(",")[0] + "\u00a0" * 13 + player.split(",")[1]
            annotated_text(
                (player_and_pos, player.split(",")[2], colors[player.split(",")[2]])
            )
        if remove_col.button("Remove 🗑️", key=f"remove_{index}", ):
            remove_player(player)

    # Place the Done button outside check_count to avoid DuplicateWidgetID error
    if len(st.session_state["players"]) == 15:
        # check positional constraints
        if (
            st.session_state["position_counter"]["GK"] < 2
            or st.session_state["position_counter"]["DEF"] < 3
            or st.session_state["position_counter"]["MID"] < 4
            or st.session_state["position_counter"]["FWD"] < 2
        ):
            with col1:
                st.warning(
                    "You must pick at least 2 GK, 4 DEF, 4 MID, and 2 FWD",
                    icon="⚠️",
                )
        if st.button("Done", key="done_picking"):
            switch_page("Main Page")
