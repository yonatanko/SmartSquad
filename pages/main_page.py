import streamlit as st
import time
import numpy as np
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
from streamlit_extras.row import row
import time
from SmartSquad import colors

st.set_page_config(page_title="Main Page")

margins_css = """
    <style>
        .main > div {
            padding-top: 2rem;
            padding-left: 0rem;
            padding-bottom: 0rem;
        }

        button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
"""

st.markdown(margins_css, unsafe_allow_html=True)

if "show_container" not in st.session_state:
    st.session_state.show_container = True


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


def draw_pitch_with_players(players, subs, colors, selected_stats, player_stats):
    fig, ax = plt.subplots(figsize=(7, 14))
    pitch = VerticalPitch(pitch_color="grass", line_color="white", stripe=True, pad_bottom=25)
    pitch.draw(ax=ax)

    used_positions = {"DEF": 0, "MID": 0, "FWD": 0, "SUB": 0}

    for position, player, club in players+subs:
        if position == "GK":
            x, y = position_mapping[position]
        else:
            x, y = position_mapping[position][used_positions[position]]
            used_positions[position] += 1

        color = colors.get(club, "grey")
        pitch.scatter(x, y, s=600, ax=ax, edgecolors="black", c=color, zorder=2)
        plt.text(y, x + 6, player, fontsize=15, ha="center", va="center")

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

        # Create a text box with background for the formatted stat values
        bbox_props = dict(
            boxstyle="round,pad=0.6", fc="white", ec="black", lw=1, alpha=0.5
        )
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


# extract player, position and club from the st.session_state['players'] list
players = [player.split(",") for player in st.session_state["players"]]

# take out the last player in each position to be used as a substitute
subs = []
for position in ["GK", "DEF", "MID", "FWD"]:
    for i, player in enumerate(players):
        if player[0] == position:
            curr_sub = players.pop(i)
            curr_sub[0] = "SUB"
            subs.append(curr_sub)
            break

# Create a dictionary to store the stats for each player
player_stats = {
    players[0][1]: {
        "Expected score": 5.2,
        "Next Game": "Team 1",
        "Price": 5.5,
        "% owned": 12,
    },
    players[1][1]: {
        "Expected score": 6.5,
        "Next Game": "Team 2",
        "Price": 6.5,
        "% owned": 15,
    },
    players[2][1]: {
        "Expected score": 4.5,
        "Next Game": "Team 3",
        "Price": 4.5,
        "% owned": 10,
    },
    players[3][1]: {
        "Expected score": 5.0,
        "Next Game": "Team 4",
        "Price": 5.0,
        "% owned": 8,
    },
    players[4][1]: {
        "Expected score": 6.0,
        "Next Game": "Team 5",
        "Price": 6.0,
        "% owned": 20,
    },
    players[5][1]: {
        "Expected score": 5.5,
        "Next Game": "Team 6",
        "Price": 5.5,
        "% owned": 18,
    },
    players[6][1]: {
        "Expected score": 5.0,
        "Next Game": "Team 7",
        "Price": 5.0,
        "% owned": 14,
    },
    players[7][1]: {
        "Expected score": 4.5,
        "Next Game": "Team 8",
        "Price": 4.5,
        "% owned": 12,
    },
    players[8][1]: {
        "Expected score": 6.0,
        "Next Game": "Team 9",
        "Price": 6.0,
        "% owned": 16,
    },
    players[9][1]: {
        "Expected score": 5.5,
        "Next Game": "Team 10",
        "Price": 5.5,
        "% owned": 14,
    },
    players[10][1]: {
        "Expected score": 5.0,
        "Next Game": "Team 11",
        "Price": 5.0,
        "% owned": 12,
    },
    subs[0][1]: {
        "Expected score": 4.5,
        "Next Game": "Team 12",
        "Price": 4.5,
        "% owned": 10,
    },
    subs[1][1]: {
        "Expected score": 5.0,
        "Next Game": "Team 13",
        "Price": 5.0,
        "% owned": 8,
    },
    subs[2][1]: {
        "Expected score": 6.0,
        "Next Game": "Team 14",
        "Price": 6.0,
        "% owned": 20,
    },
    subs[3][1]: {
        "Expected score": 5.5,
        "Next Game": "Team 15",
        "Price": 5.5,
        "% owned": 18,
    },
}

# use the club colors from SmartSquad.py

col1, col2, col3, col4 = st.columns([1.1, 0.05, 1.1, 0.05])

with col4:
    if st.button(":back:"):
        st.session_state.show_container = True


def hide_container():
    st.session_state.show_container = False


def show_recommendation():
    st.session_state.show_container = False
    with col3:
        create_recommendation()


def create_recommendation():
    st.markdown(
        """
        <div style="box-shadow: 0px 0px 20px #ccc; padding: 20px; margin-bottom: 20px; margin-top: 10px; border-radius: 15px;">
            <h3>Recommendation</h3>
            <p>Based on the current squad, we recommend the following transfer:</p>
            <ul>
                <li>Transfer out: Player 1</li>
                <li>Transfer in: Player 12</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Draw the pitch in the first column, which will span across all rows on the left side.
with col1:
    fig = draw_pitch_with_players(players, subs, colors, selected_stats, player_stats)
    st.pyplot(fig)

# Now, you can define content in the second column, which will appear to the right of the pitch.
# Each piece of content will be in its own 'row', but since Streamlit is top-down, you'll just continue adding content to col2.
with col3:
    # Create a container that will hold your content
    if st.session_state.show_container:
        with st.container():
            st.markdown(
                """
                <style>
                div.stButton > button:first-child {
                    margin: 5px 5px 5px 45px;
                }
                </style>""",
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div style="box-shadow: 0px 0px 20px #ccc; padding: 20px; margin-bottom: 20px; border-radius: 15px;">
                    Want a recommendation for a good transfer?
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([1, 1], gap="small")

            with col1:
                if st.button("Yes 👍", on_click=show_recommendation):
                    # Perform action for Yes
                    pass

            with col2:
                if st.button("No 👎", on_click=hide_container):
                    # Perform action for No: make the content disappear\
                    pass
