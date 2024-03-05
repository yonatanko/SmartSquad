import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="SmartSquad",
    page_icon="⚽",
    initial_sidebar_state="collapsed"
)

# Initialize session state for players list if not already set
if 'players' not in st.session_state:
    st.session_state['players'] = []

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

st.write("# SmartSquad ⚽")
st.write("""Welcome to SmartSquad!
         This app is designed to help you manage your fantasy football team.
         You can use it to track your players' performance, compare them to other players, and make informed decisions about your team.""")

# Function to add a player
def add_player():
    player_name = st.session_state.player_name
    if player_name:  # Check if the player_name is not empty before adding
        st.session_state['players'].append(player_name)  # Append the player name
        st.session_state.player_name = ""  # Clear the input box after adding
    else:
        st.warning("Please enter a player name.")

# Split the layout into two columns
col1, col2, col3 = st.columns([0.8, 0.2,  1])

# Column 1: Player input and "Done" button
with col1:
    player_name = st.text_input("Enter player name", key="player_name", on_change=add_player, disabled=st.session_state.disabled)

# Column 2: List and remove players
with col3:
    st.write(f"{len(st.session_state['players'])} / 15 Players picked")
    # Function to remove a player
    def remove_player(player_to_remove):
        st.session_state['players'].remove(player_to_remove)
        st.experimental_rerun()

    # Display each player with a remove button
    for index, player in enumerate(st.session_state['players'], start=1):
        player_col, remove_col = st.columns([1, 1])
        player_col.write(f"Player {index}: {player}")
        if remove_col.button("Remove 🗑️", key=f"remove_{index}"):
            remove_player(player)

    # if the count of players is 15, avoid adding more players and present the "Done" button
    if len(st.session_state['players']) == 14:
        # disable the input box
        st.session_state["disabled"] = True
    else:
        # enable the input box
        st.session_state["disabled"] = False
    if len(st.session_state['players']) == 15:
        if st.button("Done", key="done"):
            switch_page("main page")
