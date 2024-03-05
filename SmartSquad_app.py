import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, add_page_title, show_pages, hide_pages



st.set_page_config(
    page_title="SmartSquad",
    page_icon="⚽",
    initial_sidebar_state="collapsed"
)

# Initialize session state for players list if not already set
if 'players' not in st.session_state:
    st.session_state['players'] = []

st.write("# SmartSquad ⚽")
st.write("""Welcome to SmartSquad!
         This app is designed to help you manage your fantasy football team.
         You can use it to track your players' performance, compare them to other players, and make informed decisions about your team.""")


# Function to add a player
def add_player():
    player_name = st.session_state.player_name
    # Check if the player_name is not empty before adding
    if player_name:
        # Append the player name to the session state list
        st.session_state['players'].append(player_name)
        # Clear the input box after adding
        st.session_state.player_name = ""
    else:
        st.warning("Please enter a player name.")

player_name = st.text_input("Enter player name", key="player_name", on_change=add_player)
# add counter of players as text
st.write(f"{len(st.session_state['players'])} / 15 Players picked")

# Function to remove a player
def remove_player(player_to_remove):
    st.session_state['players'].remove(player_to_remove)
    # Forcing a rerun to update the UI
    st.experimental_rerun()

# Display each player with a remove button
for index, player in enumerate(st.session_state['players'], start=1):
    col1, col2 = st.columns([4, 1])
    col1.write(f"Player {index}: {player}")
    # Unique key for each button using player index
    if col2.button("Remove 🗑️", key=f"remove_{index}"):
        remove_player(player)

    
if st.button("Done"):
    switch_page("main page")


