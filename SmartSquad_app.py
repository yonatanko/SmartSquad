import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="SmartSquad",
    page_icon="⚽",
    initial_sidebar_state="collapsed"
)

players_names = [f"{i}" for i in range(15)]

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
    player_name = st.session_state.selected_player
    if player_name not in st.session_state['players']:
        st.session_state['players'].append(player_name)
        check_count()  # Update the count and disabled state after adding
    else:
        st.warning("Player already added", icon="⚠️")

# Split the layout into two columns
col1, col3 = st.columns([1, 2])

# Manage player addition logic based on the count
def check_count():
    if len(st.session_state['players']) == 3:
        st.session_state["disabled"] = True
    else:
        st.session_state["disabled"] = False

# Column 1: Player selection and "Add" button
with col1:
    selected_player = st.selectbox("Select player", players_names, key="selected_player", disabled=st.session_state["disabled"])
    add_player_button = st.button("Add Player", on_click=add_player, key="add_player", disabled=st.session_state["disabled"])
    

# Column 2: List and remove players
with col3:
    st.write(f"{len(st.session_state['players'])} / 15 Players picked")
    # Function to remove a player
    def remove_player(player_to_remove):
        st.session_state['players'].remove(player_to_remove)
        check_count()  # Update the count and disabled state after removal
        st.experimental_rerun()

    # Display each player with a remove button
    for index, player in enumerate(st.session_state['players'], start=1):
        player_col, remove_col = st.columns([3, 1])
        player_col.write(f"Player {index}: {player}")
        if remove_col.button("Remove 🗑️", key=f"remove_{index}"):
            remove_player(player)

    # Place the Done button outside check_count to avoid DuplicateWidgetID error
    if len(st.session_state['players']) == 3:
        if st.button("Done", key="done_picking"):
            switch_page("Main Page")

