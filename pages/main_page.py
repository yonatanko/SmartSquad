import streamlit as st
import time
import numpy as np
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
from streamlit_extras.row import row
import time

st.set_page_config(
    page_title="Main Page"
    )

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

if 'show_container' not in st.session_state:
    st.session_state.show_container = True


def generate_position_mapping(num_defenders, num_midfielders, num_forwards, pitch_height=80, pitch_width=100):
    position_mapping = {'Goalkeeper': (10, pitch_height // 2)}  # Goalkeeper's position is fixed at the center of the goal line
    
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

    position_mapping['Defender'] = [(32, y) for y in generate_y_positions(num_defenders)]
    position_mapping['Midfielder'] = [(60, y) for y in generate_y_positions(num_midfielders)]
    position_mapping['Forward'] = [(90, y) for y in generate_y_positions(num_forwards)]
    
    return position_mapping


# Assuming maximum number of players per role (you can adjust these numbers)
num_defenders = 4
num_midfielders = 4
num_forwards = 2

position_mapping = generate_position_mapping(num_defenders, num_midfielders, num_forwards)

def draw_pitch_with_players(players, club_colors):
    fig, ax = plt.subplots(figsize=(6, 8))
    pitch = VerticalPitch(pitch_color='grass', line_color='white', stripe=True)
    pitch.draw(ax=ax)

    # Keep track of used positions to avoid overlap
    used_positions = {'Defender': 0, 'Midfielder': 0, 'Forward': 0}

    for player, position, club in players:
        if position == 'Goalkeeper':
            x, y = position_mapping[position]
        else:
            x, y = position_mapping[position][used_positions[position]]
            used_positions[position] += 1

        color = club_colors.get(club, 'grey')  # Default to grey if club color not found
        pitch.scatter(x, y, s=600, ax=ax, edgecolors='black', c=color, zorder=2)
        plt.text(y, x-8, player, fontsize=10, ha='center', va='center')

    return fig

# Example usage
players = [
    ('Player 1', 'Goalkeeper', 'Club A'),
    ('Player 2', 'Forward', 'Club B'),
    ('Player 3', 'Defender', 'Club C'),
    ('Player 4', 'Midfielder', 'Club A'),
    ('Player 5', 'Midfielder', 'Club B'),
    ('Player 6', 'Midfielder', 'Club C'),
    ('Player 7', 'Midfielder', 'Club A'),
    ('Player 8', 'Forward', 'Club B'),
    ('Player 9', 'Defender', 'Club C'),
    ('Player 10', 'Defender', 'Club A'),
    ('Player 11', 'Defender', 'Club D')
]

club_colors = {
    'Club A': 'blue',
    'Club B': 'red',
    'Club C': 'green',
    'Club D': 'purple'
}

col1, col2, col3, col4 = st.columns([1.1, 0.05, 1.1, 0.05])

with col4:
    if st.button(':back:'):
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
        unsafe_allow_html=True
    )

# Draw the pitch in the first column, which will span across all rows on the left side.
with col1:
    fig = draw_pitch_with_players(players, club_colors)
    st.pyplot(fig)

# Now, you can define content in the second column, which will appear to the right of the pitch.
# Each piece of content will be in its own 'row', but since Streamlit is top-down, you'll just continue adding content to col2.
with col3:
    # Create a container that will hold your content
    if st.session_state.show_container:
        with st.container():
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    margin: 5px 5px 5px 45px;
                }
                </style>""", unsafe_allow_html=True)

            st.markdown(
                """
                <div style="box-shadow: 0px 0px 20px #ccc; padding: 20px; margin-bottom: 20px; border-radius: 15px;">
                    Want a recommendation for a good transfer?
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns([1, 1], gap="small")

            with col1:
                if st.button("Yes 👍", on_click=show_recommendation):
                    # Perform action for Yes
                    pass

            with col2:
                if st.button('No 👎', on_click=hide_container):
                    # Perform action for No: make the content disappear\
                    pass
                

st.sidebar.markdown("### Which data should i show you?")
selected_layers = [
    layer
    for layer_name, layer in {"Expected score":"Expected score", "Next Game": "Next Game", "Price":"Price", "% owned": "% owned"}.items()
    if st.sidebar.checkbox(layer_name, False)
]
