import streamlit as st
import time
import numpy as np
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
from streamlit_extras.row import row

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

def draw_pitch():
    # Create a figure and axes with custom dimensions
    fig, ax = plt.subplots(figsize=(6, 8))  # Adjust figsize as needed
    pitch = VerticalPitch(pitch_color='grass', line_color='white', stripe=True)
    pitch.draw(ax=ax)
    return fig  # Return the figure object


row1 = row([2.6,2,1], vertical_align="center")
# draw the pitch
fig = draw_pitch()
row1.pyplot(fig)


st.sidebar.markdown("### Which data should i show you?")
selected_layers = [
    layer
    for layer_name, layer in {"Expected score":"Expected score", "Next Game": "Next Game", "Price":"Price", "% owned": "% owned"}.items()
    if st.sidebar.checkbox(layer_name, False)
]
