import streamlit as st
import time
import numpy as np

st.set_page_config(
    page_title="Main Page"
    )

st.markdown("# Plotting Demo")
# st.sidebar.header("Main Page")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

st.sidebar.markdown("### Which data should i show you?")
selected_layers = [
    layer
    for layer_name, layer in {"Expected score":"Expected score", "Next Game": "Next Game", "Price":"Price", "% owned": "% owned"}.items()
    if st.sidebar.checkbox(layer_name, False)
]

# print chosen players from the welcome page
if "players" in st.session_state and len(st.session_state["players"]) == 3:
    # print as str with commas
    st.info(f"Players chosen: {', '.join(st.session_state['players'])}")
else:
    st.info("No players chosen yet")
