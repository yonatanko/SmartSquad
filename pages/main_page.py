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

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
