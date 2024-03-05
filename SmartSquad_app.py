import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from st_pages import Page, add_page_title, show_pages, hide_pages


st.set_page_config(
    page_title="SmartSquad",
    page_icon="⚽",
    initial_sidebar_state="collapsed"
)

st.write("# SmartSquad ⚽")

if st.button("Go to new page"):
    switch_page("main page")


