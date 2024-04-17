import streamlit as st

def get_gemini_key():
    gemini_key = st.secrets["gemini_key"] # get the api key from the streamlit secrets

    # here you can set the api key for the model by using the configure method:
    #
    # with open('config.json', 'r') as file:
    #         config = json.load(file)
    # gemini_key = config['gemini_key']
    #
    return gemini_key