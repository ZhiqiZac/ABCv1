import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

st.title("AI Bootcamp")
st.subheader("Buying HDB flat in the resale market")
st.write("Zhiqi Wang")

st.markdown("Topic: This is the home page, and the assignment is due by 27 Oct")

st.write("Click on the side bar to explore the various functionalities of the two tools developed, as well as some background for interested readers")


# Set up and run this Streamlit App
import streamlit as st
from logics.rag import process_user_message, process_user_message2
import pandas as pd
from openai import OpenAI
import tiktoken

#load passwords from env file for security
import os

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("LLM Bot that responds to queries on HDB resale")

form = st.form(key="form")
form.subheader("Prompt")

user_prompt = form.text_area("Enter your prompt here", height=200)

if form.form_submit_button("Submit"):
    #st.toast(f"User Input Submitted - {user_prompt}") #this 
    response = process_user_message(user_prompt)

    print(f"User Input is {user_prompt}")
    st.write(response) # <--- This displays the response generated by the LLM onto the frontend 🆕

with st.expander("Disclaimer"):
    st.write(
    """

    IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

    Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

    Always consult with qualified professionals for accurate and personalized advice.

    """
    )
    
    