
import streamlit as st


st.title("About Us")

st.write("A detailed page outlining the project scope, objectives, data sources, and features.")


st.subheader("A. Project Scope :mount_fuji:")
st.write("The application allows the user to better understand the housing options for his flat of interest based on a LLM chatbot and data analytics to help first time buyers better navigate the complicated process. ")

st.subheader("B. Objectives :tokyo_tower:")
st.write("Objective of the project is to provide an independent opinion, backed by data, to provide housing participants information to make better decisions.")

st.subheader("C. Data source :statue_of_liberty:")
st.write("The data source are from publicly accessible data from HDB website for the Bot, accessed in October 2024. The HDB resale statistics was access through www.data.gov.sg")
st.write( "Resale prices data is available on https://data.gov.sg/datasets?query=HDB+resale+price&page=1&resultId=189")

st.subheader("D. Features :moyai:")
st.write("""
         1. Uses an intelligent bot to digest information from HDB, into simple to understand responses that are accurate. 
         2. Uses a data analysis tool to suggest the average housing price in an area, and allow for users to interact, to provide more granular data.
         """)


st.write("Made by Zhiqi Wang, as part of the AI Bootcamp Champions course submissions")
st.write ("A big thank you for all who have guided me through the process, and answer my questions :heart:")