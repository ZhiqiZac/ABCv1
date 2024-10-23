import streamlit as st

st.title ("Methodology")

st.write("""A comprehensive explanation of the data flows and implementation details.
A flowchart illustrating the process flow for each of the use cases in the application. For example, if the application has two main use cases: a) chat with information and b) intelligent search, each of these use cases should have its own flowchart.
Refer to the sample here Links to an external site. for the samples of the flowcharts and methodology (Slide 13, 14, and 15).""")



st.subheader ("LLM Bot")
st.write ("""
          1. Access the replies from HDB Website 
          2. Using Retrival Augmented Generation (RAG) to reduce the likelihood of hallucinations and provide verfiable answers. 
          3. 

          N. A wider variety of data can be added to allow a greater range of questions to be answered. Other model such as Agent models can be used in the future to provide higher level of detail. 

          
          """)


st.subheader("Data Analytics")
st.write ("""
          1. Access data from data.gov.sg, looking specifically at the more recent housing transactions, as it will be closer to new launches 
          2. Develop a dataframe using the existing data, and functions based on the data, for columns that are not previously available
          3. Setup the user input functions via radio (for flat type and Location), and slider (budget). These parameters are likely some of the key decision criteria for potential buyers of flats
          4. Provide data analytics to understand the availability of houses,
          5. Future work: include more features to filter, based on existing data. Use of live data such as HDB resale portal to provide more up-to-date information.   
        
          """)