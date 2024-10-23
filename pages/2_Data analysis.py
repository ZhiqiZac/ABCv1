
import streamlit as st
import pandas as pd
import plotly.express as px

# Title for the app
st.title("CSV Data Visualization App of HDB Resale Flats with Flat Type, Location, and Price Filters :building_construction:")
st.write("This was based on public data of year 2021, to provide a guidelines of the choices that buyers might have")


# Load the CSV data into a DataFrame
df = pd.read_csv("data/Resale prices.csv")

# Show the first few rows of the DataFrame
st.write("Here is the preview of the data:")



#some data preparation to ensure all the relevant 
df["Year"]= df['month'].str.slice(0,4) #created the year variable. 
df["Year"] = df["Year"].astype(int)
df = df[(df['Year'] == 2021)] #narrowed to most recent year
#df=df.head(2000) #reduce size of file to reduce likelihood of crashing, to adapt if process power is better 
df['Location']= df['town']
df['Price']=df['resale_price']
st.dataframe(df.head())

# Filter by Location
unique_locations = sorted(df['Location'].unique().tolist())
selected_location = st.radio("Filter by Location:", unique_locations)

# Filter by Flat Type 
unique_flattypes = sorted(df['flat_type'].unique().tolist())
selected_flat_type = st.radio("Filter by Location:", unique_flattypes)


# Get the minimum and maximum values of the Price column
min_price = df['Price'].min()
max_price = df['Price'].max()

# Slider to filter by minimum and maximum price
price_range = st.slider(
        "Select the price range:", 
        min_value=float(min_price), 
        max_value=float(max_price), 
        value=(float(min_price), float(max_price))
)

filtered_df = df[
    (df['flat_type'] == selected_flat_type) & 
    (df['Location'] == selected_location) & 
    ((df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1]))
    ]

# Display the filtered data
st.write(f"Filtered Data for Flat type: {selected_flat_type}, Location: {selected_location}, Price Range: {min_price, max_price}:")
st.write(f"Number of flats available: {len(filtered_df)}")
st.dataframe(filtered_df.head())

#Data visualisation for plots on the range

# Choose numeric columns for plotting (aside from the filtered columns)
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if len(numeric_columns) >= 2:
    x_axis = st.selectbox("Select the X-axis:", numeric_columns)
    y_axis = st.selectbox("Select the Y-axis:", numeric_columns)

    # Plot the filtered data using Plotly
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color='Location', 
                        title=f"Scatter plot of {x_axis} vs {y_axis}")
    st.plotly_chart(fig)
else:
    st.warning("Not enough numeric columns for plotting.")

with st.expander("Some finer details"):
    summary = filtered_df.describe()
    st.write (summary)

