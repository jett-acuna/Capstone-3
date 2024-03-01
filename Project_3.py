# Import the required libraries and dependencies
import pandas as pd
import hvplot.pandas
import datetime as dt
import holoviews as hv
from prophet import Prophet
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import streamlit as st

## Step-1 Clean Data
# Import Meta Movie Data
meta_data = pd.read_csv(
    Path("movies_data/movies_metadata.csv"),
    index_col="title")

# Drop null values
meta_data.dropna(axis=0, inplace=True)

# Sort Data with Descending Vote_average
meta_data.sort_values(("vote_average"), ascending=[False])

# Drop Unecassary Columns
meta_data.drop(['adult', 'belongs_to_collection', 'homepage', 'id',
       'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 'status', 'tagline', 'video' ], axis=1)

## Step-2 Create Linear Regression Model

# Create the Features for the Linear Regression Model
features = ['vote_count', 'revenue', 'runtime', 'popularity']

target = 'vote_average'

X = meta_data[features]
y = meta_data[target]

# Test Train Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Data into StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Data into Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Create Predictions based on Model
predictions = model.predict(X_test_scaled)

# Create new DataFrame for Customer Inputs
Customer_data = pd.DataFrame({'vote_count': [500], 'revenue': [1000000], 'runtime': [120], 'popularity': [20]})
Customer_data_scaled = scaler.transform(Customer_data)
predicted_rating = model.predict(Customer_data_scaled)

## Step-3 Create Streamlit Application

# Create Title for Streamlit Application
st.title('Movie Rating Prediction App')

# Assemble Customizable Sidebar
st.sidebar.header('Enter Movie Features:')
vote_count = st.sidebar.slider('vote_count', min_value=0, max_value=10000, value=500)
revenue = st.sidebar.slider('revenue', min_value=0, max_value=100000000, value=1000000)
runtime = st.sidebar.slider('runtime (minutes)', min_value=0, max_value=300, value=120)
popularity = st.sidebar.slider('popularity', min_value=0, max_value=100, value=50)

# Create Output
if st.sidebar.button('Predict Rating'):
    Customer_input = pd.DataFrame({'vote_count': [vote_count], 'revenue': [revenue], 'runtime': [runtime], 'popularity': [popularity]})
    Customer_input_scaled = scaler.transform(Customer_input)
    Predicted_vote_average = model.predict(Customer_input_scaled)
    
    st.subheader('Predicted Movie Rating:')
    st.write(Predicted_vote_average[0])
    
## Step-4 Run Streamlit Application