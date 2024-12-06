import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# Load dataset
@st.cache_data
def load_data(file_path="dataset/earthquake_data.csv"):
    data = pd.read_csv(file_path)
    return data

# Preprocess data
def preprocess_data(data):
    # Convert date_time to datetime
    data['date_time'] = pd.to_datetime(data['date_time'], format='%d-%m-%Y %H:%M')
    
    # Sort data
    data = data.sort_values('date_time')
    
    # Feature engineering
    data['time_since_last_event'] = data['date_time'].diff().dt.total_seconds().fillna(0)
    data['latitude_diff'] = data['latitude'].diff().fillna(0)
    data['longitude_diff'] = data['longitude'].diff().fillna(0)
    data['depth_diff'] = data['depth'].diff().fillna(0)
    data['hour'] = data['date_time'].dt.hour
    data['day_of_week'] = data['date_time'].dt.dayofweek
    data['month'] = data['date_time'].dt.month
    
    # Target: Is Magnitude >= 4 (Moderate earthquake)
    data['significant_event'] = (data['magnitude'] >= 4).astype(int)
    
    # Drop unnecessary columns
    data = data.drop(columns=['date_time'])
    
    # Split features and target
    features = ['latitude', 'longitude', 'depth', 'time_since_last_event', 
                'latitude_diff', 'longitude_diff', 'depth_diff', 
                'hour', 'day_of_week', 'month']
    X = data[features]
    y = data['significant_event']
    
    return train_test_split(X, y, test_size=0.2, random_state=42), data

# Train model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Streamlit App
st.title("QuakeSense: Seismic Activity Forecasting")
st.markdown("An ML-powered tool for earthquake risk prediction.")

# Data Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset.")
    data = load_data()

# Show dataset preview
st.write("### Dataset Preview")
st.dataframe(data.head())

# Preprocess data
(X_train, X_test, y_train, y_test), processed_data = preprocess_data(data)

# Train the model
model = train_model(X_train, y_train)

# Predictions and Performance
y_pred = model.predict(X_test)
st.write("### Model Performance")
st.text(classification_report(y_test, y_pred))

# Visualizations
st.write("### Earthquake Risk Map")
map_data = processed_data[['latitude', 'longitude', 'significant_event']].copy()
map_data['weight'] = map_data['significant_event'] * 10  # Weight for significant events
m = folium.Map(location=[map_data['latitude'].mean(), map_data['longitude'].mean()], zoom_start=5)
HeatMap(map_data[['latitude', 'longitude', 'weight']].values, radius=10).add_to(m)
st_folium(m, width=700, height=500)

# User Input for Prediction
st.sidebar.header("Risk Prediction")
latitude = st.sidebar.number_input("Latitude", value=35.0, step=0.1)
longitude = st.sidebar.number_input("Longitude", value=-120.0, step=0.1)
depth = st.sidebar.number_input("Depth (km)", value=10.0, step=1.0)
hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Day of the Week", range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
month = st.sidebar.selectbox("Month", range(1, 13))

# Prepare input data
input_data = np.array([[latitude, longitude, depth, 0, 0, 0, 0, hour, day_of_week, month]])
risk = model.predict_proba(input_data)[0][1]

# Display prediction
st.write("### Risk Prediction")
st.write(f"**Probability of Significant Earthquake:** {risk:.2%}")

# Magnitude Distribution
st.write("### Magnitude Distribution")
fig, ax = plt.subplots()
ax.hist(data['magnitude'], bins=20, color='skyblue', edgecolor='black')
ax.set_title("Magnitude Distribution")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Frequency")
st.pyplot(fig)
