import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D

# Load dataset (use your dataset here)
@st.cache_data
def load_data():
    data = pd.read_csv('dataset/earthquake_data.csv')  # Replace with your dataset
    return data

# Preprocess data
def preprocess_data(data):
    # Convert date_time to datetime
    data['date_time'] = pd.to_datetime(data['date_time'], format='%d-%m-%Y %H:%M')
    
    # Sort data by date_time
    data = data.sort_values('date_time')
    
    # Compute time difference (in seconds) between events
    data['time_since_last_event'] = data['date_time'].diff().dt.total_seconds().fillna(0)
    
    # Add feature for previous location differences
    data['latitude_diff'] = data['latitude'].diff().fillna(0)
    data['longitude_diff'] = data['longitude'].diff().fillna(0)
    data['depth_diff'] = data['depth'].diff().fillna(0)
    
    # Extract datetime features
    data['year'] = data['date_time'].dt.year
    data['month'] = data['date_time'].dt.month
    data['day'] = data['date_time'].dt.day
    data['hour'] = data['date_time'].dt.hour
    data['minute'] = data['date_time'].dt.minute
    
    # Drop original datetime column
    data = data.drop(columns=['date_time'])
    
    # Define features and targets
    X = data[['latitude', 'longitude', 'depth', 'time_since_last_event',
              'latitude_diff', 'longitude_diff', 'depth_diff',
              'year', 'month', 'day', 'hour', 'minute']]
    y_time = data['time_since_last_event']  # Target for time prediction
    y_lat = data['latitude']
    y_long = data['longitude']
    y_depth = data['depth']
    y_mag = data['magnitude']  # Target for magnitude prediction
    
    return train_test_split(X, y_time, y_lat, y_long, y_depth, y_mag, test_size=0.2, random_state=42)

# Train models
@st.cache_resource
def train_models(X_train, y_train_time, y_train_lat, y_train_long, y_train_depth, y_train_mag):
    # Time model
    model_time = RandomForestRegressor()
    model_time.fit(X_train, y_train_time)
    
    # Latitude model
    model_lat = RandomForestRegressor()
    model_lat.fit(X_train, y_train_lat)
    
    # Longitude model
    model_long = RandomForestRegressor()
    model_long.fit(X_train, y_train_long)
    
    # Depth model
    model_depth = RandomForestRegressor()
    model_depth.fit(X_train, y_train_depth)
    
    # Magnitude model
    model_mag = RandomForestRegressor()
    model_mag.fit(X_train, y_train_mag)
    
    return model_time, model_lat, model_long, model_depth, model_mag

# Streamlit App
st.title("QuakeSense: Seismic Event Prediction")
st.markdown("An ML-powered tool to predict seismic events.")

# Load and display data
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    st.write("Using default dataset...")
    data = load_data()

st.write("### Dataset Preview")
st.write(data.head())

# Preprocess and train
X_train, X_test, y_train_time, y_test_time, y_train_lat, y_test_lat, y_train_long, y_test_long, y_train_depth, y_test_depth, y_train_mag, y_test_mag = preprocess_data(data)
model_time, model_lat, model_long, model_depth, model_mag = train_models(
    X_train, y_train_time, y_train_lat, y_train_long, y_train_depth, y_train_mag
)

# Predictions
st.sidebar.header("Input Parameters")
latitude = st.sidebar.number_input("Latitude", value=35.0, step=0.1)
longitude = st.sidebar.number_input("Longitude", value=-120.0, step=0.1)
depth = st.sidebar.number_input("Depth (km)", value=10.0, step=1.0)
time = st.sidebar.number_input("Time (in seconds since epoch)", value=1633072800)

# Prepare input data
input_data = np.array([[latitude, longitude, depth, 0, 0, 0, 0, 2024, 11, 29, 12, 0]])
time_pred = model_time.predict(input_data)[0]
lat_pred = model_lat.predict(input_data)[0]
long_pred = model_long.predict(input_data)[0]
depth_pred = model_depth.predict(input_data)[0]
mag_pred = model_mag.predict(input_data)[0]

# Calculate predicted date
# Convert the current epoch time to a datetime object
current_time = datetime.fromtimestamp(time)

# Add the predicted time (in seconds) as a timedelta
predicted_date = current_time + timedelta(seconds=time_pred)

# Get today's date (without time)
today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# Display predictions only if the predicted date is today or in the future
if predicted_date >= today:
    st.write("### Prediction Results")
    st.write(f"Predicted Date: **{predicted_date.strftime('%Y-%m-%d %H:%M')}**")
    st.write(f"Predicted Location: **Lat {lat_pred:.2f}, Long {long_pred:.2f}, Depth {depth_pred:.2f} km**")
    st.write(f"Predicted Magnitude: **{mag_pred:.2f}**")
else:
    st.write("### Prediction Results")
    st.warning("No future events predicted.")

# Visualizations
st.write("### Data Visualization")
st.map(data[['latitude', 'longitude']])

# Historical Analysis
st.write("### Earthquake History")
fig, ax = plt.subplots()
scatter = ax.scatter(data['longitude'], data['latitude'], c=data['magnitude'], cmap='viridis', alpha=0.7)
ax.scatter(long_pred, lat_pred, c='red', marker='X', s=200, label="Predicted Event")
ax.set_title("Earthquake Locations")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
st.pyplot(fig)

# Heatmap visualization
st.write("### Heatmap of Earthquake Magnitudes")
map_data = data[['latitude', 'longitude', 'magnitude']].dropna()
m = folium.Map(location=[map_data['latitude'].mean(), map_data['longitude'].mean()], zoom_start=5)

# Add heatmap

HeatMap(data=map_data[['latitude', 'longitude', 'magnitude']].values, radius=10).add_to(m)

st_folium(m, width=700, height=500)

# Magnitude Distribution
st.write("### Magnitude Distribution")
fig, ax = plt.subplots()
ax.hist(data['magnitude'], bins=20, color='skyblue', edgecolor='black')
ax.set_title("Magnitude Distribution")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Clustering
st.write("### Clustering of Earthquake Locations")
coords = data[['latitude', 'longitude']].dropna()
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(coords)

# Visualize clusters
fig, ax = plt.subplots()
scatter = ax.scatter(data['longitude'], data['latitude'], c=data['cluster'], cmap='tab10', alpha=0.7)
ax.set_title("Earthquake Location Clusters")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
st.pyplot(fig)


# 3D Visualization
st.write("### 3D Visualization of Earthquake Data")
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['longitude'], data['latitude'], data['depth'], c=data['magnitude'], cmap='viridis')
ax.set_title("3D Earthquake Visualization")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Depth")
fig.colorbar(scatter, ax=ax, label="Magnitude")
st.pyplot(fig)
