# app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="AI RideWise - Hourly Prediction", layout="wide")

# -------------------------------
# Force white background & black text
# -------------------------------
st.markdown("""
<style>
/* Main background & text */
.stApp {
    background-color: #ffffff !important;
    color: #0b0b0b !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
    color: #0b0b0b !important;
}

/* Predict button only */
.stButton button {
    background-color: #e63946 !important;  /* Red background */
    color: white !important;               /* White text */
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: bold;
}
.stButton button:hover {
    background-color: #d62839 !important; /* Darker red on hover */
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "models/tuned_xgb_hour_model.joblib"
try:
    model = joblib.load(MODEL_PATH, mmap_mode='r')
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -------------------------------
# Load Historical Data
# -------------------------------
HISTORICAL_PATH = "hour.csv"
hour_data = pd.read_csv(HISTORICAL_PATH)
hour_data['dteday'] = pd.to_datetime(hour_data['dteday'])
hourly_avg = hour_data.groupby('hr')['cnt'].mean()

# -------------------------------
# App Header
# -------------------------------
st.markdown("""
<h1 style='text-align:center; color:red; font-size:48px; font-weight:bold'>
🚴 AI RideWise — Predict Your Next Bike Ride!
</h1>
<p style='text-align:center; font-size:20px; color:#0b0b0b'>
Forecast hourly bike rentals based on <b>weather</b>, <b>time</b>, and <b>city patterns</b>.
</p>
<hr>
""", unsafe_allow_html=True)

# -------------------------------
# Inputs in 3 Columns
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    selected_date = st.date_input("Select Date", datetime.date(2012, 1, 1))
    hr = st.slider("Hour of Day", 0, 23, 12)
    weekday = selected_date.strftime("%A")
    season = st.selectbox("Season", ["Spring","Summer","Fall","Winter"])

with col2:
    holiday = st.selectbox("Holiday", ["No","Yes"])
    workingday = st.selectbox("Working Day", ["No","Yes"])
    weathersit = st.selectbox("Weather Situation", ["Clear/Cloudy", "Mist", "Light Rain/Snow", "Heavy Rain/Snow"])
    temp = st.slider("Temperature (0-1)", 0.0, 1.0, 0.5, 0.01)
    atemp = st.slider("Feels-like Temp (0-1)", 0.0, 1.0, 0.5, 0.01)

with col3:
    hum = st.slider("Humidity (0-1)", 0.0, 1.0, 0.5, 0.01)
    windspeed = st.slider("Windspeed (0-1)", 0.0, 0.3, 0.1, 0.01)

# -------------------------------
# Feature Engineering
# -------------------------------
weekday_num = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"].index(weekday)
holiday_num = 1 if holiday=="Yes" else 0
workingday_num = 1 if workingday=="Yes" else 0

sin_month = np.sin(2*np.pi*selected_date.month/12)
cos_month = np.cos(2*np.pi*selected_date.month/12)
sin_weekday = np.sin(2*np.pi*weekday_num/7)
cos_weekday = np.cos(2*np.pi*weekday_num/7)
sin_hour = np.sin(2*np.pi*hr/24)
cos_hour = np.cos(2*np.pi*hr/24)

trend = (selected_date - hour_data['dteday'].min().date()).days

season_map = {"Spring":0,"Summer":0,"Fall":0,"Winter":0}
season_map[season] = 1
season_2, season_3, season_4 = season_map["Summer"], season_map["Fall"], season_map["Winter"]

weathersit_map = {"Clear/Cloudy":0, "Mist":0, "Light Rain/Snow":0, "Heavy Rain/Snow":0}
weathersit_map[weathersit] = 1
weathersit_2, weathersit_3, weathersit_4 = weathersit_map["Mist"], weathersit_map["Light Rain/Snow"], weathersit_map["Heavy Rain/Snow"]

features = [
    selected_date.year, selected_date.month, holiday_num, weekday_num, workingday_num,
    temp, atemp, hum, windspeed,
    selected_date.year, selected_date.month, hr,
    sin_month, cos_month, sin_weekday, cos_weekday, sin_hour, cos_hour,
    trend,
    season_2, season_3, season_4,
    weathersit_2, weathersit_3, weathersit_4
]
features = np.array(features).reshape(1,-1)

# -------------------------------
# Predict Button & Display
# -------------------------------
st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
predict_clicked = st.button("Predict Hourly Ride Count")
st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    with st.spinner("Predicting..."):
        try:
            prediction = model.predict(features)
            predicted_count = int(prediction[0])

            st.markdown(f"""
            <div style='text-align:center; background-color:#f0f0f0; padding:20px; border-radius:15px; margin-top:20px'>
            <h2 style='font-size:64px; color:#0b0b0b'>{predicted_count}</h2>
            <p style='font-size:18px; color:#0b0b0b'>Predicted bike rentals for the selected hour</p>
            </div>
            """, unsafe_allow_html=True)

            col1_graph, col2_graph = st.columns(2)

            with col1_graph:
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(hourly_avg.index, hourly_avg.values, label="Historical Avg", marker='o')
                ax.scatter(hr, predicted_count, color='red', s=100, label="Predicted")
                ax.set_xlabel("Hour")
                ax.set_ylabel("Bike Rentals")
                ax.set_title("Predicted vs Historical Avg")
                ax.legend()
                st.pyplot(fig)

            with col2_graph:
                importances = model.feature_importances_
                feat_names = ["yr","mnth","holiday","weekday","workingday",
                              "temp","atemp","hum","windspeed",
                              "yr2","mnth2","hr2",
                              "sin_month","cos_month","sin_weekday","cos_weekday","sin_hour","cos_hour",
                              "trend",
                              "season_s","season_f","season_w",
                              "weathersit_mist","weathersit_lightsnow","weathersit_heavysnow"]
                sorted_idx = np.argsort(importances)
                fig, ax = plt.subplots(figsize=(7,5))
                ax.barh(np.array(feat_names)[sorted_idx], importances[sorted_idx], color="#0b6b3a")
                ax.set_title("Feature Importance")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
