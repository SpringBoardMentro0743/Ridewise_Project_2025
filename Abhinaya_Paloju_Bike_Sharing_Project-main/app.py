import streamlit as st
from xgboost import XGBRegressor
import json
import pandas as pd
import numpy as np
import datetime
# ---------------------------
# Load trained model + features
# ---------------------------
FEATURES_PATH = "artifacts/features.json"
model = XGBRegressor()
model.load_model("artifacts/model.json")

with open(FEATURES_PATH, "r") as f:
    features = json.load(f)["features"]

st.set_page_config(page_title="Bike Sharing Predictor 🚲", page_icon="🚲", layout="centered")

# ---------------------------
# Header
# ---------------------------
st.markdown(
    "<h1 style='white-space: nowrap; text-align:center; color:black;'>🚲 Bike Sharing Demand Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='color:#ff4b4b; text-align:center; padding:10px;'>🎯 <strong>ABHINAYA PALOJU</strong> 🎯</h2>",
    unsafe_allow_html=True
)

st.markdown("### 🌦️ Environment Settings")

# ---------------------------
# Location Input (Indian States)
# ---------------------------
india_states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", 
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

state = st.selectbox("Select State", india_states)

# Map state to numeric for model
state_map = {st_name: i+1 for i, st_name in enumerate(india_states)}
state_encoded = state_map[state]

# ---------------------------
# User Inputs
# ---------------------------
season = st.selectbox("Season", ["🌸 Spring", "☀️ Summer", "🍂 Fall", "❄️ Winter"])
season_map = {"🌸 Spring": 1, "☀️ Summer": 2, "🍂 Fall": 3, "❄️ Winter": 4}

weather = st.selectbox(
    "Weather",
    ["☀️ Clear / Few clouds", "🌫️ Mist / Cloudy", "🌧️ Light Snow / Rain", "⛈️ Heavy Rain / Thunderstorm"]
)
weather_map = {
    "☀️ Clear / Few clouds": 1,
    "🌫️ Mist / Cloudy": 2,
    "🌧️ Light Snow / Rain": 3,
    "⛈️ Heavy Rain / Thunderstorm": 4
}

col1, col2 = st.columns(2)
with col1:
    temp_c = st.slider("Temperature (°C)", -5.0, 40.0, 20.0)
    temp = temp_c / 41.0  # normalized
with col2:
    hum_pct = st.slider("Humidity (%)", 0, 100, 55)
    hum = hum_pct / 100.0  # normalized

windspeed_kmh = st.slider("Windspeed (km/h)", 0, 50, 10)
windspeed = windspeed_kmh / 67.0  # normalized

st.markdown("### ⏰ Time Settings")
hour = st.slider("Hour of Day", 0, 23, 14)


selected_date = st.date_input(
    "Select Date",
    datetime.date.today()  # default to today
)

st.write(f"Selected Date: {selected_date}")


weekday = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
weekday_map = {day: i for i, day in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}

month_options = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
month = st.selectbox("Month", month_options)
month_map = {name: i+1 for i, name in enumerate(month_options)}

is_weekend = 1 if weekday in ["Saturday", "Sunday"] else 0

# ---------------------------
# Feature Engineering
# ---------------------------
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
dow_sin = np.sin(2 * np.pi * weekday_map[weekday] / 7)
dow_cos = np.cos(2 * np.pi * weekday_map[weekday] / 7)
month_sin = np.sin(2 * np.pi * month_map[month] / 12)
month_cos = np.cos(2 * np.pi * month_map[month] / 12)

# ---------------------------
# Historical / Lag Features (optional)
# ---------------------------
with st.expander("📊 Historical Bike Data (optional)"):
    cnt_lag1 = st.number_input("Previous Hour Count", value=200)
    cnt_lag24 = st.number_input("Previous Day Count", value=250)
    cnt_roll3 = st.number_input("Rolling 3-hr Avg", value=230)
    cnt_roll6 = st.number_input("Rolling 6-hr Avg", value=220)
    cnt_roll12 = st.number_input("Rolling 12-hr Avg", value=210)

# Interaction features
hour_temp = hour * temp
hour_humidity = hour * hum
temp_humidity = temp * hum

# ---------------------------
# Prepare Data
# ---------------------------
input_data = {
    "season": season_map[season],
    "weathersit": weather_map[weather],
    "temp": temp,
    "hum": hum,
    "windspeed": windspeed,
    "hour": hour,
    "day_of_week": weekday_map[weekday],
    "month": month_map[month],
    "is_weekend": is_weekend,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "dow_sin": dow_sin,
    "dow_cos": dow_cos,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "cnt_lag1": cnt_lag1,
    "cnt_lag24": cnt_lag24,
    "cnt_roll3": cnt_roll3,
    "cnt_roll6": cnt_roll6,
    "cnt_roll12": cnt_roll12,
    "hour_temp": hour_temp,
    "hour_humidity": hour_humidity,
    "temp_humidity": temp_humidity,
    "state": state_encoded  # Added Indian state feature
}

df = pd.DataFrame([input_data]).reindex(columns=features)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔮 Predict Bike Demand"):
    with st.spinner("Predicting..."):
        prediction = model.predict(df)[0]
        st.success(f"🚴 Predicted Bike Count: **{prediction:.0f}**")
        st.progress(min(1.0, prediction / 1000))
