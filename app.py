# app.py
import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bike Demand Prediction", layout="wide")

# ---------------- Load Model ----------------
def load_model_try(names):
    for name in names:
        if os.path.exists(name):
            try:
                mdl = joblib.load(name)
                return mdl, name, None
            except Exception as e:
                return None, name, f"Error loading {name}: {e}"
    return None, None, "No model file found. Tried: " + ", ".join(names)

possible_models = [
    "lgbm_best_model.pkl",
    "xgb_best_model.pkl",
    "xgb_bike_model.pkl",
    "model.pkl",
    "bike_demand_model.pkl"
]

model, used_name, load_err = load_model_try(possible_models)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    div.block-container {
        padding-top: 2rem !important;  /* fix heading cut */
        padding-bottom: 0rem;
    }
    .stSlider, .stSelectbox, .stRadio {
        margin-bottom: -15px;  /* compact spacing */
    }
    h1 {
        margin-top: 0.5rem;   /* fix title margin */
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<h1 style='text-align:center;color:#F5D5E0;margin-bottom:0;'>🚲 Bike Sharing Demand</h1>
<p style='text-align:center;color:gray;margin-top:-8px;margin-bottom:12px;'>
Quick forecast of bike rentals — set inputs & predict instantly
</p>
""", unsafe_allow_html=True)

# ---------------- Input Layout (3 columns) ----------------
col1, col2, col3 = st.columns(3)

with col1:
    season = st.selectbox("Season", ["Winter","Spring","Summer","Fall"])
    month_name = st.selectbox("Month", 
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], index=5)
    weekday = st.selectbox("Day of Week", 
        ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])
    hour = st.selectbox("Hour (24h)", list(range(0,24)), index=12)

with col2:
    workingday = st.radio("Working Day?", ["Yes","No"], horizontal=True)
    holiday = st.radio("Holiday?", ["Yes","No"], horizontal=True)
    weathersit = st.selectbox("Weather", ["Clear","Mist","Light Snow/Rain","Heavy Rain"])

with col3:
    temp = st.slider("Temperature (°C)", 0, 40, 20)
    atemp = st.slider("Feels-like Temp (°C)", 0, 45, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    windspeed = st.slider("Windspeed (km/h)", 0, 50, 10)

# ---------------- Preprocess Inputs ----------------
month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
month = month_map[month_name]
workingday_val = 1 if workingday=="Yes" else 0
holiday_val = 1 if holiday=="Yes" else 0

input_df = pd.DataFrame([{
    "yr":1,
    "holiday": holiday_val,
    "workingday": workingday_val,
    "temp": temp/40,
    "atemp": atemp/50,
    "hum": humidity/100,
    "windspeed": windspeed/67,
    "season": season,
    "mnth": month,
    "hr": hour,
    "weekday": weekday,
    "weathersit": weathersit
}])

# Extra engineered features
input_df["comfort_index"] = (temp/40 + atemp/50)/2 - (humidity/100)
input_df["temp_wind_interaction"] = (temp/40) * (windspeed/67)

# ---------------- Prediction ----------------
if model is None:
    st.error(f"Model not loaded. {load_err}")
    st.info("Place your model file in the same folder as app.py. Possible names: " + ", ".join(possible_models))
else:
    if st.button("🚀 Predict Demand", use_container_width=True):
        try:
            # expand categorical variables
            expanded = pd.get_dummies(
                input_df, 
                columns=["season","mnth","hr","weekday","weathersit"],
                drop_first=True
            )

            # get expected features from model
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
                expected_features = list(model.booster_.feature_name())
            else:
                expected_features = expanded.columns.tolist()

            # add missing cols
            for col in expected_features:
                if col not in expanded.columns:
                    expanded[col] = 0
            expanded = expanded[expected_features]

            # predict
            prediction = model.predict(expanded)[0]

            # classify demand (Low, Medium, High)
            if prediction < 130:
                demand_status = "🔴 Low Demand"
                color = "#FF5252"  # red
            elif 130 <= prediction < 150:
                demand_status = "🟡 Medium Demand"
                color = "#FFD700"  # yellow/golden
            else:
                demand_status = "🟢 High Demand"
                color = "#2E7D32"  # green

            # display result
            st.markdown(
                f"<div style='padding:12px;border-radius:10px;background:{color};color:white;font-weight:bold;text-align:center;'>"
                f"{demand_status}<br><span style='font-size:22px'>Predicted rentals: {int(prediction)}</span>"
                "</div>", 
                unsafe_allow_html=True
            )

            # extra friendly note
            st.success("✅ Prediction generated successfully! Adjust inputs above to explore different scenarios.")

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

# ---------------- Footer ----------------
st.markdown("<hr style='margin-top:5px;margin-bottom:5px;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:gray;font-size:13px;'>"
    "🚲 Bike Demand Prediction Project | Developed by <b>Banothu Anusha</b> | Powered by Machine Learning"
    "</div>", 
    unsafe_allow_html=True
)