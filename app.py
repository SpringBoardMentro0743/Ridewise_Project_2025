import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------
# Load Model & Dataset
# -------------------
gbr_hour = joblib.load("final_gbr_hour.pkl")
df_historical = pd.read_csv("hour.csv")

# -------------------
# Page Config
# -------------------
st.set_page_config(page_title="Bike Rental Prediction", page_icon="🚲", layout="wide")

# -------------------
# Custom CSS - White Form with Light Blue Background
# -------------------
st.markdown("""
    <style>
    /* Main background - Light blue */
    .stApp {
        background-color: #e6f2ff;
    }
    
    /* White container for the main form */
    .main-form-container {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 20px auto;
        max-width: 900px;
        border: 2px solid #d4e6ff;
    }
    
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .section-header {
        font-size: 20px;
        color: #2c5aa0;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e6f2ff;
        font-weight: 600;
    }
    
    .prediction-card {
        background-color: #f0f8ff;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #2E8B57;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-title {
        font-size: 24px;
        font-weight: bold;
        color: #2E8B57;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .prediction-time {
        font-size: 18px;
        color: #333333;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .prediction-value {
        font-size: 42px;
        font-weight: bold;
        color: #2c5aa0;
        text-align: center;
        margin: 15px 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    .prediction-unit {
        font-size: 20px;
        color: #333333;
        text-align: center;
    }
    
    .input-section {
        background-color: #f8fbff;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 1px solid #e6f2ff;
    }
    
    .stButton>button {
        background-color: #2c5aa0;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        border: none;
        width: 100%;
        margin-top: 15px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #2E8B57;
        color: white;
    }
    
    /* Radio button styling fix */
    .stRadio > div {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
    }
    
    .stRadio label {
        color: #333333 !important;
        font-weight: 600;
    }

    /* Fix general label colors */
    .stSelectbox label, .stSlider label, .stDateInput label {
        color: #333333 !important;
        font-weight: 600;
    }

    /* Ensure all text in white form is dark */
    .stMarkdown, .stText, .stWrite {
        color: #333333 !important;
    }

    /* Fix dataframe text color */
    .stDataFrame {
        color: #333333 !important;
    }

    /* Specific fix for radio button text */
    div[data-testid="stRadio"] label {
        color: #333333 !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------
# Main White Form Container
# -------------------
st.markdown("<div class='main-form-container'>", unsafe_allow_html=True)

# -------------------
# Header
# -------------------
st.markdown("<div class='main-header'>🚲 Bike Rental Predictor</div>", unsafe_allow_html=True)
st.markdown("<span style='color: #333333;'>Predict how many bikes will be rented based on weather and time conditions</span>", unsafe_allow_html=True)

# -------------------
# Input Section
# -------------------
st.markdown("<div class='section-header'>📅 Select Date & Time</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("*Date*", value=datetime.now())
with col2:
    hours = [f"{h:02d}:00" for h in range(24)]
    selected_time_str = st.selectbox("*Time*", options=hours, index=12)
    selected_time = datetime.strptime(selected_time_str, "%H:%M").time()

# Date & Time confirmation (unchanged and working)
st.markdown(
    f"<span style='color: #333333; font-weight: 600;'>Selected: {selected_date.strftime('%b %d, %Y')} at {selected_time.strftime('%I:%M %p')}</span>",
    unsafe_allow_html=True
)

# -------------------
# Weather Conditions
# -------------------
st.markdown("<div class='section-header'>🌤 Weather Conditions</div>", unsafe_allow_html=True)
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    temp_c = st.slider("*Temperature (°C)*", -5.0, 40.0, 20.0, 1.0)
    humidity_pct = st.slider("*Humidity (%)*", 0, 100, 50)
with col2:
    feels_c = st.slider("*Feels Like (°C)*", -5.0, 40.0, 22.0, 1.0)
    windspeed_kmh = st.slider("*Wind Speed (km/h)*", 0, 50, 10)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------
# Other Conditions
# -------------------
st.markdown("<div class='section-header'>🏙 Other Conditions</div>", unsafe_allow_html=True)
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    season = st.selectbox("*Season*", ["Spring", "Summer", "Fall", "Winter"])
with col2:
    weather = st.selectbox("*Weather*", ["Clear", "Cloudy", "Light Rain", "Heavy Rain"])
with col3:
    holiday = st.selectbox("*Holiday*", ["No", "Yes"])

# Working Day section — now properly visible and bold
st.markdown("*Working Day*", unsafe_allow_html=True)
workingday = st.radio(" ", ["No", "Yes"], horizontal=True, label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------
# Prediction Button
# -------------------
if st.button("🔮 Predict Bike Rentals", use_container_width=True):
    # Normalize inputs
    season_map = {"Spring":1, "Summer":2, "Fall":3, "Winter":4}
    weather_map = {"Clear":1, "Cloudy":2, "Light Rain":3, "Heavy Rain":4}
    
    temp = temp_c / 50.0
    atemp = feels_c / 50.0
    hum = humidity_pct / 100.0
    windspeed = windspeed_kmh / 67.0
    holiday_val = 1 if holiday == "Yes" else 0
    workingday_val = 1 if workingday == "Yes" else 0

    # Single prediction
    hour = selected_time.hour
    data = pd.DataFrame([{
        "season": season_map[season],
        "hr": hour,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "holiday": holiday_val,
        "workingday": workingday_val,
        "weathersit": weather_map[weather]
    }])

    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=gbr_hour.feature_names_in_, fill_value=0)
    prediction = gbr_hour.predict(data_encoded)[0]

    # Display prediction
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-title'>📊 PREDICTION RESULT</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='prediction-time'>Selected Time: {selected_time.strftime('%I:%M %p')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='prediction-value'>{int(prediction)}</div>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-unit'>BIKES</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------
    # 24-Hour Forecast
    # -------------------
    st.markdown("<div class='section-header'>📈 24-Hour Forecast</div>", unsafe_allow_html=True)
    
    hourly_preds = []
    for h in range(24):
        if h in df_historical['hr'].values:
            temp_h = df_historical.loc[df_historical['hr']==h, 'temp'].mean()
            atemp_h = df_historical.loc[df_historical['hr']==h, 'atemp'].mean()
            hum_h = df_historical.loc[df_historical['hr']==h, 'hum'].mean()
            wind_h = df_historical.loc[df_historical['hr']==h, 'windspeed'].mean()
        else:
            temp_h, atemp_h, hum_h, wind_h = temp, atemp, hum, windspeed

        row = pd.DataFrame([{
            "season": season_map[season],
            "hr": h,
            "temp": temp_h,
            "atemp": atemp_h,
            "hum": hum_h,
            "windspeed": wind_h,
            "holiday": holiday_val,
            "workingday": workingday_val,
            "weathersit": weather_map[weather]
        }])
        row_encoded = pd.get_dummies(row)
        row_encoded = row_encoded.reindex(columns=gbr_hour.feature_names_in_, fill_value=0)
        hourly_preds.append(gbr_hour.predict(row_encoded)[0])

    # Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    hours_list = list(range(24))
    max_rentals = max(hourly_preds)
    colors = ['#ff6b6b' if x == max_rentals else '#4ecdc4' for x in hourly_preds]
    bars = ax.bar([f"{h:02d}:00" for h in hours_list], hourly_preds, color=colors, alpha=0.8)
    ax.axvline(x=hour, color='red', linestyle='--', linewidth=2, label='Selected Time')
    ax.set_title("24-Hour Bike Rental Predictions", fontsize=16, fontweight='bold', color='#333333')
    ax.set_xlabel("Hour of Day", fontsize=12, color='#333333')
    ax.set_ylabel("Predicted Rentals", fontsize=12, color='#333333')
    ax.tick_params(axis='x', rotation=45, colors='#333333')
    ax.tick_params(axis='y', colors='#333333')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2., height + max_rentals*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, color='#333333')
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    st.pyplot(fig)

    # Table
    df_preds = pd.DataFrame({
        "Hour": list(range(24)),
        "Predicted_Rentals": [int(p) for p in hourly_preds],
        "Hour_Label": [f"{h:02d}:00" for h in range(24)]
    })
    styled_df = df_preds[['Hour_Label', 'Predicted_Rentals']].rename(
        columns={'Hour_Label': 'Time', 'Predicted_Rentals': 'Rentals'}
    ).style.background_gradient(subset=['Rentals'], cmap='YlGnBu')
    st.markdown("<span style='color: #333333;'>Hourly Rental Data</span>", unsafe_allow_html=True)
    st.dataframe(styled_df, use_container_width=True, height=400)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "Bike Rental Prediction Tool • Simple and Easy to Use"
    "</div>", unsafe_allow_html=True
)