import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import os

# --- 1. SETTINGS AND COLORS (Simple & Clean) ---
BLUE_ACCENT = "#1E90FF"  # Dodger Blue (Clean, bright blue)
DARK_BG = "#121212"     # Very dark background
RESULT_COLOR = "#00FF7F" # Spring Green (Bright, positive result color)
FONT_FAMILY = "Arial, sans-serif"

# NOTE: The reliable bike image link is retained.
BACKGROUND_URL = "https://images.unsplash.com/photo-1558981403-c5f9899a280b?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
MODEL_FILE = "ridewise_best_model.pkl"

# --- MAPPING DICTIONARIES (kept same values) ---
SEASON_DICT = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
WEATHER_DICT = {"Clear / Sunny": 1, "Mist / Cloudy": 2, "Light Rain / Snow": 3, "Heavy Rain / Ice": 4}
EVENT_DICT = {"No": 0, "Yes": 1}


# ----------------------------------------
# 2. PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="Bike Demand Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# 3. CUSTOM CSS (Simple & Professional Dark Mode)
# ----------------------------------------
simple_css = f"""
<style>
/* 1. Global Background and Font */
.stApp {{
    background-image: url("{BACKGROUND_URL}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}}
.stApp::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.9); /* Dark overlay */
    z-index: -1;
}}
* {{
    font-family: {FONT_FAMILY} !important;
}}

/* 2. Headers and General Text */
h1, h2, h3, h4, label, .stMarkdown {{
    color: #FFFFFF !important; /* White text for clarity */
}}
.stMarkdown h3 {{
    color: {BLUE_ACCENT} !important; /* Blue for section titles */
}}

/* 3. Sidebar */
[data-testid="stSidebar"] {{
    background-color: {DARK_BG};
    border-right: 3px solid {BLUE_ACCENT};
}}
[data-testid="stSidebar"] .stMarkdown h1 {{
    color: {BLUE_ACCENT} !important;
    font-size: 2.2rem;
    padding-top: 20px;
    padding-bottom: 5px;
}}
[data-testid="stSidebar"] .stMarkdown p {{
    color: #999999;
    font-size: 0.9rem;
}}

/* 4. Input Containers (Clean Cards) */
div[data-testid="stContainer"] {{
    background-color: rgba(255, 255, 255, 0.05); /* Very light, transparent card */
    border-radius: 10px;
    padding: 25px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle white border */
}}
.stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox div[data-testid="stSelectboxContainer"] div[data-testid="stSelectboxValue"] {{
    color: #FFFFFF !important;
    background: rgba(0, 0, 0, 0.2) !important;
    border-radius: 5px;
}}

/* 5. Button (High Contrast) */
.stButton>button {{
    display: block;
    width: 100%;
    padding: 15px;
    background-color: {BLUE_ACCENT};
    color: white;
    border-radius: 8px;
    font-size: 1.2rem;
    font-weight: bold;
    transition: all 0.2s ease;
    border: 2px solid {BLUE_ACCENT};
}}
.stButton>button:hover {{
    background-color: {DARK_BG};
    color: {BLUE_ACCENT};
    border: 2px solid {BLUE_ACCENT};
}}

/* 6. Result Box (Highlight) */
.result-box {{
    background-color: rgba(0, 255, 127, 0.1); /* Light green transparent background */
    border-radius: 12px;
    padding: 30px;
    border: 3px solid {RESULT_COLOR};
    text-align: center;
}}
.result-box h3 {{
    color: #FFFFFF !important;
    font-size: 1.5rem !important;
}}
.result-value {{
    color: {RESULT_COLOR};
    font-size: 4.0rem; /* Big number */
    font-weight: 900;
    display: block;
    line-height: 1.1;
}}

/* 7. Info Box (Feedback) */
.info-box {{
    background: rgba(255, 255, 255, 0.08);
    border-left: 5px solid {RESULT_COLOR};
    padding: 20px;
    margin-top: 15px;
    border-radius: 8px;
    color: #F0F0F0;
    line-height: 1.6;
}}
.info-box strong {{
    color: {BLUE_ACCENT};
    font-size: 1.05rem;
    display: block;
    padding-bottom: 5px;
}}
</style>
"""
st.markdown(simple_css, unsafe_allow_html=True)

# ----------------------------------------
# 4. SIDEBAR (Title)
# ----------------------------------------
with st.sidebar:
    st.markdown(f"<h1>RideWise</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>Bike Demand Predictor</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Status")
    if os.path.exists(MODEL_FILE):
        st.success(f"Prediction Model: **READY**")
    else:
        st.error(f"Prediction Model: **MISSING**")
    
    st.markdown("---")
    st.markdown("This tool predicts how many bikes people are likely to rent based on weather and time, helping optimize availability and improve service efficiency.")

# ----------------------------------------
# 5. MAIN CONTENT (Header)
# ----------------------------------------
st.markdown('<h1>Bike Rental Forecast Based on Weather and Urban Events</h1>', unsafe_allow_html=True)
st.markdown('<h3>Enter the details below to see the expected bike demands.</h3>', unsafe_allow_html=True)

# ----------------------------------------
# 6. MODEL LOADING
# ----------------------------------------
try:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError()
        
    @st.cache_resource
    def load_model(file_path):
        return joblib.load(file_path)

    model = load_model(MODEL_FILE)
    
except FileNotFoundError:
    st.stop() # Stops execution if model is not found, relying on sidebar error.
except Exception as e:
    st.error(f"❌ Error loading the model file. Please check the file.")
    st.stop()


# ----------------------------------------
# 7. USER INPUTS
# ----------------------------------------

st.markdown("### When and Where")
with st.container(border=False):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        date_input = st.date_input("📅 Choose Date", datetime.today(), 
                                   help="The day you want to predict.")
        
    with col2:
        hour = st.slider("🕒 Choose Hour (0-23)", 0, 23, 17, 
                         help="The specific hour of the day (e.g., 17 is 5 PM).")
    
    with col3:
        season = st.selectbox("🌦️ What Season is it?", list(SEASON_DICT.keys()), key='season_in',
                              help="Season helps predict general demand level.")
        season_value = SEASON_DICT[season]
        
    with col4:
        weathersit = st.selectbox("⛈️ How is the Weather?", list(WEATHER_DICT.keys()), key='weather_in',
                                  help="Weather condition is the biggest factor.")
        weathersit_value = WEATHER_DICT[weathersit]

st.markdown("### More Details")
with st.container(border=False):
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        temp = st.number_input("🌡️ Temperature (0=Cold, 1=Hot)", min_value=0.0, max_value=1.0, step=0.01, value=0.6, key='temp_in',
                               help="Input the temperature, scaled from 0 (very cold) to 1 (very hot).")
        
    with col_b:
        humidity = st.number_input("💧 Humidity (0=Dry, 1=Wet)", min_value=0.0, max_value=1.0, step=0.01, value=0.45, key='hum_in',
                                   help="Input the air humidity, scaled from 0 to 1.")
    
    with col_c:
        event = st.selectbox("🎉 Is there a Urban Event?", list(EVENT_DICT.keys()), key='event_in',
                             help="Select 'Yes' if there is a major event today.", index=0)
        event_value = EVENT_DICT[event]

st.markdown("---")

# ----------------------------------------
# 8. PREDICTION AND RESULT DISPLAY
# ----------------------------------------
if st.button(" Check Demand Now", key="predict"):
    
    # 1. Prepare Input Data
    input_data = pd.DataFrame([{
        'season': season_value,
        'weathersit': weathersit_value,
        'mnth': date_input.month,
        'hr': hour,
        'temp': temp,
        'hum': humidity,
        'event': event_value
    }])

    # 2. OHE and Feature Reindexing (Model Compatibility)
    input_data = pd.get_dummies(input_data, columns=['hr', 'mnth', 'season', 'weathersit', 'event'])
    
    try:
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    except AttributeError:
        st.error("❌ Model feature error. Cannot predict.")
        st.stop()

    # 3. Predict and Format Result
    prediction = model.predict(input_data)
    predicted_rentals = int(max(0, prediction[0]))

    
    col_result, col_insights = st.columns([1, 2])

    with col_result:
        st.markdown(f"""
            <div class="result-box">
                <p style="color: white; margin: 0; font-size: 1.1rem; font-weight: 500;">BIKES NEEDED</p>
                <span class="result-value">{predicted_rentals}</span>
                <p style="color: {BLUE_ACCENT}; margin: 0; font-size: 1.1rem;">Total for this hour</p>
            </div>
        """, unsafe_allow_html=True)
    
    # ----------------------------------------
    # Insights (Simple Text)
    # ----------------------------------------
    with col_insights:
        suggestions = []

        if predicted_rentals < 20:
            suggestions.append("⚠️ **Low Demand:** You won't need many bikes. **Action:** Focus on cleaning or repairs.")
            if weathersit_value >= 3:
                suggestions.append("   - Main Reason: **Bad weather** (rain/snow).")
            if temp < 0.3 or temp > 0.8:
                suggestions.append("   - Also: **Extreme temperature** (too hot or too cold).")
        elif predicted_rentals > 150:
            suggestions.append("🔥 **High Demand!** You need all your bikes ready. **Action:** Move bikes to busy spots fast.")
            if event_value == 1:
                suggestions.append("   - Main Reason: **A big event** is happening.")
            if weathersit_value == 1 and 7 <= hour <= 19:
                suggestions.append("   - Also: **Perfect weather** during peak hours.")
        else:
            suggestions.append("✅ **Normal Demand:** Keep things running as usual. **Action:** Check bike supply often.")


        st.markdown(f"""
            <div class="info-box">
                <strong>SYSTEM ADVICE:</strong><br>
                {"<br>".join(suggestions)}
            </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------
    # Input Summary
    # ----------------------------------------
    st.markdown(f"""
        <div class="info-box" style="border-left: 5px solid {BLUE_ACCENT};">
            <strong>YOUR INPUTS:</strong><br>
            <span style="color: #999999;">TIME:</span> {date_input.strftime('%Y-%m-%d')} @ **{hour}:00**<br>
            <span style="color: #999999;">WEATHER:</span> **{weathersit}** | Temp: **{temp:.2f}** | Event: **{event}**
        </div>
    """, unsafe_allow_html=True)