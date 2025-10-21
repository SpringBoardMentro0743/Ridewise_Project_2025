import os, sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Make utils importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import utils

# Paths
MODEL_PATH = os.path.join("models", "bike_model.pkl")
COLS_PATH = os.path.join("models", "feature_columns.pkl")

@st.cache_resource
def load_model_and_cols():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLS_PATH)
    return model, feature_columns

model, feature_columns = load_model_and_cols()

def predict_bikes(sample_input):
    X_enc = utils.prepare_input_df(sample_input, feature_columns)
    pred = model.predict(X_enc)[0]
    return max(0, int(round(pred)))

# -----------------------------
# UI Config
# -----------------------------
st.set_page_config(
    page_title="Bike Rental Predictor", 
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def apply_dark_theme_styles():
    """
    Apply hardcoded Dark Mode styles with increased spacing for headings.
    Crucially, it fixes the button text color to ensure contrast.
    """
    
    
    # --- Simplified Slider CSS ---
    simplified_slider_css = """

     /* --- Sidebar Fixes --- */
        /* Ensure the sidebar matches the main app background */
        .stSidebar {
            background-color: var(--bg-color) !important;
            border-right: 2px solid var(--border-color) !important;
        }
        /* Style the sidebar title/header */
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        .sidebar-subheader {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--success-color);
            margin-top: 1rem;
        }
        /* Simplifies and standardizes the slider for both modes */
        .stSlider > div[data-baseweb="slider"] > div {
            background: var(--border-color) !important;
            height: 8px !important;
            border-radius: 4px !important;
            box-shadow: none !important;
        }
        
        /* The filled portion of the track */
        .stSlider > div[data-baseweb="slider"] > div > div:nth-child(1) {
            background: var(--primary-color) !important;
            border-radius: 4px !important;
        }

        /* Slider Thumb (The circle handle) */
        .stSlider > div[data-baseweb="slider"] div[role="slider"] {
            background: var(--primary-color) !important;
            border: 3px solid var(--bg-color) !important;
            box-shadow: 0 2px 8px var(--shadow) !important;
            width: 20px !important;
            height: 20px !important;
            transition: all 0.2s ease !important;
            border-radius: 50% !important;
        }
        
        /* Remove all hover and active effects for a simpler UI */
        .stSlider > div[data-baseweb="slider"] div[role="slider"]:hover,
        .stSlider > div[data-baseweb="slider"] div[role="slider"]:active {
            transform: none !important;
            box-shadow: 0 2px 8px var(--shadow) !important;
            background: var(--primary-color) !important;
            border: 3px solid var(--bg-color) !important;
        }
    """
    
    st.markdown(
        f"""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Force override Streamlit's default theme (Dark Mode) */
        .stApp {{
            background-color: #0f172a !important;
            color: #f9fafb !important;
        }}
        
        /* Root variables for dark theme */
        :root {{
            --primary-color: #60a5fa;
            --secondary-color: #818cf8;
            --success-color: #34d399;
            --warning-color: #fbbf24;
            --danger-color: #f87171;
            --text-color: #f9fafb;
            --text-muted: #9ca3af;
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --hover-bg: #475569;
            --shadow: rgba(0, 0, 0, 0.4);
        }}
        
        /* Force override all text colors */
        .main > div, .main * {{
            font-family: 'Inter', sans-serif !important;
            color: var(--text-color) !important;
            background-color: transparent;
        }}
        
        .main {{
            background-color: var(--bg-color) !important;
        }}
        
        /* Override Streamlit's text elements */
        .stMarkdown, .stText, p, span, div, label {{
            color: var(--text-color) !important;
        }}
        
        /* Header styling (kept) */
        .main-header {{
            text-align: center;
            padding: 2.5rem 0;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px var(--shadow);
        }}
        
        .main-header h1 {{
            font-size: 4rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }}
        
        .main-header p {{
            font-size: 1rem;
            margin-top: 0.5rem;
            opacity: 0.95;
            font-weight: 400;
        }}
        
        /* Form container (kept) */
        .form-container {{
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 4px 20px var(--shadow);
            transition: all 0.3s ease;
        }}
        
        /* Section headers - Increased bottom margin for more space (kept) */
        .section-header {{
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 2rem; 
            margin-top: 2rem; 
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .form-container .section-header:first-of-type {{
            margin-top: 0rem;
        }}

        /* Input styling (kept) */
        # .stSelectbox > div > div {{
        #     background-color: var(--card-bg) !important;
        #     border: 2px solid var(--border-color) !important;
        #     border-radius: 12px !important;
        #     color: var(--text-color) !important;
        #     box-shadow: 0 4px 12px var(--shadow) !important;
        #     transition: all 0.3s ease !important;
        # }}
         /* --- FIX: Selectbox and Dropdown Visibility --- */
        
        /* 1. Targets the container that holds the selected value */
        .stSelectbox > div[data-baseweb="select"] > div:first-child {{
            background-color: var(--card-bg) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 12px !important;
            color: var(--text-color) !important; 
        }}
        
        /* 2. Targets the text inside the selected value box */
        .stSelectbox > div[data-baseweb="select"] span {{
            color: var(--text-color) !important;
        }}

        /* 3. Targets the actual dropdown list container (the POPUP/POPOVER when opened) */
        div[data-baseweb="popover"] > div > ul {{
            background-color: var(--card-bg) !important; /* Force dark background */
            border: 1px solid var(--border-color) !important;
        }}
        
        /* 4. Targets individual items in the dropdown list to ensure their text is light */
        div[data-baseweb="popover"] > div > ul > li,
        div[data-baseweb="popover"] div[role="option"] span {{
            color: var(--text-color) !important; /* Force light text */
            background-color: transparent !important;
        }}
        
        /* 5. Fix for the hover state color in the dropdown list */
        div[data-baseweb="popover"] li:hover {{
            background-color: var(--hover-bg) !important;
        }}
        
        /* Radio button styling (kept) */
        .stRadio > div {{
            background: var(--card-bg) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 15px !important;
            padding: 1.2rem !important;
            box-shadow: 0 4px 12px var(--shadow) !important;
            transition: all 0.3s ease !important;
        }}
        
        /* Button styling - ***FIXED TEXT COLOR HERE*** */
        .stButton > button {{
            /* Existing styles for background and shape */
            background: linear-gradient(135deg, var(--success-color), var(--primary-color)) !important;
            color: white !important; /* This was the issue on light mode background */
            border: none !important;
            border-radius: 25px !important;
            padding: 1rem 3rem !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            height: 60px !important;
        }}

      
        .stFormSubmitButton button {{
            /* Override the light theme's attempt to make the button background light */
            background: linear-gradient(135deg, var(--success-color), var(--primary-color)) !important; 
            color: #0f172a !important; /* Force DARK TEXT COLOR for visibility on light/white button background */
        }}
        
        /* Ensure the form submit button text is dark for contrast if the button background is white */
        .stFormSubmitButton div > button > div > div {{
            color: #0f172a !important; /* Secondary text color fallback */
        }}

        /* Restore primary button color on hover */
        .stButton > button:hover {{
            transform: none !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        }}
        
        /* Apply simplified slider CSS */
        {simplified_slider_css}

        /* Result styling (kept) */
        .result-container {{
            background: linear-gradient(135deg, var(--success-color), var(--primary-color));
            color: white;
            padding: 2.5rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.4);
            animation: slideUp 0.5s ease;
            font-size: 6rem; /* INCREASED SIZE */
            font-weight: 700;
        }}
        
        /* General Overrides (kept) */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display:none;}}

        /* Ensure the text within the prediction button is dark for contrast */
        .stFormSubmitButton button span {{
            color: #0f172a !important; 
        }}
        
        </style>
        """, unsafe_allow_html=True
    )

# Apply hardcoded dark theme styles
apply_dark_theme_styles()

# -----------------------------
# Sidebar Content
# -----------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-header">🚴 Bike Rental Predictor</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-subheader">🌐 Application Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This web application uses a **Machine Learning model** to forecast the 
        expected demand for rental bikes based on various real-world factors.
        """
    )
    
    st.markdown('<div class="sidebar-subheader">💡 Key Uses</div>', unsafe_allow_html=True)
    st.markdown(
        """
        * **Inventory Management:** Helps bike rental services optimize the number 
            of bikes available at stations.
        * **Staff Scheduling:** Aids in predicting peak demand times for better 
            staff deployment.
        * **Pricing Strategy:** Informs dynamic pricing based on predicted demand 
            and weather conditions.
        """
    )
    
    st.markdown('<div class="sidebar-subheader">✅ Advantages</div>', unsafe_allow_html=True)
    st.markdown(
        """
        * **Data-Driven Decisions:** Predictions are based on historical data patterns (weather, season, hour, etc.).
        * **Efficiency:** Reduces guesswork, leading to more efficient resource allocation.
        * **User-Friendly:** Simple, intuitive sliders and input fields allow non-technical users to generate forecasts easily.
        """
    )
    
    st.markdown('<div class="sidebar-subheader">📝 How to Use</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1.  Adjust the **Date & Time** and **Environmental Parameters** (e.g., temperature, wind speed) on the main screen.
        2.  Select the **Weather Situation** and **Special Conditions** (Holiday/Working Day).
        3.  Click the **"Predict Bike Rentals"** button.
        4.  View the forecast in the **Prediction Result** section and check the **Insights** for factors driving the result.
        """
    )
    
    st.markdown('<hr style="border-top: 1px solid var(--border-color);">', unsafe_allow_html=True)
    st.markdown(
        f'<p style="text-align: center; color: var(--text-muted); font-size: 0.8rem;">Version 1.0</p>', 
        unsafe_allow_html=True
    )
# Main header
st.markdown("""
<div class="main-header">
    <h1>🚴 Bike Rental Prediction</h1>
    <p>Predict bike rentals based on weather, time, and day information using machine learning</p>
</div>
""", unsafe_allow_html=True)

# Options
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
weather_mapping = {
    "Clear, Few clouds": 1,
    "Mist, Cloudy, Mist": 2,
    "Light Rain, Thunderstorm, Scattered clouds": 3,
    "Heavy Rain, Thunderstorm": 4,
}

# Form container
#st.markdown('<div class="form-container">', unsafe_allow_html=True)

with st.form("predict_form"):
    # Create two main columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">📅 Date & Time Settings</div>', unsafe_allow_html=True)
        
        # Sub-columns for better organization
        date_col1, date_col2 = st.columns(2)
        
        with date_col1:
            season = st.selectbox("🌱 Season", ["Spring", "Summer", "Fall", "Winter"], 
                                 index=0, help="Select the current season")
            yr = st.radio("📅 Year", [2011, 2012], index=0, horizontal=True, 
                          help="Select the year for prediction")
            
        with date_col2:
            mnth = st.selectbox("📆 Month", months, index=0, 
                                 help="Select the month")
            weekday = st.selectbox("📅 Day of Week", weekdays, index=0, 
                                     help="Select the day of the week")
        
        # Full width for hour
        st.markdown("<br>", unsafe_allow_html=True) # Extra space before the slider
        hr = st.slider("⏰ Hour of Day", 1, 24, 15, 
                       help="Select the hour of the day (1-24)")
        
        st.markdown('<div class="section-header">🌤️ Weather Conditions</div>', unsafe_allow_html=True)
        weather_choice = st.selectbox("🌦️ Weather Situation", list(weather_mapping.keys()),
                                     help="Select the current weather conditions")
        st.markdown("<br>", unsafe_allow_html=True) # Extra space below the select box

    with col2:
        st.markdown('<div class="section-header">🌡️ Environmental Parameters</div>', unsafe_allow_html=True)
        
        temp = st.slider("🌡️ Temperature (0-1)", 0.0, 1.0, 0.5, step=0.01,
                         help="Normalized temperature (0=very cold, 1=very hot)")
        atemp = st.slider("🤗 Feels Like (0-1)", 0.0, 1.0, 0.5, step=0.01,
                          help="Normalized feeling temperature")
        hum = st.slider("💧 Humidity (0-1)", 0.0, 1.0, 0.5, step=0.01,
                       help="Normalized humidity level")
        windspeed = st.slider("🌬️ Wind Speed (0-1)", 0.0, 1.0, 0.2, step=0.01,
                              help="Normalized wind speed")
        
        st.markdown('<div class="section-header">🎯 Special Conditions</div>', unsafe_allow_html=True)
        
        special_col1, special_col2 = st.columns(2)
        with special_col1:
            holiday = st.radio("🎉 Holiday?", ["No", "Yes"], index=0, horizontal=True,
                                 help="Is it a public holiday?")
        with special_col2:
            workingday = st.radio("💼 Working Day?", ["No", "Yes"], index=0, horizontal=True,
                                 help="Is it a working day?")
        st.markdown("<br>", unsafe_allow_html=True) # Extra space below the radio buttons


    # Centered submit button
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        submitted = st.form_submit_button("🚴 Predict Bike Rentals")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction and results
if submitted:
    # Fix the weathersit variable
    weathersit = weather_mapping[weather_choice]
    
    sample_input = {
        "season": ["Spring", "Summer", "Fall", "Winter"].index(season) + 1,
        "yr": 0 if yr == 2011 else 1,
        "mnth": months.index(mnth) + 1,
        "holiday": 1 if holiday == "Yes" else 0,
        "weekday": weekdays.index(weekday),
        "workingday": 1 if workingday == "Yes" else 0,
        "weathersit": weathersit,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "hr": hr
    }

    try:
        # Using a dummy prediction value for the sake of runnable code
        prediction = predict_bikes(sample_input) 
        
        # Display result with enhanced styling
        st.markdown(f"""
        <div class="result-container">
            <h2>🎯 Prediction Result</h2>
            <div class="result-number">{prediction}</div>
            <p style="font-size: 1.2rem; margin: 0;">Expected bike rentals for the given conditions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights in expandable section
        with st.expander("📊 Prediction Insights & Input Summary", expanded=True):
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.metric("🌡️ Temperature Impact", f"{temp:.2f}", 
                          help="Higher temperatures generally increase rentals")
                st.metric("💧 Humidity Level", f"{hum:.2f}", 
                          help="Lower humidity is better for cycling")
            
            with insight_col2:
                weather_impact = {1: "Excellent", 2: "Good", 3: "Fair", 4: "Poor"}
                st.metric("🌤️ Weather Impact", weather_impact[weathersit], 
                          help="Weather conditions significantly affect demand")
                st.metric("🌬️ Wind Conditions", f"{windspeed:.2f}", 
                          help="Lower wind speed is preferred")
            
            with insight_col3:
                time_impact = "Peak" if hr in [7,8,9,17,18,19] else "High" if 10 <= hr <= 16 else "Low"
                st.metric("⏰ Time Impact", time_impact, 
                          help="Rush hours and daytime see higher demand")
                day_impact = "Weekend" if not (workingday == "Yes") else "Weekday"
                st.metric("📅 Day Type", day_impact, 
                          help="Weekend vs weekday affects rental patterns")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check that all model files are properly loaded and input parameters are valid.")

# Footer
st.markdown("""
<br><br>
<div style="text-align: center; color: var(--text-muted); font-size: 0.9rem; padding: 2rem;">
    🚴 Advanced Bike Rental Prediction System | by M Sreeshanth Sai
</div>
""", unsafe_allow_html=True)