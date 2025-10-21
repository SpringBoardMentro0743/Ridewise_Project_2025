# RideWise Bike Demand Prediction - Streamlit Version
# Streamlit application for bike sharing demand prediction

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import datetime
import os

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="🚴‍♂️ RideWise - Bike Demand Prediction",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Custom CSS
# ==============================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .prediction-box {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-number {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    h1 {
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Custom Feature Engineer Class
# ==============================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer that creates derived features
    from raw bike sharing data.
    """
    def __init__(self, peak_hours=None):
        if peak_hours is None:
            peak_hours = (7, 8, 9, 17, 18, 19)
        self.peak_hours = peak_hours

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Convert dteday to datetime if present and extract features
        if 'dteday' in X.columns:
            X['dteday'] = pd.to_datetime(X['dteday'], errors='coerce')
            X['year'] = X['dteday'].dt.year
            X['month'] = X['dteday'].dt.month
            X['day'] = X['dteday'].dt.day
            X['weekday'] = X['dteday'].dt.weekday
        else:
            X['year'] = X.get('yr', 2024)
            X['month'] = X.get('mnth', 1)
            X['day'] = X.get('day', 1)
            X['weekday'] = X.get('weekday', 0)

        # Weekend indicator
        X['is_weekend'] = X['weekday'].apply(lambda d: 1 if int(d) >= 5 else 0)

        # Peak hour indicator
        if 'hr' in X.columns:
            peak_set = set(self.peak_hours)
            X['is_peak_hour'] = X['hr'].apply(lambda h: 1 if int(h) in peak_set else 0)
        else:
            X['is_peak_hour'] = 0

        # Handle holiday and workingday columns
        if 'holiday' in X.columns:
            X['holiday'] = X['holiday'].fillna(0).astype(int)
        if 'workingday' in X.columns:
            X['workingday'] = X['workingday'].fillna(0).astype(int)

        return X

# ==============================
# Load or Create Model Pipeline
# ==============================
@st.cache_resource
def load_or_create_pipeline():
    """Load existing pipeline or create a new one"""
    try:
        # Try to load existing pipeline
        pipeline = joblib.load("ridewise_pipeline.pkl")
        return pipeline, True
    except FileNotFoundError:
        # Create sample pipeline if not found
        return create_sample_pipeline(), False

def create_sample_pipeline():
    """Create and train a new pipeline with sample data"""
    try:
        # Generate sample training data
        np.random.seed(42)
        n_samples = 1000
        
        # Create sample datetime range
        date_range = pd.date_range('2023-01-01', periods=n_samples, freq='H')
        
        sample_data = pd.DataFrame({
            'dteday': date_range,
            'season': np.random.choice([1,2,3,4], n_samples),
            'hr': date_range.hour,
            'holiday': np.random.choice([0,1], n_samples, p=[0.95, 0.05]),
            'weekday': date_range.weekday,
            'workingday': np.random.choice([0,1], n_samples, p=[0.3, 0.7]),
            'weathersit': np.random.choice([1,2,3,4], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            'temp': np.random.uniform(0.1, 0.9, n_samples),
            'atemp': np.random.uniform(0.1, 0.9, n_samples),
            'hum': np.random.uniform(0.3, 0.9, n_samples),
            'windspeed': np.random.uniform(0.1, 0.6, n_samples),
        })
        
        # Create realistic target
        base_demand = 50
        peak_boost = sample_data['hr'].apply(lambda h: 100 if h in [7,8,9,17,18,19] else 0)
        weather_penalty = sample_data['weathersit'].apply(lambda w: 0 if w<=2 else -30*(w-2))
        weekend_boost = sample_data['weekday'].apply(lambda d: 50 if d >= 5 else 0)
        temp_boost = sample_data['temp'] * 200
        
        sample_data['cnt'] = (base_demand + peak_boost + weather_penalty + 
                             weekend_boost + temp_boost + 
                             np.random.normal(0, 30, n_samples)).clip(0, 1000)
        
        # Prepare features and target
        y = sample_data['cnt']
        X = sample_data.drop(['cnt'], axis=1)
        
        # Define feature groups
        numeric_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'year', 'month', 'day', 'weekday']
        categorical_features = ['season', 'weathersit']
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')
        
        # Create full pipeline
        xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        
        pipeline = Pipeline([
            ('feature_engineer', FeatureEngineer()),
            ('preprocessor', preprocessor),
            ('model', xgb_model)
        ])
        
        # Train the pipeline
        pipeline.fit(X, y)
        
        # Save the trained pipeline
        joblib.dump(pipeline, "ridewise_pipeline.pkl")
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Error creating pipeline: {e}")
        return None

# ==============================
# Utility Functions
# ==============================
def build_input_dataframe(date, hour, season, weathersit, temp, humidity, windspeed, holiday, workingday):
    """Convert inputs to DataFrame for prediction"""
    dteday = pd.to_datetime(date)
    
    data = {
        'dteday': [dteday],
        'season': [season],
        'yr': [dteday.year - 2011],
        'mnth': [dteday.month],
        'hr': [hour],
        'holiday': [holiday],
        'weekday': [dteday.weekday()],
        'workingday': [workingday],
        'weathersit': [weathersit],
        'temp': [temp],
        'atemp': [temp],
        'hum': [humidity],
        'windspeed': [windspeed]
    }
    
    return pd.DataFrame(data)

# ==============================
# Main Application
# ==============================
def main():
    # Header
    st.markdown("<h1>🚴‍♂️ RideWise</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>AI-Powered Bike Sharing Demand Prediction</p>", unsafe_allow_html=True)
    
    # Load pipeline
    pipeline, loaded_from_file = load_or_create_pipeline()
    
    if pipeline is None:
        st.error("❌ Failed to load or create model pipeline. Please check the logs.")
        return
    
    # Status indicator
    if loaded_from_file:
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ Using sample model. For better predictions, train with real data.")
    
    st.markdown("---")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📅 Date & Time")
        date_col, hour_col = st.columns(2)
        with date_col:
            selected_date = st.date_input("Date", datetime.date.today())
        with hour_col:
            selected_hour = st.slider("Hour (0-23)", 0, 23, 12)
        
        st.markdown("### 🌤️ Weather Conditions")
        season = st.selectbox(
            "Season",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "🌸 Spring", 2: "☀️ Summer", 3: "🍂 Fall", 4: "❄️ Winter"}[x],
            index=1
        )
        
        weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)
        with weather_col1:
            if st.button("☀️ Clear", use_container_width=True):
                st.session_state.weather = 1
        with weather_col2:
            if st.button("☁️ Cloudy", use_container_width=True):
                st.session_state.weather = 2
        with weather_col3:
            if st.button("🌧️ Light Rain", use_container_width=True):
                st.session_state.weather = 3
        with weather_col4:
            if st.button("⛈️ Heavy Rain", use_container_width=True):
                st.session_state.weather = 4
        
        if 'weather' not in st.session_state:
            st.session_state.weather = 1
        
        weathersit = st.session_state.weather
        weather_names = {1: "☀️ Clear", 2: "☁️ Cloudy", 3: "🌧️ Light Rain", 4: "⛈️ Heavy Rain"}
        st.info(f"Selected: {weather_names[weathersit]}")
        
        st.markdown("### 🌡️ Environmental Conditions")
        temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5, 0.01)
        humidity = st.slider("Humidity (normalized)", 0.0, 1.0, 0.6, 0.01)
        windspeed = st.slider("Wind Speed (normalized)", 0.0, 1.0, 0.2, 0.01)
        
        st.markdown("### 📌 Special Days")
        special_col1, special_col2 = st.columns(2)
        with special_col1:
            holiday = st.checkbox("🎉 Holiday")
        with special_col2:
            workingday = st.checkbox("💼 Working Day", value=True)
    
    with col2:
        st.markdown("### 📊 Quick Info")
        st.info(f"""
        **Date:** {selected_date.strftime('%B %d, %Y')}  
        **Time:** {selected_hour}:00  
        **Season:** {season}  
        **Weather:** {weather_names[weathersit]}  
        **Temperature:** {temp:.2f}  
        **Humidity:** {humidity:.2f}  
        **Wind Speed:** {windspeed:.2f}
        """)
        
        st.markdown("### 🎯 Prediction")
        if st.button("🚀 Predict Demand", use_container_width=True):
            try:
                # Build input
                input_df = build_input_dataframe(
                    selected_date, selected_hour, season, weathersit,
                    temp, humidity, windspeed,
                    1 if holiday else 0,
                    1 if workingday else 0
                )
                
                # Make prediction
                with st.spinner("Calculating..."):
                    prediction = pipeline.predict(input_df)[0]
                    prediction = max(0, int(round(prediction)))
                
                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🚴‍♂️ Predicted Bike Demand</h3>
                    <div class="prediction-number">{prediction}</div>
                    <p>bikes expected to be rented</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                if prediction > 400:
                    st.success("🔥 High demand expected!")
                elif prediction > 200:
                    st.info("📈 Moderate demand expected")
                else:
                    st.warning("📉 Low demand expected")
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: white;'>
    RideWise © 2025
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
