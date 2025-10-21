import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Ride Insights",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #f5f5dc;
    }
    
    /* Dark mode styling */
    .dark-mode {
        background-color: #1e1e2e;
        color: #e0e0e0;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        padding: 10px;
    }
    
    .app-title {
        color: #4b6c43;
        font-size: 28px;
        margin-left: 10px;
        font-weight: bold;
    }
    
    /* Input parameters container */
    .input-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Dark mode input container */
    .dark-input-container {
        background-color: #2d2d3a;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Slider styling */
    .slider-label {
        font-weight: bold;
        margin-bottom: 5px;
        color: #333333;
    }
    
    /* Ensure all input labels and values are visible */
    .stMarkdown, .stSlider, .stSelectbox, .stRadio, .stCheckbox {
        color: #333333 !important;
    }
    
    /* Emoji labels */
    .emoji-label {
        font-weight: bold;
        color: #333333;
        font-size: 16px;
        margin-bottom: 5px;
    }
    
    /* Button styling */
    .predict-button {
        background-color: #e67e22;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        width: 100%;
        text-align: center;
    }
    
    .predict-button:hover {
        background-color: #d35400;
    }
    
    /* Toggle switch for dark mode */
    .toggle-container {
        position: absolute;
        top: 20px;
        right: 20px;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom dropdown styling */
    .custom-dropdown {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 8px 12px;
    }
    
    /* Dark mode dropdown */
    .dark-dropdown {
        background-color: #3d3d4b;
        border: 1px solid #555;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and feature columns or create them if not found
@st.cache_resource
def load_model():
    try:
        import pickle
        import joblib
        import os
        
        # Check if model files exist
        if os.path.exists('models/ridewise_model.pkl') and os.path.exists('models/x_columns.pkl'):
            # Load model and feature columns from the models directory
            model = joblib.load('models/ridewise_model.pkl')
            x_columns = joblib.load('models/x_columns.pkl')
            return model, x_columns
        elif os.path.exists('ridewise_model.pkl') and os.path.exists('x_columns.pkl'):
            # Load model and feature columns from the root directory
            model = joblib.load('ridewise_model.pkl')
            x_columns = joblib.load('x_columns.pkl')
            return model, x_columns
        else:
            # Model files not found, generate them
            return generate_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to generate the model if not found
def generate_model():
    try:
        import pandas as pd
        import numpy as np
        import joblib
        import os
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Check if dataset exists in dataset folder or root
        if os.path.exists('dataset/hour.csv'):
            df = pd.read_csv('dataset/hour.csv')
        elif os.path.exists('hour.csv'):
            df = pd.read_csv('hour.csv')
        else:
            st.error("Dataset not found. Please upload hour.csv to the app.")
            return None, None
        
        # Rename columns to match the app
        df = df.rename(columns={
            'yr': 'year',
            'mnth': 'month',
            'hr': 'hour',
            'hum': 'humidity',
            'weathersit': 'weather',
            'cnt': 'count'
        })
        
        # Apply log transformation to the target variable
        df['count'] = np.log(df['count'])
        
        # Drop unnecessary columns
        df = df.drop(columns=['instant', 'dteday', 'year', 'atemp', 'casual', 'registered'])
        
        # Create dummy variables for categorical features
        categorical_features = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
        df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)
        
        # Split features and target
        X = df_encoded.drop('count', axis=1)
        x_columns = X.columns.tolist()
        Y = df_encoded['count']
        
        # Split data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Train a RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, Y_train)
        
        # Save the model and feature columns
        joblib.dump(model, 'models/ridewise_model.pkl')
        joblib.dump(x_columns, 'models/x_columns.pkl')
        
        return model, x_columns
    except Exception as e:
        st.error(f"Error generating model: {e}")
        return None, None

model, x_columns = load_model()

# Function to preprocess input data
def preprocess_input(input_data, x_columns):
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])
    
    # Create dummy variables for categorical features
    categorical_features = ['season', 'month', 'hour', 'holiday', 'weekday', 'workingday', 'weather']
    
    # Initialize the preprocessed DataFrame with the numerical features
    preprocessed_df = pd.DataFrame()
    preprocessed_df['temp'] = input_df['temp']
    preprocessed_df['humidity'] = input_df['humidity']
    preprocessed_df['windspeed'] = input_df['windspeed']
    
    # One-hot encode categorical features
    for feature in categorical_features:
        # Create a prefix for the feature
        prefix = f"{feature}_"
        
        # Get the feature value
        value = input_df[feature].iloc[0]
        
        # Add columns for all possible values in x_columns
        for col in x_columns:
            if col.startswith(prefix):
                # Extract the value from the column name
                col_value = col.replace(prefix, "")
                
                # Set the column to 1 if it matches the input value, 0 otherwise
                try:
                    col_value_int = int(col_value)
                    preprocessed_df[col] = 1 if col_value_int == value else 0
                except ValueError:
                    # Handle non-integer column values if any
                    preprocessed_df[col] = 1 if col_value == str(value) else 0
    
    # Ensure all columns from x_columns are present in preprocessed_df
    for col in x_columns:
        if col not in preprocessed_df.columns:
            preprocessed_df[col] = 0
    
    # Reorder columns to match x_columns
    preprocessed_df = preprocessed_df[x_columns]
    
    return preprocessed_df

# Function to make predictions
def predict_rentals(input_data, model, x_columns):
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data, x_columns)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)
    
    # Convert from log scale back to original scale
    return np.exp(prediction[0])

# Function to predict for all hours of the day
def predict_all_hours(input_data, model, x_columns):
    hourly_predictions = []
    
    # Create a copy of the input data
    for hour in range(24):
        # Update the hour in the input data
        hourly_input = input_data.copy()
        hourly_input['hour'] = hour
        
        # Make prediction for this hour
        prediction = predict_rentals(hourly_input, model, x_columns)
        hourly_predictions.append(prediction)
    
    return hourly_predictions

# Function to add tooltips and custom styling
def add_tooltips():
    return {
        "temperature": "The temperature in Celsius",
        "humidity": "Relative humidity in percentage",
        "wind_speed": "Wind speed in km/h",
        "weather": "1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain/Snow"
    }

# Main app
def main():
    # Get tooltips
    tooltips = add_tooltips()
    
    # Initialize dark mode in session state if not exists
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # App header with logo and title
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        st.markdown("""
        <div class="header-container">
            <div style="background-color: #4b6c43; width: 50px; height: 50px; border-radius: 10px; display: flex; justify-content: center; align-items: center;">
                <span style="color: white; font-size: 24px;">🚴</span>
            </div>
            <div class="app-title">Ride Insights</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Dark mode toggle in the top right corner
    with col3:
        # Function to toggle dark mode
        def toggle_dark_mode():
            st.session_state.dark_mode = not st.session_state.dark_mode
        
        # Add theme toggle button using Streamlit components
        if st.session_state.dark_mode:
            if st.button("🌙", key="dark_toggle", on_click=toggle_dark_mode):
                pass
        else:
            if st.button("🌞", key="light_toggle", on_click=toggle_dark_mode):
                pass
        
        # Apply CSS based on current theme
        if st.session_state.dark_mode:
            st.markdown("""
            <style>
                .stApp {background-color: #1e1e2e;}
                .input-container {background-color: #2d2d3a !important;}
                .slider-label, h1, h2, h3, p {color: #e0e0e0 !important;}
                .stMarkdown, .stSlider, .stSelectbox, .stRadio, .stCheckbox, .stToggleSwitch {color: #e0e0e0 !important;}
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
                .stApp {background-color: #f5f5dc;}
                .input-container {background-color: #ffffff;}
                .slider-label, h1, h2, h3, p {color: #333333 !important;}
                .stToggleSwitch {color: #333333 !important;}
            </style>
            """, unsafe_allow_html=True)
    
    # Main title
    st.markdown("<h1 style='text-align: center; color: #333333; margin-top: 20px;'>Predict Hourly Bike Rentals</h1>", unsafe_allow_html=True)
    
    # Subtitle
    st.markdown("<p style='text-align: center; color: #666666; margin-bottom: 30px;'>✨ Powered by AI-driven historical data analysis</p>", unsafe_allow_html=True)
    
    # Input parameters container
    st.markdown("<div class='input-container'><h2 style='margin-top: 0; color: #333333;'>Input Parameters</h2>", unsafe_allow_html=True)
    
    # Create three columns for the first row of inputs
    col1, col2, col3 = st.columns(3)
    
    # Temperature slider
    with col1:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>🌡️ Temperature</div>", unsafe_allow_html=True)
        temp_celsius = st.slider("", min_value=0.0, max_value=40.0, value=25.0, step=0.5, key="temp_slider", format="%.1f°C", help=tooltips["temperature"])
        # Convert Celsius to normalized value for the model (0-1 scale)
        temp = (temp_celsius + 5) / 45
    
    # Season dropdown
    with col2:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>Season</div>", unsafe_allow_html=True)
        season = st.selectbox(
            "",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}[x],
            index=3  # Default to Fall
        )
    
    # Hour of the day
    with col3:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>🕒 Hour of the Day</div>", unsafe_allow_html=True)
        hour = st.selectbox(
            "",
            options=list(range(24)),
            format_func=lambda x: f"{x:02d}:00",
            index=13  # Default to 13:00
        )
    
    # Create three columns for the second row of inputs
    col1, col2, col3 = st.columns(3)
    
    # Humidity slider
    with col1:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>🔥 Humidity</div>", unsafe_allow_html=True)
        humidity_percent = st.slider("", min_value=0, max_value=100, value=60, step=1, key="humidity_slider", format="%d%%", help=tooltips["humidity"])
        # Convert percentage to normalized value for the model (0-1 scale)
        humidity = humidity_percent / 100
    
    # Month dropdown
    with col2:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>Month</div>", unsafe_allow_html=True)
        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month = st.selectbox(
            "",
            options=list(range(1, 13)),
            format_func=lambda x: month_names[x-1],
            index=8  # Default to September
        )
    
    # Weather condition
    with col3:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>Weather Condition</div>", unsafe_allow_html=True)
        weather = st.selectbox(
            "",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Clear / Few Clouds", 2: "Mist", 3: "Light Rain/Snow", 4: "Heavy Rain/Snow"}[x],
            index=0,  # Default to Clear
            help=tooltips["weather"]
        )
    
    # Create three columns for the third row of inputs
    col1, col2, col3 = st.columns(3)
    
    # Wind speed slider
    with col1:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>💨 Wind Speed</div>", unsafe_allow_html=True)
        wind_kmh = st.slider("", min_value=0.0, max_value=50.0, value=15.0, step=0.5, key="wind_slider", format="%.1f km/h", help=tooltips["wind_speed"])
        # Convert km/h to normalized value for the model (0-1 scale)
        windspeed = wind_kmh / 50
    
    # Day of the week
    with col2:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>Day of the Week</div>", unsafe_allow_html=True)
        weekday = st.selectbox(
            "",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            index=0  # Default to Monday
        )
    
    # Holiday checkbox
    with col3:
        st.markdown("<div class='slider-label' style='color: #333333; font-weight: bold;'>📅 Is it a holiday?</div>", unsafe_allow_html=True)
        # Add custom styling for the toggle
        st.markdown("""
        <style>
        /* Improve toggle visibility in light mode */
        .stToggleSwitch {
            background-color: #f0f0f0 !important;
            border: 1px solid #999 !important;
            padding: 2px !important;
            border-radius: 15px !important;
        }
        .stToggleSwitch > div {
            background-color: #4b6c43 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        holiday = st.toggle("", value=False)
    
    # Working day is derived from holiday and weekday
    workingday = 0 if holiday or weekday >= 5 else 1
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("Predict Rentals", type="primary", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close input container
    
    # Store the input data
    input_data = {
        'season': season,
        'month': month,
        'hour': hour,
        'holiday': int(holiday),
        'weekday': weekday,
        'workingday': workingday,
        'weather': weather,
        'temp': temp,
        'humidity': humidity,
        'windspeed': windspeed
    }
    
    # Initialize session state for prediction history if it doesn't exist
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Main content area
    if model is not None and x_columns is not None and predict_button:
        # Make prediction with a loading spinner
        with st.spinner("🔄 Calculating bike rental prediction..."):
            prediction = predict_rentals(input_data, model, x_columns)
            
            # Round prediction to nearest integer
            prediction_rounded = round(prediction)
            
            # Add current prediction to history
            timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.prediction_history.append({
                'timestamp': timestamp,
                'input': input_data.copy(),
                'prediction': prediction_rounded
            })
        
        # Display prediction with improved styling
        st.markdown(f"""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h2 style="color: #333; margin-top: 0; text-align: center;">Prediction Result</h2>
            <div style="display: flex; justify-content: center; align-items: center; margin: 30px 0;">
                <div style="font-size: 48px; font-weight: bold; color: #e67e22;">
                    {prediction_rounded}
                </div>
                <div style="margin-left: 15px; font-size: 18px; color: #666;">
                    Expected bike rentals
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add option to show hourly predictions
    show_hourly = st.checkbox("Show hourly predictions", value=False)
    
    # Show hourly predictions if requested
    if show_hourly:
        st.markdown("""<div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #4CAF50;">
            <h2 style="color: #4CAF50; margin-top: 0;">📊 Hourly Predictions</h2>
            <p>See how bike rentals vary throughout the day with your selected parameters.</p>
        </div>""", unsafe_allow_html=True)
        
        # Get predictions for all hours with a loading spinner
        with st.spinner("Generating hourly predictions..."):
            hourly_predictions = predict_all_hours(input_data, model, x_columns)
            
            # Create a DataFrame for plotting
            hourly_df = pd.DataFrame({
                'Hour': range(24),
                'Predicted Rentals': hourly_predictions
            })
            
            # Add time of day categories
            hourly_df['Time of Day'] = hourly_df['Hour'].apply(lambda x: 
                'Night (0-5)' if x < 6 else 
                'Morning (6-11)' if x < 12 else 
                'Afternoon (12-17)' if x < 18 else 
                'Evening (18-23)'
            )
            
            # Plot the hourly predictions with improved styling
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Define colors for different times of day
            colors = {
                'Night (0-5)': '#3949AB',
                'Morning (6-11)': '#FFA726',
                'Afternoon (12-17)': '#FF7043',
                'Evening (18-23)': '#5E35B1'
            }
            
            # Plot bars with different colors based on time of day
            for time_of_day, group in hourly_df.groupby('Time of Day'):
                ax.bar(group['Hour'], group['Predicted Rentals'], 
                      label=time_of_day, color=colors[time_of_day])
                
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Predicted Bike Rentals', fontsize=12)
            ax.set_title(f'Predicted Bike Rentals by Hour', fontsize=14)
            ax.set_xticks(range(24))
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend(title='Time of Day')
            
            # Add peak annotation
            peak_hour = hourly_df['Hour'][hourly_df['Predicted Rentals'].idxmax()]
            peak_rentals = hourly_df['Predicted Rentals'].max()
            ax.annotate(f'Peak: {round(peak_rentals)} bikes', 
                       xy=(peak_hour, peak_rentals),
                       xytext=(peak_hour, peak_rentals + 20),
                       arrowprops=dict(facecolor='#E91E63', shrink=0.05),
                       ha='center', fontweight='bold', color='#E91E63')
            
            # Display the plot
            st.pyplot(fig)
            
            # Add insights about the hourly pattern
            st.markdown("""<div style="background-color: #F3E5F5; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <h3 style="color: #7B1FA2; margin-top: 0;">📋 Hourly Pattern Insights</h3>""", unsafe_allow_html=True)
            
            # Calculate some insights
            avg_rentals = round(hourly_df['Predicted Rentals'].mean())
            peak_hour_formatted = f"{peak_hour if peak_hour < 12 else peak_hour-12 if peak_hour > 12 else 12}{' AM' if peak_hour < 12 else ' PM'}"
            
            morning_avg = round(hourly_df[hourly_df['Hour'].between(6, 11)]['Predicted Rentals'].mean())
            afternoon_avg = round(hourly_df[hourly_df['Hour'].between(12, 17)]['Predicted Rentals'].mean())
            evening_avg = round(hourly_df[hourly_df['Hour'].between(18, 23)]['Predicted Rentals'].mean())
            night_avg = round(hourly_df[hourly_df['Hour'].between(0, 5)]['Predicted Rentals'].mean())
            
            st.markdown(f"""
                <ul>
                    <li><strong>Peak Rental Time:</strong> {peak_hour_formatted} with {round(peak_rentals)} bikes</li>
                    <li><strong>Daily Average:</strong> {avg_rentals} bikes per hour</li>
                    <li><strong>Time of Day Averages:</strong>
                        <ul>
                            <li>Morning (6-11 AM): {morning_avg} bikes</li>
                            <li>Afternoon (12-5 PM): {afternoon_avg} bikes</li>
                            <li>Evening (6-11 PM): {evening_avg} bikes</li>
                            <li>Night (12-5 AM): {night_avg} bikes</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the data in a table with improved styling
            with st.expander("View hourly data"):
                st.dataframe(hourly_df, use_container_width=True)
    
    # Display prediction history with improved styling
    if st.session_state.prediction_history:
        st.markdown("""
        <div style="background-color: #FFF8E1; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #FFA000;">
            <h2 style="color: #FFA000; margin-top: 0;">📜 Prediction History</h2>
            <p>Review your previous predictions and compare different scenarios.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Table View", "📋 Detailed View", "📈 Comparison"])
        
        with tab1:
            # Create a DataFrame from the prediction history
            history_df = pd.DataFrame([
                {
                    'Timestamp': entry['timestamp'],
                    'Season': entry['input']['season'],
                    'Month': entry['input']['month'],
                    'Hour': entry['input']['hour'],
                    'Weather': entry['input']['weather'],
                    'Temperature': entry['input']['temp'],
                    'Predicted Rentals': entry['prediction']
                } for entry in st.session_state.prediction_history
            ])
            
            # Format the data for better display
            display_df = history_df.copy()
            
            # Format season
            season_map = {1: "Winter ❄️", 2: "Spring 🌱", 3: "Summer ☀️", 4: "Fall 🍂"}
            display_df['Season'] = display_df['Season'].map(season_map)
            
            # Format weather
            weather_map = {1: "Clear ☀️", 2: "Mist 🌫️", 3: "Light Rain/Snow 🌧️", 4: "Heavy Rain/Snow ⛈️"}
            display_df['Weather'] = display_df['Weather'].map(weather_map)
            
            # Format month
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            display_df['Month'] = display_df['Month'].apply(lambda x: month_names[x-1])
            
            # Display as a table with improved formatting
            st.dataframe(
                display_df,
                column_config={
                    "Predicted Rentals": st.column_config.NumberColumn(
                        "Predicted Rentals",
                        help="Number of predicted bike rentals",
                        format="%d bikes"
                    ),
                    "Temperature": st.column_config.NumberColumn(
                        "Temperature",
                        help="Temperature in normalized units",
                        format="%.2f"
                    ),
                    "Timestamp": st.column_config.DatetimeColumn(
                        "Timestamp",
                        help="When the prediction was made",
                        format="MMM DD, YYYY, hh:mm a"
                    )
                },
                use_container_width=True
            )
        
        with tab2:
            # Display detailed information for each prediction with improved styling
            for i, entry in enumerate(reversed(st.session_state.prediction_history)):
                # Determine prediction level and color for visual indicator
                pred_value = entry['prediction']
                if pred_value < 50:
                    demand_level = "Very Low Demand 📉"
                    demand_color = "#F44336"  # Red
                elif pred_value < 150:
                    demand_level = "Low Demand 🔻"
                    demand_color = "#FF9800"  # Orange
                elif pred_value < 300:
                    demand_level = "Moderate Demand ⚖️"
                    demand_color = "#FFC107"  # Amber
                elif pred_value < 500:
                    demand_level = "High Demand 🔺"
                    demand_color = "#4CAF50"  # Green
                else:
                    demand_level = "Very High Demand 📈"
                    demand_color = "#2196F3"  # Blue
                
                # Create expander with colored header based on prediction level
                with st.expander(f"Prediction {len(st.session_state.prediction_history) - i}: {entry['timestamp']} - {demand_level}"):
                    # Add a visual indicator of prediction level
                    st.markdown(f"""
                    <div style="background-color: {demand_color}20; padding: 10px; border-radius: 5px; margin-bottom: 15px; text-align: center; border: 1px solid {demand_color};">
                        <h3 style="color: {demand_color}; margin: 0;">{entry['prediction']} bikes - {demand_level}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div style="background-color: #E1F5FE; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <h3 style="color: #0288D1; margin-top: 0;">📅 Date & Time Factors</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Format season
                        season_text = {1: "Winter ❄️", 2: "Spring 🌱", 3: "Summer ☀️", 4: "Fall 🍂"}[entry['input']['season']]
                        
                        # Format month
                        month_names = ["January", "February", "March", "April", "May", "June", 
                                      "July", "August", "September", "October", "November", "December"]
                        month_text = month_names[entry['input']['month']-1]
                        
                        # Format weekday
                        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        weekday_text = weekday_names[entry['input']['weekday']]
                        
                        # Format hour with AM/PM
                        hour = entry['input']['hour']
                        hour_text = f"{hour if hour < 12 else hour-12 if hour > 12 else 12}{' AM' if hour < 12 else ' PM'}"
                        
                        st.markdown(f"""
                        <div style="margin-left: 10px;">
                            <p><strong>Season:</strong> {season_text}</p>
                            <p><strong>Month:</strong> {month_text}</p>
                            <p><strong>Hour:</strong> {hour_text}</p>
                            <p><strong>Weekday:</strong> {weekday_text}</p>
                            <p><strong>Holiday:</strong> {'Yes' if entry['input']['holiday'] == 1 else 'No'}</p>
                            <p><strong>Working Day:</strong> {'Yes' if entry['input']['workingday'] == 1 else 'No'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="background-color: #F1F8E9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <h3 style="color: #689F38; margin-top: 0;">🌤️ Weather Factors</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Format weather
                        weather_text = {1: "Clear ☀️", 2: "Mist 🌫️", 3: "Light Rain/Snow 🌧️", 4: "Heavy Rain/Snow ⛈️"}[entry['input']['weather']]
                        
                        st.markdown(f"""
                        <div style="margin-left: 10px;">
                            <p><strong>Weather:</strong> {weather_text}</p>
                            <p><strong>Temperature:</strong> {entry['input']['temp']:.2f}</p>
                            <p><strong>Humidity:</strong> {entry['input']['humidity']:.2f}</p>
                            <p><strong>Wind Speed:</strong> {entry['input']['windspeed']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.metric("Predicted Bike Rentals", entry['prediction'])
        
        with tab3:
            if len(st.session_state.prediction_history) > 1:
                # Create a comparison chart of predictions
                history_df = pd.DataFrame([
                    {
                        'Timestamp': entry['timestamp'],
                        'Prediction': entry['prediction']
                    } for entry in st.session_state.prediction_history
                ])
                
                # Add a prediction number for easier reference
                history_df['Prediction Number'] = range(1, len(history_df) + 1)
                
                # Create a bar chart comparing predictions
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(history_df['Prediction Number'], history_df['Prediction'], 
                             color=plt.cm.viridis(history_df['Prediction']/history_df['Prediction'].max()))
                
                ax.set_xlabel('Prediction Number', fontsize=12)
                ax.set_ylabel('Predicted Bike Rentals', fontsize=12)
                ax.set_title('Comparison of Prediction Results', fontsize=14)
                ax.set_xticks(history_df['Prediction Number'])
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontweight='bold')
                
                # Display the chart
                st.pyplot(fig)
                
                st.info("Make multiple predictions with different parameters to compare results.")
            else:
                st.info("Make at least two predictions to enable comparison.")
        
        # Add a button to clear history
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
            

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
    }
    .stSidebar .block-container {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .stMetric label {
        color: #000000 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stCaption {
        font-weight: bold;
        margin-top: 5px;
        color: #4b5563;
    }
    
    /* Improve button styling */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Improve tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    /* Improve dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Improve slider styling */
    .stSlider {
        padding-top: 0.5rem;
        padding-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Add tooltips and help text
def add_tooltips():
    # Create a dictionary of tooltips
    tooltips = {
        "temperature": "Temperature affects bike rentals significantly. Warmer temperatures typically increase rentals, while extreme heat or cold reduces them.",
        "humidity": "High humidity levels often decrease bike rentals as it makes physical activity less comfortable.",
        "wind_speed": "Strong winds can discourage cycling, especially for casual riders.",
        "season": "Seasonal patterns strongly influence bike rental behavior, with summer typically seeing the highest demand.",
        "weather": "Weather conditions directly impact cycling decisions. Clear days see more rentals than rainy or snowy days.",
        "hour": "Time of day shows distinct patterns, with commuting hours (8-9 AM, 5-6 PM) typically showing peaks on weekdays.",
        "holiday": "Holidays often show different rental patterns compared to regular weekdays.",
        "working_day": "Working days typically show commuting patterns, while non-working days show more recreational usage patterns."
    }
    
    return tooltips

if __name__ == "__main__":
    main()
