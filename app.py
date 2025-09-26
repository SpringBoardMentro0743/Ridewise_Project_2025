import streamlit as st
import pandas as pd
import joblib

# ------------------- Load trained model -------------------
model = joblib.load("lgbm_best_model.pkl")

# ------------------- Custom CSS for Text Size -------------------
st.markdown(
    """
    <style>
    /* Title size */
    .title {
        font-size: 40px !important;
        font-weight: bold;
        text-align: center;
    }

    /* Subheader size */
    .subheader {
        font-size: 22px !important;
        text-align: center;
        color: gray;
    }

    /* General text size */
    .stMarkdown, .stText, p {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- Page Header -------------------
st.markdown("<div class='title'>🚲 Bike Sharing Demand Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Enter details below to predict bike rental demand</div>", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
with st.sidebar:
    st.title("About this App")
    st.write("""
    🚲 *Bike Sharing Demand Prediction App*  

    This app predicts bike rental demand based on:  
    - 🌡 Temperature (°C)  
    - 🤒 Feels Like Temp (°C)  
    - 💧 Humidity (%)  
    - 🍃 Windspeed (km/h)  
    - 📅 Day, Month & Hour  
    - 🎉 Holiday / Working Day  
    - 🌦 Weather Conditions  

    🔧 *Model Used*: LightGBM  
    👩‍💻 *Created by*: Banothu Anusha  
    """)

# ------------------- Main UI -------------------
st.subheader("🔽 Input Features")

season = st.selectbox("🍂 Season", ["Winter", "Spring", "Summer", "Fall"])

# Month mapping dictionary
months = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
month_name = st.selectbox("📆 Month", list(months.keys()), index=5)  # June default
month = months[month_name]

hour = st.selectbox("🕒 Hour of Day", list(range(1, 25)), index=12)
weekday = st.selectbox("📅 Day of Week",
                       ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])
workingday = st.radio("🏢 Working Day?", [0, 1],
                      format_func=lambda x: "Yes" if x == 1 else "No")
holiday = st.radio("🎉 Holiday?", [0, 1],
                   format_func=lambda x: "Yes" if x == 1 else "No")
weathersit = st.selectbox("🌦 Weather",
                          ["Clear", "Mist", "Light Snow/Rain", "Heavy Rain"])

# Sliders
temp = st.slider("🌡 Temperature (°C)", 0, 40, 20)
atemp = st.slider("🤒 Feels-like Temp (°C)", 0, 45, 25)
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
windspeed = st.slider("🍃 Windspeed (km/h)", 0, 50, 10)

# ------------------- Prepare Input -------------------
input_data = pd.DataFrame([{
    "yr": 1,
    "holiday": holiday,
    "workingday": workingday,
    "temp": temp / 40,
    "atemp": atemp / 50,
    "hum": humidity / 100,
    "windspeed": windspeed / 67,
    "season": season,
    "mnth": month,
    "hr": hour,
    "weekday": weekday,
    "weathersit": weathersit
}])

input_data = pd.get_dummies(input_data,
                            columns=["season","mnth","hr","weekday","weathersit"],
                            drop_first=True)

input_data["comfort_index"] = (temp/40 + atemp/50)/2 - (humidity/100)
input_data["temp_wind_interaction"] = (temp/40) * (windspeed/67)

expected_features = model.feature_name_ if hasattr(model, "feature_name_") else None
if expected_features:
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_features]


 # ------------------- Thresholds -------------------
threshold = 150   # cutoff for low vs high


# ------------------- Prediction -------------------
if st.button("🚀 Predict Demand"):
    prediction = model.predict(input_data)[0]

    if prediction < threshold:
        demand_status = "🔴 Low Demand"
        bg_color = "linear-gradient(135deg, #FF5252, #D32F2F)"
    else:
        demand_status = "🟢 High Demand"
        bg_color = "linear-gradient(135deg, #66BB6A, #2E7D32)"

    # Show prediction box
    st.markdown(f"""  
        <div style="  
            background: {bg_color};  
            padding: 25px;  
            border-radius: 15px;  
            text-align: center;  
            color: white;  
            font-size: 28px;  
            font-weight: bold;  
            box-shadow: 2px 2px 10px rgba(0,0,0,0.3);  
        ">  
            {demand_status}<br>  
            ✅ Predicted Bike Rentals: {int(prediction)} 🚴‍♂  
        </div>  
    """, unsafe_allow_html=True)

    # ------------------- Demand Comparison -------------------
    import matplotlib.pyplot as plt

    st.markdown("### 📊 Demand Comparison")

    demand_levels = ["Low Demand (<150)", "High Demand (≥150)", "Your Prediction"]
    demand_values = [threshold - 1, threshold, int(prediction)]
    colors = ["#FF5252", "#66BB6A", "#42A5F5"]  # red, green, blue

    fig, ax = plt.subplots()
    bars = ax.bar(demand_levels, demand_values, color=colors)

    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Bike Rentals")
    ax.set_title("Demand Comparison")
    st.pyplot(fig)

    # ------------------- Summary -------------------
    st.markdown(f"""
    ### 📝 Summary
    Based on the inputs, the predicted bike rental demand is *{demand_status}*  
    with an expected count of *{int(prediction)} bikes*.
    """)