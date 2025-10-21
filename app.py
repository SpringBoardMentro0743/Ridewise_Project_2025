import streamlit as st
import pandas as pd
import altair as alt
import pickle
import numpy as np

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Ridewise - Bike Rentals Predictor", layout="wide")

# ---------------------------
# Title and Subtitle
# ---------------------------
st.markdown(
    "<h1 style='text-align: center; color:#FF69B4; font-style:italic; font-size:48px;'>🚲 Ridewise</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center; color:#2E86AB; font-size:28px;'>🚀 Bike Sharing Demand Prediction App 🚲</h2>",
    unsafe_allow_html=True
)

# ---------------------------
# Load Dataset
# ---------------------------
st.header("📊 Bike Rentals Sample Dataset")
hour = pd.read_csv("/home/vibhuti/Documents/hour.csv")

# Map season numbers to names
season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
hour["season"] = hour["season"].map(season_map)

# Convert normalized temps to Celsius
hour["temp_c"] = hour["temp"] * 100
hour["atemp_c"] = hour["atemp"] * 100

st.dataframe(hour.head())

# ---------------------------
# Analytics Section
# ---------------------------
st.header("📈 Bike Rentals Analytics")

scale = alt.Scale(
    domain=["spring", "summer", "fall", "winter"],
    range=["#e7ba52", "#aec7e8", "#1f77b4", "#9467bd"],
)
color = alt.Color("season:N", scale=scale)

brush = alt.selection_interval(encodings=["x"])
click = alt.selection_point(fields=["season"])

points = (
    alt.Chart(hour)
    .mark_point()
    .encode(
        x=alt.X("temp_c:Q", title="Daily Temperature (°C)"),
        y=alt.Y("cnt:Q", title="Bike Rentals Count"),
        color=alt.condition(brush, color, alt.value("lightgray")),
        size=alt.Size("atemp_c:Q", title="Feels-like Temp (°C)"),
        tooltip=["temp_c", "atemp_c", "cnt", "season"],
    )
    .properties(width=550, height=300)
    .add_params(brush)
    .transform_filter(click)
)

bars = (
    alt.Chart(hour)
    .mark_bar()
    .encode(
        x="cnt:N",
        y="season:N",
        color=alt.condition(click, color, alt.value("lightgray")),
    )
    .transform_filter(brush)
    .properties(width=550, height=170)
    .add_params(click)
)

chart = alt.vconcat(points, bars, title="Bike Rentals Count corresponding to different temperatures")
st.altair_chart(chart, use_container_width=True)

st.scatter_chart(
    hour,
    x="cnt",
    y=["hum", "windspeed"],
    color=["#FF0000", "#0000FF"],
    x_label="Bike Rental Count",
    y_label="Humidity & Windspeed",
    height=400
)

# ---------------------------
# Prediction Section
# ---------------------------
st.header("🚀 Bike Rentals Predictor")

# Season selector
season_dict = {1: "Spring 🌸", 2: "Summer ☀️", 3: "Fall 🍂", 4: "Winter ❄️"}
season = st.selectbox("Select a season:", options=list(season_dict.keys()), format_func=lambda x: season_dict[x])
st.markdown(f"<h3 style='color:#2E86AB;'>Selected season: {season_dict[season]}</h3>", unsafe_allow_html=True)

# Year selector
year = st.selectbox("Select a year:", options=[2011, 2012])
st.markdown(f"<h3 style='color:#2E86AB;'>Selected year: {year}</h3>", unsafe_allow_html=True)

# Hour selector
col1, col2 = st.columns([1, 2])
with col1:
    hour = st.number_input("Enter hour (0–23):", min_value=0, max_value=23, value=0, step=1)
with col2:
    clock_emojis = {0: "🕛", 12: "🕛", 1: "🕐", 13: "🕐", 2: "🕑", 14: "🕑", 3: "🕒", 15: "🕒", 4: "🕓", 16: "🕓", 5: "🕔", 17: "🕔", 6: "🕕", 18: "🕕", 7: "🕖", 19: "🕖", 8: "🕗", 20: "🕗", 9: "🕘", 21: "🕘", 10: "🕙", 22: "🕙", 11: "🕚", 23: "🕚"}
    if hour == 0:
        hour_display = 12; period = "AM"
    elif hour < 12:
        hour_display = hour; period = "AM"
    elif hour == 12:
        hour_display = 12; period = "PM"
    else:
        hour_display = hour - 12; period = "PM"
    emoji = clock_emojis.get(hour, "❓")
    st.markdown(f"<h3 style='font-size:24px;'>{hour_display}:00 {period} {emoji}</h3>", unsafe_allow_html=True)

# Month selector
month_dict = {1: "January 🕛", 2: "February 🕐", 3: "March 🕒", 4: "April 🌱", 5: "May 🌼", 6: "June ☀️", 7: "July ☀️", 8: "August 🌞", 9: "September 🍂", 10: "October 🎃", 11: "November 🍁", 12: "December 🎄"}
month = st.selectbox("Select a month:", options=list(month_dict.keys()), format_func=lambda x: month_dict[x])
st.markdown(f"<h3 style='color:#FF6347;'>Selected month: {month_dict[month]}</h3>", unsafe_allow_html=True)

# Weekday selector
weekday = st.selectbox("Select weekday:", options=list(range(0, 7)), format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][x])

# Holiday selector
holiday = st.selectbox("Is it a holiday day?", options=[0, 1], format_func=lambda x: "No ❌" if x == 0 else "Yes ✅")

# Weather situation selector
weathersit_dict = {1: "Clear, Few clouds ☀️", 2: "Mist + Cloudy ☁️", 3: "Light Snow 🌦", 4: "Heavy Rain ⛈"}
weathersit = st.selectbox("Weather situation:", options=list(weathersit_dict.keys()), format_func=lambda x: weathersit_dict[x])

# Other inputs
temp = st.number_input("Temperature (0 to 1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
atemp = st.number_input("Feels like temperature (0 to 1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
hum = st.number_input("Humidity (0 to 1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
windspeed = st.number_input("Windspeed (0 to 1):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
registered = st.number_input("Registered bike users:", min_value=0.0, max_value=10000.0, value=0.5, step=0.01)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    with open("/home/vibhuti/Documents/lasso_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# ---------------------------
# Prediction Button
# ---------------------------
st.subheader("🚀 Predict Bike Rentals")

def part_of_day(hour):
    if 6 <= hour < 10:
        return "morning"
    elif 10 <= hour < 17:
        return "noon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

input_df = pd.DataFrame({
    "season": [season],
    "yr": [year],
    "mnth": [month],
    "holiday": [holiday],
    "hr": [hour],
    "weekday": [weekday],
    "weathersit": [weathersit],
    "temp": [temp],
    "atemp": [atemp],
    "hum": [hum],
    "windspeed": [windspeed],
    "registered": [registered]
})

input_df["part_of_day"] = input_df["hr"].apply(part_of_day)
input_df["working_hr_interaction"] = input_df["weekday"] * input_df["hr"]
input_df["bad_weather_flag"] = input_df["weathersit"].apply(lambda x: 1 if x >= 3 else 0)

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Bike Rentals: {prediction[0]:.2f}")
