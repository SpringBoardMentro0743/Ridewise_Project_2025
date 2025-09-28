🚴 Bike Demand Prediction Web App

This is a _Streamlit web application_ that predicts bike rental demand based on various input features.  
It uses _Machine Learning models_ such as Random Forest, XGBoost, LightGBM, and Linear Regression to provide accurate demand predictions.

## 🌟 Features

- 📊 Interactive UI built with _Streamlit_
- 🎨 Customized color themes for a better user experience
- 🤖 Predicts bike demand using trained ML models
- 📈 Includes visualizations and insights for data analysis
- 💾 Pre-trained models ready to use (XGBoost, LightGBM, etc.)

## 🛠 Technologies Used

- Python 3
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- XGBoost, LightGBM

## 📂 Project Structure

intern/ │── .streamlit/ # Streamlit configuration
│ └── config.toml
│── analysis.ipynb # Data analysis & model building notebook
│── app.py # Main Streamlit web application
│── feature_importance.csv # Feature importance scores
│── features.pkl # Saved feature data
│── hour.csv # Original dataset
│── lgbm_best_model.pkl # Trained LightGBM model
│── model_comparison.csv # Model comparison results
│── plot_by_hour.png # Visualization - usage by hour
│── plot_by_season.png # Visualization - usage by season
│── plot_correlation.png # Correlation heatmap
│── plot_temp_vs_cnt.png # Visualization - temp vs. demand
│── preprocessed_bike.csv # Preprocessed dataset
│── README.md # Project documentation
│── requirements.txt # Dependencies
│── scaler.pkl # Data scaler object
│── thresholds.pkl # Threshold values
│── xgb_best_model.pkl # Trained XGBoost model
│── xgb_bike_model.pkl # Another saved XGBoost model
│── xgboost_best_model.pkl # Best performing XGBoost model

## 📂 About the Dataset

The dataset used in this project is the _Bike Sharing Dataset_, which contains hourly and daily rental data.  
It includes information such as:

- Date and time
- Weather conditions (temperature, humidity, wind speed, etc.)
- Season and holiday information
- Count of total bikes rented

This dataset is commonly used for _time-series forecasting_ and _regression tasks_.

## 🚀 How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/MrsRathode/bike-demand-prediction.git
   cd bike-demand-prediction

   ```

2. Install dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run app.py

📊 Results & Insights

The app compares multiple ML models and selects the best performing one.

Visualizations such as hourly trends, seasonal demand, and correlations provide deeper insights into bike rental behavior.

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and create pull requests.

📜 License

This project is licensed under the MIT License.
