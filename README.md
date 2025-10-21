# 🚲 RideWise: Bike Rental Prediction System 🚲

## 📘 Abstract
RideWise is a sophisticated bike rental prediction system that leverages machine learning to forecast rental demand based on weather conditions and temporal factors. This intelligent system helps bike rental companies optimize their fleet management and improve service availability.

## 🌟 Features

- 📊 **Predictive Analytics**: Uses Random Forest Regression to predict bike rental demand
- 🖥️ **Interactive UI**: User-friendly interface with sliders and selectors for input parameters
- 🕒 **Hourly Predictions**: Option to view predicted rentals for all hours of the day
- 📜 **Prediction History**: Tracks and displays previous predictions
- 📈 **Data Visualization**: Graphical representation of hourly predictions

## 🧰 Technical Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming language |
| Pandas & NumPy | Data manipulation and processing |
| Scikit-learn | Machine learning model implementation |
| Streamlit | Web application framework |
| Matplotlib | Data visualization |
| Joblib | Model serialization |

## 🗂️ Project Structure

- `hour.csv`: Raw dataset containing bike rental information
- `create_model.py`: Script for data preprocessing and model training
- `app.py`: Streamlit application for user interaction
- `Project.ipynb`: Jupyter notebook with initial data exploration

## 🔄 Data Processing Workflow

1. **Data Preprocessing**:
   - Column renaming for better readability
   - Log transformation of the target variable
   - Removal of unnecessary columns
   - One-hot encoding of categorical features

2. **Model Training**:
   - Random Forest Regressor with 100 estimators
   - 80/20 train-test split
   - Model serialization for later use

3. **Prediction Process**:
   - User input collection through the UI
   - Input preprocessing to match model requirements
   - Prediction using the trained model
   - Conversion of log predictions back to original scale

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Kush2250/RideWise-BikeDemand-Prediction.git
cd RideWise-BikeDemand-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

Then open your browser and access the application!

## 📊 Model Performance

The Random Forest Regressor was chosen for its ability to handle both numerical and categorical features effectively. The model was trained on historical bike rental data with the following preprocessing steps:

- Log transformation of the target variable to handle skewness
- One-hot encoding of categorical features
- Feature selection to remove redundant information

## 🔮 Future Improvements

- Integration with real-time weather data
- Addition of more advanced models for comparison
- Geographic visualization of rental predictions
- User authentication and personalized predictions

## 👨‍💻 Author

**Kushagra Prakash Singhal**
- 📧 Email: kushagraa756@gmail.com

- 🔗 GitHub: [Kush2250](https://github.com/Kush2250)

## 📝 License

This project is open-source and available for educational and commercial use.

## 🙏 Acknowledgements

- The dataset used in this project is the Bike Sharing Dataset from the UCI Machine Learning Repository
- Open-source data science ecosystem: pandas, scikit-learn, Streamlit