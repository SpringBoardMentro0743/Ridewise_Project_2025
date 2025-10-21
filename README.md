# 🚴‍♂️ Sreeshanth_RideWise  

**RideWise: Predicting Bike-Sharing Demand Based on Weather and Urban Events**  

---

## 📌 Project Overview  
RideWise is a predictive web application built using **Streamlit** and a **Machine Learning model** (e.g., Random Forest or Linear Regression).  
Its purpose is to **forecast the hourly demand** for shared bike rentals based on environmental, seasonal, and temporal factors.  

The application allows users to interactively adjust variables like **temperature, weather condition, time of day, and holiday status** to see an immediate prediction of bike rentals.  

---

## ✨ Features  
- **Interactive Input** → Sliders & dropdown menus for conditions (Temperature, Humidity, Wind Speed, Time, Season).  
- **Real-time Prediction** → Instant forecast of bike rental counts.  
- **Light & Dark Mode UI** → Clean, user-friendly interface with custom CSS.  
- **Deployment Ready** → Structured for deployment on **Streamlit Cloud** or similar platforms.  

---

## 📁 Project Structure  
```bash
Bike-app/
├── data/              # Final, cleaned datasets for training/testing
├── models/            # Trained ML model files (e.g., model.pkl / joblib)
├── notebooks/         # Jupyter notebooks (EDA, model training & evaluation)
├── src/               # Modular Python scripts (preprocessing, training, utils)
├── venv/              # Virtual environment (ignored by Git)
├── .gitignore         # Git ignore rules
├── Bikeapp.py         # Main Streamlit application script
├── README.md          # Project documentation (this file)
└── requirements.txt   # Python dependencies
```

---

## ⚙️ Installation & Setup  

### 🔧 Prerequisites  
- Python **3.8+**  

### 1. Clone the Repository  
```bash
git clone https://github.com/sreeshanthsai10/Sreeshanth_RideWise.git
cd Sreeshanth_RideWise
```

### 2. Create a Virtual Environment  
```bash
# Create environment
python -m venv venv

# Activate environment
# On Windows (PowerShell)
.\venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4. Run the Application  
```bash
streamlit run Bikeapp.py
```
➡️ The app will open in your browser at [http://localhost:8501](http://localhost:8501).  

---

## 📈 Model Details  
- **Algorithm**: Random Forest Regressor (or similar)  
- **Dataset**: Hourly Bike Sharing Data (e.g., Capital Bikeshare)  
- **Key Predictors**:  
  - Hour of Day  
  - Temperature (Normalized)  
  - Weather Situation  
  - Working Day Status  
- **Model File Location**: `models/`  

---

## 🤝 Contribution  
Contributions, issues, and feature requests are welcome!  
👉 Check the [issues page](https://github.com/sreeshanthsai10/Sreeshanth_RideWise/issues).  

---

## 📄 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

## 📧 Contact  
**Sreeshanth Sai**  
📩 Email: [sreeshanthsai10@gmail.com](mailto:sreeshanthsai10@gmail.com)  
🔗 Project Link: [GitHub Repo](https://github.com/sreeshanthsai10/Sreeshanth_RideWise)  
