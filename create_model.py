import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('hour.csv')

# Rename columns to match the app
df = df.rename(columns={
    'yr': 'year',
    'mnth': 'month',
    'hr': 'hour',  # Rename hr to hour
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
joblib.dump(model, 'ridewise_model.pkl')
joblib.dump(x_columns, 'x_columns.pkl')

print("Model and feature columns saved successfully!")
print(f"Number of features: {len(x_columns)}")
print(f"Feature columns: {x_columns[:5]}...")
