

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(data_path, model_path):
    # Load data
    df = pd.read_csv(data_path)

    # Features: Year and Month
    X = df[['Year', 'Month']]
    # Target: Average Temperature
    y = df['Average_Temperature_C']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5  # Compute RMSE manually for older scikit-learn
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.2f}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    # Paths
    data_path = 'data/cleaned/cleaned_climate_data.csv'
    model_path = 'models/climate_rf_model.pkl'

    train_model(data_path, model_path)