import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check the current directory and list files
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())

# Load the dataset from a CSV file
try:
    print("Loading dataset...")
    df = pd.read_csv(r"E:\keshar language\python\chapter-ps\house price Prediction\house_price_prediction_data_set.csv")  # Full path
    print("Data loaded successfully!")
    print(df.head())  # Check the first few rows of the dataset
except Exception as e:
    print(f"Error loading data: {e}")

# Check the column names
print("Column names in the dataset:", df.columns)

# Specify the target and features based on the actual column name
# Assuming the column name for house price per unit area is 'Y house price of unit area'
X = df.drop(columns=['Y house price of unit area'])  
y = df['Y house price of unit area']  

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split successfully.")

# Standardize the data
print("Standardizing the data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data standardized.")

# Initialize models
lr_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
rf_model = RandomForestRegressor(n_estimators=10, random_state=42)  # Reduced n_estimators for testing

# Train models
print("Training Linear Regression model...")
lr_model.fit(X_train, y_train)
print("Linear Regression model trained.")

print("Training Ridge Regression model...")
ridge_model.fit(X_train, y_train)
print("Ridge Regression model trained.")

print("Training Random Forest model...")
rf_model.fit(X_train, y_train)
print("Random Forest model trained.")

# Model evaluation on test data
print("Evaluating models...")
models = {'Linear Regression': lr_model, 'Ridge Regression': ridge_model, 'Random Forest': rf_model}
    
for name, model in models.items():
    print(f"--- {name} ---")
    y_pred = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R-squared Score:", r2_score(y_test, y_pred))

# Feature importance (for Random Forest model)
feature_importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
print("Plotting feature importance for Random Forest model...")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance for House Price Prediction")
plt.show()
