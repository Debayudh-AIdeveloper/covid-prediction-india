"""
COVID-19 Case Prediction for India (State-wise Time Series)
=============================================================
Predicts confirmed COVID-19 cases using lag-based feature engineering
and regression models, focused on Maharashtra (highest case count state).

Author: [Your Name]
Dataset: Daily COVID-19 records for Indian states (Jan 2020 - present)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------------------------------
# STEP 1: Load Dataset
# -------------------------------------------------------------------
url = "https://raw.githubusercontent.com/<your-dataset-source>/covid_19_india.csv"
df = pd.read_csv(url)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nData info:")
print(df.info())

# -------------------------------------------------------------------
# STEP 2: Clean & Explore Data
# -------------------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'])

top_states = df.groupby('Name of State / UT')['Total Confirmed cases'].max().nlargest(5)
print("\nTop 5 states by cases:\n", top_states)

# Time series plot for Maharashtra
mah = df[df['Name of State / UT'] == 'Maharashtra']
plt.figure(figsize=(12, 5))
plt.plot(mah['Date'], mah['Total Confirmed cases'])
plt.title('COVID-19 Cases in Maharashtra Over Time')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.grid(True)
plt.savefig('maharashtra_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTotal cases by state (top 5):")
print(df.groupby('Name of State / UT')['Total Confirmed cases'].max().nlargest(5))

# -------------------------------------------------------------------
# STEP 3: Feature Engineering (Lag Features)
# -------------------------------------------------------------------
mah_data = df[df['Name of State / UT'] == 'Maharashtra'].copy()
mah_data = mah_data.sort_values('Date').reset_index(drop=True)
print(f"\nMaharashtra data: {len(mah_data)} days")

mah_data['cases_lag1'] = mah_data['Total Confirmed cases'].shift(1)
mah_data['cases_lag7'] = mah_data['Total Confirmed cases'].shift(7)
mah_data['cases_lag14'] = mah_data['Total Confirmed cases'].shift(14)
mah_data['new_cases_lag1'] = mah_data['New cases'].shift(1)

mah_data = mah_data.dropna().reset_index(drop=True)
print(f"After feature engineering: {len(mah_data)} rows")

# -------------------------------------------------------------------
# STEP 4: Train/Test Split
# -------------------------------------------------------------------
features = ['cases_lag1', 'cases_lag7', 'cases_lag14', 'new_cases_lag1']
X = mah_data[features]
y = mah_data['Total Confirmed cases']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # preserve time order
)

# -------------------------------------------------------------------
# STEP 5: Train Models
# -------------------------------------------------------------------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

print("\n LINEAR REGRESSION:")
print(f"MAE: {lr_mae:.2f}")
print(f"RMSE: {lr_rmse:.2f}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print("\n RANDOM FOREST:")
print(f"MAE: {rf_mae:.2f}")
print(f"RMSE: {rf_rmse:.2f}")

print("\n Feature Importance (Random Forest):")
for feat, imp in zip(X.columns, rf_model.feature_importances_):
    print(f"{feat}: {imp:.3f}")

# -------------------------------------------------------------------
# STEP 6: Predictions & Visualization (Best Model: Linear Regression)
# -------------------------------------------------------------------
predictions = lr_model.predict(X_test)
actual = y_test.values

plt.figure(figsize=(12, 6))
plt.plot(range(len(actual)), actual, label='Actual', marker='o')
plt.plot(range(len(predictions)), predictions, label='Predicted', marker='x', linestyle='--')
plt.title('Maharashtra COVID Cases: Actual vs Predicted (Linear Regression)')
plt.xlabel('Days')
plt.ylabel('Total Cases')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

mape = np.mean(np.abs((actual - predictions) / actual)) * 100
print(f"\n Prediction Accuracy: {100 - mape:.2f}%")
print(f"Average error: ±{lr_mae:.0f} cases")
