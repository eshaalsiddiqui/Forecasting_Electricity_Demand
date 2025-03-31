import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ontario_electricity_demand.csv")

# Parse date and extract features
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['hour'].astype(int)

# Select features and target
features = ['hour', 'month', 'dayofweek', 'hourly_average_price']
target = 'hourly_demand'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156)

# Train baseline Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=156)
model_rf.fit(X_train, y_train)

# Predict on test set
y_test_pred = model_rf.predict(X_test)

# Evaluation metrics
print("Random Forest Regressor - Performance:")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R² Score:", r2_score(y_test, y_test_pred))

# Plot actual vs predicted demand (first 100 points)
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(100), y=y_test.values[:100], label='Actual')
sns.lineplot(x=range(100), y=y_test_pred[:100], label='Predicted')
plt.title("Electricity Demand: Actual vs Predicted (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Demand (kWh)")
plt.legend()
plt.tight_layout()
plt.show()
