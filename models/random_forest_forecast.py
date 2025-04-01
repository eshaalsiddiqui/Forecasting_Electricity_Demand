import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and parse data
df = pd.read_csv('ontario_electricity_demand.csv')
df['date'] = pd.to_datetime(df['date'])

# Sort chronologically
df = df.sort_values(by=['date', 'hour']).reset_index(drop=True)

# Feature Engineering
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['hour'].astype(int)
df['hourly_average_price'] = df['hourly_average_price'].astype(float)

# Lag features
df['demand_t-1'] = df['hourly_demand'].shift(1)
df['demand_t-2'] = df['hourly_demand'].shift(2)
df['price_t-1'] = df['hourly_average_price'].shift(1)

# Rolling average of previous 3 hours
df['rolling_mean_3'] = df['hourly_demand'].rolling(3).mean().shift(1)

# Drop rows with NaN (from shift & rolling)
df.dropna(inplace=True)

# Feature set
features = ['hour', 'month', 'dayofweek', 'hourly_average_price',
            'demand_t-1', 'demand_t-2', 'price_t-1', 'rolling_mean_3']

X = df[features]
y = df['hourly_demand']
dates = df['date']

# Chronological train-test split (60% train, 40% test)
split_index = int(len(df) * 0.6)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
dates_train, dates_test = dates.iloc[:split_index], dates.iloc[split_index:]

# Train model
model_rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=156)
model_rf.fit(X_train, y_train)

# Predict
y_test_pred = model_rf.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Random Forest Regressor Performance")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Plot: Actual vs Predicted (scatter)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Random Forest: Actual vs Predicted Hourly Demand")
plt.grid(True)
plt.show()

# Plot: First 100 time-ordered predictions
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(100), y=y_test.values[:100], label='Actual')
sns.lineplot(x=range(100), y=y_test_pred[:100], label='Predicted')
plt.title("Random Forest: Actual vs Predicted (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Demand (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
