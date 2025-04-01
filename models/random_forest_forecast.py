import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and parse data
df = pd.read_csv('ontario_electricity_demand.csv')
df['date'] = pd.to_datetime(df['date'])

# Sort by time
df = df.sort_values(by=['date', 'hour']).reset_index(drop=True)

# Feature Engineering: Lags & Rolling Averages
df['demand_t-1'] = df['hourly_demand'].shift(1)
df['demand_t-2'] = df['hourly_demand'].shift(2)
df['price_t-1'] = df['hourly_average_price'].shift(1)
df['rolling_mean_3'] = df['hourly_demand'].rolling(3).mean().shift(1)

# Drop rows with NaN caused by shifting/rolling
df.dropna(inplace=True)

# Define features and target
features = ['hour', 'demand_t-1', 'demand_t-2', 'price_t-1', 'rolling_mean_3']
X = df[features]
y = df['hourly_demand']

# Time-based split (no shuffle)
split_index = int(len(df) * 0.6)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Random Forest Model
model_rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=156
)

model_rf.fit(X_train, y_train)
preds_test = model_rf.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, preds_test)
mse = mean_squared_error(y_test, preds_test)
r2 = r2_score(y_test, preds_test)

print("Random Forest Model Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Random Forest: Actual vs Predicted")
plt.grid(True)
plt.show()


# Plot: Line graph comparison of actual vs predicted demand
plt.figure(figsize=(10, 6))

# Sort by index so the lines line up
sns.lineplot(x=range(100), y=y_test.values[:100], label='Actual', linewidth=2)
sns.lineplot(x=range(100), y=y_test_pred[:100], label='Predicted', linewidth=2)

plt.title("Random Forest: Actual vs Predicted Electricity Demand (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Demand (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# First test sample
print("First test sample:")
print("Date:", dates_test.values[0])
print("Actual:", y_test.values[0])
print("Predicted:", y_test_pred[0])

# 100th test sample
print("100th test sample:")
print("Date:", dates_test.values[99])
print("Actual:", y_test.values[99])
print("Predicted:", y_test_pred[99])
