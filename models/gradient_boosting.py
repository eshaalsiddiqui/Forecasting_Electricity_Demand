import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(ontario_electricity_demand.csv)

# Parse datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by time
df = df.sort_values(by=['date', 'hour']).reset_index(drop=True)

# Create lag features
df['demand_t-1'] = df['hourly_demand'].shift(1)
df['demand_t-2'] = df['hourly_demand'].shift(2)
df['price_t-1'] = df['hourly_average_price'].shift(1)

# Rolling average
df['rolling_mean_3'] = df['hourly_demand'].rolling(3).mean().shift(1)

# Drop rows with NaN from shifting
df.dropna(inplace=True)

# Define features and target
features = ['hour', 'demand_t-1', 'demand_t-2', 'price_t-1', 'rolling_mean_3']
X = df[features]
y = df['hourly_demand']

# Use time-based split (not random)
split_index = int(len(df) * 0.6)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Gradient Boosting Model
model_gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=156
)

model_gb.fit(X_train, y_train)
preds_test = model_gb.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, preds_test)
mse = mean_squared_error(y_test, preds_test)
r2 = r2_score(y_test, preds_test)

print("Gradient Boosting Model Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Gradient Boosting: Actual vs Predicted")
plt.grid(True)
plt.show()
