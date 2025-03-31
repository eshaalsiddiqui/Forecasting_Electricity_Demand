import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
df = pd.read_csv('ontario_electricity_demand.csv')

# Select feature and target (same as other models)
X = df[['hour']]
y = df['hourly_demand']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=56)

# Fit the Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=156)
model_rf.fit(X_train, y_train)

# Make predictions
y_test_pred = model_rf.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Random Forest Regressor Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Random Forest: Actual vs Predicted Hourly Demand")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
plt.grid(True)
plt.show()
