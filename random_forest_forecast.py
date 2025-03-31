import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
df = pd.read_csv("ontario_electricity_demand.csv")

# Parse date and create time-based features
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['hour'].astype(int)

features = ['hour', 'month', 'dayofweek', 'hourly_average_price']
target = 'hourly_demand'

X = df[features]
y = df[target]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156)

# Fit baseline Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=156)

model_rf.fit(X_train, y_train)

# Predict and evaluate
preds_test = model_rf.predict(X_test)
print("Random Forest Regressor - Baseline Performance:")
print("MAE:", mean_absolute_error(y_test, preds_test))
print("MSE:", mean_squared_error(y_test, preds_test))
print("R² Score:", r2_score(y_test, preds_test))

# Hyperparameter tuning
params = {
    'max_depth': np.arange(5, 30, 5),
    'n_estimators': np.arange(50, 210, 50)
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=156),
                           param_grid=params,
                           cv=5,
                           scoring='neg_mean_absolute_error',
                           n_jobs=-1,
                           verbose=1,
                           return_train_score=True)

grid_search.fit(X_train, y_train)

print("\nBest Hyperparameters:", grid_search.best_params_)

# Evaluate tuned model
clf_best = grid_search.best_estimator_
y_test_pred = clf_best.predict(X_test)

print("\nRandom Forest Regressor - Tuned Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R² Score:", r2_score(y_test, y_test_pred))

import matplotlib.pyplot as plt
import seaborn as sns

# Plot predicted vs actual demand (first 100 samples)
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(100), y=y_test.values[:100], label='Actual')
sns.lineplot(x=range(100), y=y_test_pred[:100], label='Predicted')
plt.title("Electricity Demand: Actual vs Predicted (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Demand (kWh)")
plt.legend()
plt.tight_layout()

# Show the graph in your VS Code terminal window
plt.show()
