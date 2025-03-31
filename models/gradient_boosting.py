import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('ontario_electricity_demand.csv')

X = df[['hour']]  
y = df['hourly_demand']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=56)

model_gb = GradientBoostingRegressor(
    n_estimators=30,
    learning_rate=0.01,
    max_features=1,  
    max_depth=5,
    random_state=156
)

model_gb.fit(X_train, y_train)

preds_test = model_gb.predict(X_test)

mse = mean_squared_error(y_test, preds_test)
mae = mean_absolute_error(y_test, preds_test)
r2 = r2_score(y_test, preds_test)

print("Gradient Boosting Model Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_test, alpha=0.7)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Gradient Boosting: Actual vs Predicted Hourly Demand")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
plt.grid(True)
plt.show()
