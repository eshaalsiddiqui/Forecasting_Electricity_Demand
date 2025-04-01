import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('ontario_electricity_demand.csv')
df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['hour'].astype(int)
df['hourly_average_price'] = df['hourly_average_price'].astype(float)

features = ['hour', 'month', 'dayofweek', 'hourly_average_price']
X = df[features]
y = df['hourly_demand']

split_index = int(len(df) * 0.6)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

lm = LinearRegression()
lm.fit(X_train, y_train)

print("Intercept:", lm.intercept_)
coef_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coef_df)

predictions = lm.predict(X_test)
print("Predicted demand:", predictions)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Hourly Demand")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         '--', color='gray')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(x=range(100), y=y_test.values[:100], label='Actual')
sns.lineplot(x=range(100), y=predictions[:100], label='Predicted')
plt.title("Linear Regression: Actual vs Predicted (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Demand (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)
