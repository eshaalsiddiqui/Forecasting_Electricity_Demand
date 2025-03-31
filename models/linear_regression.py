import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

df = pd.read_csv('ontario_electricity_demand.csv')

X = df[['hour']]  
y = df['hourly_demand'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=56)

lm = LinearRegression()
lm.fit(X_train, y_train)

intercept = lm.intercept_
coefficient = lm.coef_

print("Intercept:", intercept)
print("Coefficient:", coefficient)

coef_df = pd.DataFrame(coefficient, X.columns, columns=['Coefficient'])
print(coef_df)

plt.figure(figsize=(8, 5))
sns.regplot(x=X_train['hour'], y=y_train, order=1, ci=None, scatter_kws={'color': 'r', 's': 9})
plt.xlabel("Hour of Day")
plt.ylabel("Hourly Demand")
plt.title("Linear Regression: Hour vs Demand")
plt.xlim(0, 25)
plt.ylim(ymin=0)
plt.grid(True)
plt.show()

predictions = lm.predict(X_test)
print("Predicted demand:", predictions)


plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Hourly Demand")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
plt.grid(True)
plt.show()

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

