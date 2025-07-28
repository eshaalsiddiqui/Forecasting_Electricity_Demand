## Project Overview

The goal of this project is to apply various machine learning regression techniques to forecast electricity demand in Ontario. This project serves as a practical implementation of data-driven forecasting using real-world data and contributes to understanding model performance across different algorithms.

---

## Problem Statement & Justification

Accurately forecasting electricity demand is critical for grid reliability, cost optimization, and energy planning. By modeling electricity demand patterns, utility providers and policymakers can make informed decisions to avoid overproduction or shortages. This project evaluates multiple regression algorithms to determine which yields the most accurate forecasts using historical electricity demand data.

---

## Dataset Description

- **Source:** Historical electricity demand data from Ontario (file: `ontario_electricity_demand.csv`)
- **Rows:** Each row represents one hourly interval
- **Columns:**
  - `date`: Date of observation (YYYY-MM-DD)
  - `hour`: Hour of the day (1 to 24)
  - `hourly_demand`: Electricity demand in kilowatts (kW)
  - `hourly_average_price`: Average electricity price (in CAD) for the hour
- **Data Types:**
  - `date`: String or datetime
  - `hour`: Integer
  - `hourly_demand`: Integer
  - `hourly_average_price`: Float

The dataset is designed for time-series forecasting. It captures hourly electricity demand and corresponding price, making it suitable for modeling consumption trends and predicting future demand using supervised learning techniques. Preprocessing steps included handling date-time features and normalizing/standardizing features where necessary.

---

## Machine Learning Approach

The following regression models were implemented and compared:

- **Linear Regression** (`linear_regression.py`)
- **Support Vector Regression (SVR)** (`support_vector_regression.py`)
- **Random Forest Regressor** (`random_forest_forecast.py`)
- **Gradient Boosting Regressor** (`gradient_boosting.py`)

Each model was trained on the same set of features to predict future electricity demand. Model evaluation was based on metrics such as MAE, MSE and R² scores.
