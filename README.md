# EXPERIMENT 6: EXPONENTIAL SMOOTHING METHODS

**Name:** RAJA RITHIKA M

**Reg. No.:** 2305001029

## 1. Aim

To implement and compare Simple Exponential Smoothing, Holt's Linear Trend Method, and Holt-Winters Seasonal Method for time-series forecasting.

## 2. Software Required

- Python 3
- Google Colab or Jupyter Notebook
- Pandas
- Matplotlib
- Statsmodels

## 3. Procedure

1. Import the required libraries.
2. Create sales data.
3. Apply Simple Exponential Smoothing.
4. Apply Holt's method for trend data.
5. Apply Holt-Winters method for seasonal data.
6. Forecast future values using each method.
7. Plot and compare the results.

## 4. Program

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)

# PART A: Simple Exponential Smoothing
sales = [100, 120, 110, 115, 105, 125, 118, 122, 110, 119]
data = pd.Series(sales)

model1 = SimpleExpSmoothing(data)
fit1 = model1.fit(smoothing_level=0.3, optimized=False)
forecast1 = fit1.forecast(3)

print("Simple Exponential Smoothing")
print("Forecast:")
print(forecast1)

plt.figure(figsize=(8, 4))
plt.plot(data, marker="o", label="Actual")
plt.plot(fit1.fittedvalues, marker="o", label="Fitted")
plt.plot(range(len(data), len(data) + 3), forecast1,
         marker="s", linestyle="--", label="Forecast")
plt.title("Simple Exponential Smoothing")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# PART B: Holt's Linear Trend Method
sales_trend = [100, 105, 112, 118, 125, 130, 138, 145, 152, 160]
data_trend = pd.Series(sales_trend)

model2 = Holt(data_trend)
fit2 = model2.fit()
forecast2 = fit2.forecast(3)

print("\nHolt's Method")
print("Forecast:")
print(forecast2)

plt.figure(figsize=(8, 4))
plt.plot(data_trend, marker="o", label="Actual")
plt.plot(fit2.fittedvalues, marker="o", label="Fitted")
plt.plot(range(len(data_trend), len(data_trend) + 3), forecast2,
         marker="s", linestyle="--", label="Forecast")
plt.title("Holt's Linear Trend Method")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# PART C: Holt-Winters Seasonal Method
seasonal_sales = [
    100, 130, 160, 120,
    110, 145, 175, 130,
    120, 155, 190, 140,
    130, 165, 205, 150
]
data_seasonal = pd.Series(seasonal_sales)

model3 = ExponentialSmoothing(
    data_seasonal,
    trend="add",
    seasonal="add",
    seasonal_periods=4
)
fit3 = model3.fit()
forecast3 = fit3.forecast(4)

print("\nHolt-Winters Method")
print("Forecast:")
print(forecast3)

plt.figure(figsize=(8, 4))
plt.plot(data_seasonal, marker="o", label="Actual")
plt.plot(fit3.fittedvalues, marker="o", label="Fitted")
plt.plot(range(len(data_seasonal), len(data_seasonal) + 4), forecast3,
         marker="s", linestyle="--", label="Forecast")
plt.title("Holt-Winters Seasonal Method")
plt.xlabel("Quarter")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

print("\nAll three exponential smoothing methods completed successfully.")
```

## 5. Expected Output

- Simple Exponential Smoothing forecast for the next three periods.
  <img width="1030" height="632" alt="image" src="https://github.com/user-attachments/assets/bc674c9c-8293-4a2a-a818-583ace8e8200" />

- Holt's Method forecast for the next three periods.
  <img width="956" height="645" alt="image" src="https://github.com/user-attachments/assets/0e5bf6e3-5218-4c74-b904-c9f1f7176708" />

- Holt-Winters forecast for the next four quarters.
<img width="1027" height="716" alt="image" src="https://github.com/user-attachments/assets/41f6b6ff-b0cc-4beb-9d7c-c351e9d335fa" />


## 6. Observation

- **Simple Exponential Smoothing:** Suitable for data without a clear trend or seasonality.
- **Holt's Method:** Suitable for data containing a trend.
- **Holt-Winters Method:** Suitable for data containing both trend and seasonality.

## 7. Result

Thus, Simple Exponential Smoothing, Holt's Linear Trend Method, and Holt-Winters Seasonal Method were implemented successfully to forecast time-series data.
