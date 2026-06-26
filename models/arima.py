import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# Preprocessing
gas = pd.read_csv("../building_data/gas_cleaned.csv")
panther_cols = [c for c in gas.columns if c.startswith("Panther_")]
gas["timestamp"] = pd.to_datetime(gas["timestamp"])
total = (gas[["timestamp"] + panther_cols]
           .set_index("timestamp")
           .sum(axis=1, min_count=1)
           .interpolate().ffill().bfill())
total.name = "total_demand"

total_diff = total.diff().dropna()

total = total.asfreq("H")

adf_result = adfuller(total)
p_value = adf_result[1]
print(f"ADF p-value: {p_value}")

if p_value < 0.05:
    d = 0
else:
    d = 1

print(f"d value: {d}")

#Train / test split
train = total[:-24]
test = total[-24:]

#Grid search for p, d, q
p = range(0, 5)
q = range(0, 5)

best_aic = float("inf")
best_order = None

for order in itertools.product(p, [d], q):
    try:
        model = ARIMA(train, order=order)
        results = model.fit()

        if results.aic < best_aic:
            best_aic = results.aic
            best_order = order
    except:
        continue

print(f"Best ARIMA order: {best_order}")
print(f"Best AIC: {best_aic}")
# Rolling forecast
history = list(train)
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=best_order)
    model_fit = model.fit()

    yhat = model_fit.forecast()[0]
    predictions.append(yhat)

    history.append(test.iloc[t])

#Evaluation
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

''' Plot '''
plt.figure(figsize=(10,5))
plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, predictions, label="Predicted")
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()


'''
model = ARIMA(train, order=(p,d,q))
model_fit = model.fit()

#Forecast
forecast = model_fit.forecast(steps=24)

#Evaluate
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("MAE:", mae)
print("RMSE:", rmse)
'''