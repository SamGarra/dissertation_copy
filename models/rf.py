import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAGG")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)

''' Preprocessing '''
#Load data
gas = pd.read_csv("../building_data/gas_cleaned.csv")
weather = pd.read_csv("../building_data/weather.csv")

gas["timestamp"] = pd.to_datetime(gas["timestamp"])
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

panther_cols = [c for c in gas.columns if c.startswith("Panther_")]

#Total demand
total = (gas[["timestamp"] + panther_cols]
           .set_index("timestamp")
           .sum(axis=1, min_count=1))
total.name = "total_demand"

#Weather preprocess
weather = weather[weather["site_id"] == "Panther"]
weather = weather.set_index("timestamp").drop(columns=["site_id"])

df = total.to_frame(name="total_demand").join(weather, how="inner")
df = df.resample("H").mean()
df = df.interpolate().ffill().bfill()


''' Feature Engineering '''
#Short term lag
for lag in range(1, 25):
    df[f"lag_{lag}"] = df["total_demand"].shift(lag)

#Long term lag
for log in [24, 48, 168]:
    df[f"lag_{lag}"] = df["total_demand"].shift(lag)

#Time features
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek

#Rolling stats
df["rolling_mean_24"] = df["total_demand"].rolling(24).mean()
df["rolling_std_24"] = df["total_demand"].rolling(24).std()

df = df.dropna()

''' Train Test Split'''
train = df.iloc[:-72]
test = df.iloc[-72:]

X_train = train.drop("total_demand", axis=1)
y_train = train["total_demand"]

X_test = test.drop("total_demand", axis=1)
y_test = test["total_demand"]

''' Training '''
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

''' Forecast and Evaluate '''
#Forecast
history = train.copy()
predictions = []

for t in range(len(test)):
    X_hist = history.drop("total_demand", axis=1)
    y_hist = history["total_demand"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_hist, y_hist)

    X_next = test.drop("total_demand", axis=1).iloc[t:t+1]
    yhat = model.predict(X_next)[0]

    predictions.append(yhat)

    history = pd.concat([history, test.iloc[t:t + 1]])

#Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE: ", mae)
print("RMSE", rmse)

''' Plot'''
plt.figure(figsize=(10,5))
plt.plot(test.index, y_test.values, label="Actual")
plt.plot(test.index, predictions, label="Predicted")
plt.legend()
plt.title("Random Forest Forecast vs Actual")
plt.show()

''' Plot feature importance '''
importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 8))
importances.plot(kind="barh")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()