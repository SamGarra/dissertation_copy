import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

''' Preprocessing '''
gas = pd.read_csv("../building_data/gas_cleaned.csv")
panther_cols = [c for c in gas.columns if c.startswith("Panther_")]
gas["timestamp"] = pd.to_datetime(gas["timestamp"])
total = (gas[["timestamp"] + panther_cols]
           .set_index("timestamp")
           .sum(axis=1, min_count=1)
           .interpolate().ffill().bfill())
total.name = "total_demand"
total = total.asfreq("H")

''' Train/test split '''
train = total[:-24]
test = total[-24:]

''' ARIMA Component '''
arima_order = (4, 0, 4)

arima_model = ARIMA(train, order=arima_order)
arima_fit = arima_model.fit()

arima_fitted = arima_fit.fittedvalues
residuals = (train - arima_fitted).dropna()

''' LSTM on residuals '''
res_scaler = MinMaxScaler()
res_scaled_arr = res_scaler.fit_transform(residuals.values.reshape(-1, 1))

def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


SEQ_LEN = 48

X_train, y_train = create_sequences(res_scaled_arr, SEQ_LEN)

''' Model '''
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

''' Rolling forecast '''
history_series = train.copy()
res_history = list(residuals.values)

predictions = []

for t in range(len(test)):

    arima_refit = ARIMA(history_series, order=arima_order).fit()
    arima_pred = arima_refit.forecast(steps=1).iloc[0]

    res_win = np.array(res_history[-SEQ_LEN:]).reshape(-1, 1)
    res_win_scaled = res_scaler.transform(res_win)
    lstm_input = res_win_scaled.reshape(1, SEQ_LEN, 1)

    res_pred_scaled = model.predict(lstm_input, verbose=0).flatten()[0]
    res_pred = res_scaler.inverse_transform([[res_pred_scaled]])[0][0]

    final_pred = arima_pred + res_pred
    predictions.append(final_pred)

    actual = test.iloc[t]
    history_series = pd.concat([
        history_series,
        pd.Series([actual], index=[test.index[t]])
    ])
    true_residual = actual - arima_pred
    res_history.append(true_residual)

''' Evaluation '''
y_test = test.values

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE :", mae)
print("RMSE:", rmse)

''' Plot '''
plt.figure(figsize=(10, 5))
plt.plot(test.index, y_test,    label="Actual")
plt.plot(test.index, predictions, label="Predicted")
plt.legend()
plt.title("ARIMA-LSTM Hybrid Forecast vs Actual")
plt.tight_layout()
plt.show()