import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

''' Preprocessing '''
gas = pd.read_csv("../building_data/gas_cleaned.csv")
weather = pd.read_csv("../building_data/weather.csv")

gas["timestamp"] = pd.to_datetime(gas["timestamp"])
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

panther_cols = [c for c in gas.columns if c.startswith("Panther_")]

total = gas[["timestamp"] + panther_cols].set_index("timestamp").sum(axis=1, min_count=1)
total.name = "total_demand"

weather = weather[weather["site_id"] == "Panther"].set_index("timestamp").drop(columns=["site_id"])

df = total.to_frame().join(weather, how="inner").resample("H").mean()
df = df.interpolate().ffill().bfill()

''' Train/test split '''
train = df.iloc[:-24].copy()
test = df.iloc[-24:].copy()

''' Scaling '''
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_scaled = feature_scaler.fit_transform(train)
test_scaled = feature_scaler.transform(test)

target_scaler.fit(train[["total_demand"]])

''' Sequences '''
SEQ_LEN = 48


def create_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled, SEQ_LEN)

''' Model '''
n_features = X_train.shape[2]

model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, n_features)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

''' Training '''
history_fit = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

''' Loss plot '''
plt.figure(figsize=(10, 4))
plt.plot(history_fit.history["loss"], label="Train loss")
plt.plot(history_fit.history["val_loss"], label="Val loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.tight_layout()
plt.show()

print("Epochs run:", len(history_fit.history["loss"]))
print("Best val_loss:", min(history_fit.history["val_loss"]))

''' Prediction '''
full_scaled = feature_scaler.transform(df)
preds_scaled = []

for t in range(len(test)):
    start = len(train) - SEQ_LEN + t
    seq = full_scaled[start:start + SEQ_LEN].reshape(1, SEQ_LEN, n_features)
    yhat = model.predict(seq, verbose=0)[0][0]
    preds_scaled.append(yhat)

predictions = target_scaler.inverse_transform(
    np.array(preds_scaled).reshape(-1, 1)
).flatten()

''' Permutation Importance '''
def permutation_importance(model, X, y_scaled, target_scaler, feature_names, n_repeats=5):
    baseline_preds = model.predict(X, verbose=0).flatten()
    baseline_actual = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    baseline_pred_r = target_scaler.inverse_transform(baseline_preds.reshape(-1, 1)).flatten()
    baseline_mae = mean_absolute_error(baseline_actual, baseline_pred_r)

    importances = []
    for i in range(X.shape[2]):
        maes = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            idx = np.random.permutation(len(X_perm))
            X_perm[:, :, i] = X_perm[idx, :, i]

            perm_preds = model.predict(X_perm, verbose=0).flatten()
            perm_preds_r = target_scaler.inverse_transform(perm_preds.reshape(-1, 1)).flatten()
            maes.append(mean_absolute_error(baseline_actual, perm_preds_r))

        importances.append(np.mean(maes) - baseline_mae)

    return importances


#Random subsample of training data to avoid temporal bias
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), size=2000, replace=False)
X_imp = X_train[idx]
y_imp = y_train[idx]

feature_names = df.columns.tolist()
importances = permutation_importance(model, X_imp, y_imp, target_scaler, feature_names)

sorted_idx = np.argsort(importances)
plt.figure(figsize=(10, 6))
plt.barh(np.array(feature_names)[sorted_idx], np.array(importances)[sorted_idx])
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
plt.title("Permutation Feature Importance — LSTM")
plt.xlabel("Mean increase in MAE when feature is shuffled")
plt.tight_layout()
plt.show()

''' Evaluate '''
y_test = test["total_demand"].values

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

''' Forecast plot '''
plt.figure(figsize=(10, 5))
plt.plot(test.index, y_test, label="Actual")
plt.plot(test.index, predictions, label="Predicted")
plt.legend()
plt.title("LSTM Forecast vs Actual")
plt.tight_layout()
plt.show()
