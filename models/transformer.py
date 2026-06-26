import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models
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


''' Positional Encoding '''
def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles, dtype=tf.float32)


''' Transformer Block '''
def transformer_block(x, num_heads=4, key_dim=32, ff_dim=256, dropout=0.1):
    d_model = x.shape[-1]

    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x_norm, x_norm)
    x = layers.Add()([x, attn])

    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(ff_dim, activation="gelu")(x_norm)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model)(ff)
    return layers.Add()([x, ff])


''' Model '''
D_MODEL = 128
n_features = X_train.shape[2]

inputs = layers.Input(shape=(SEQ_LEN, n_features))

x = layers.Dense(D_MODEL)(inputs)
x = x + positional_encoding(SEQ_LEN, D_MODEL)

x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=256, dropout=0.1)
x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=256, dropout=0.1)
x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=256, dropout=0.1)

x = layers.LayerNormalization(epsilon=1e-6)(x)
x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(128, activation="gelu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="gelu")(x)

outputs = layers.Dense(1)(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.Huber(),
    metrics=["mae"]
)
model.summary()


''' Training '''
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=7, restore_best_weights=True
)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)


''' Prediction '''
full_scaled = feature_scaler.transform(df)

predictions_scaled = []
for t in range(len(test)):
    start = len(train) - SEQ_LEN + t
    seq = full_scaled[start:start + SEQ_LEN].reshape(1, SEQ_LEN, n_features)
    yhat = model.predict(seq, verbose=0)[0][0]
    predictions_scaled.append(yhat)

predictions = target_scaler.inverse_transform(
    np.array(predictions_scaled).reshape(-1, 1)
).flatten()


''' Evaluate '''
y_test = test["total_demand"].values

mae  = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


''' Plot '''
plt.figure(figsize=(10, 5))
plt.plot(test.index, y_test,      label="Actual")
plt.plot(test.index, predictions, label="Predicted")
plt.legend()
plt.title("Transformer Forecast vs Actual")
plt.tight_layout()
plt.show()
