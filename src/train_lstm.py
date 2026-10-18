import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from prepare_data import prepare_data

X_train, X_test, y_train, y_test, test_df = prepare_data()

test_timestamps = pd.to_datetime(test_df["measured_at"])

np.random.seed(42)
tf.random.set_seed(42)

lookback = 36
epochs = 50
batch_size = 64


# creating sequences ending at each forecast-origin row
def create_sequences(X, y, lookback):
    X_sequences = []
    y_targets = []

    for i in range(lookback - 1, min(len(X), len(y))):
        X_sequences.append(X[i - lookback + 1 : i + 1])
        y_targets.append(y[i])

    return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_targets, dtype=np.float32)

feature_columns = list(X_train.columns)

# converting to numpy arrays
X_train = X_train.to_numpy(dtype=np.float32)
X_test = X_test.to_numpy(dtype=np.float32)
y_train = y_train.to_numpy(dtype=np.float32).reshape(-1,1)
y_test = y_test.to_numpy(dtype=np.float32).reshape(-1,1)

# splitting training/validation data for lstm
split_index = int(len(X_train)*0.85)
X_train_fit = X_train[:split_index]
X_val = X_train[split_index:]

y_train_fit = y_train[:split_index]
y_val = y_train[split_index:]

# fitting scalers on training only
X_scaler = StandardScaler()
X_train_fit = X_scaler.fit_transform(X_train_fit).astype(np.float32)
X_val = X_scaler.transform(X_val).astype(np.float32)
X_test = X_scaler.transform(X_test).astype(np.float32)

y_scaler = StandardScaler()
y_train_fit = y_scaler.fit_transform(y_train_fit).astype(np.float32)
y_val = y_scaler.transform(y_val).astype(np.float32)
y_test_scaled = y_scaler.transform(y_test).astype(np.float32)


# training sequences
X_train_seq, y_train_seq = create_sequences(
    X_train_fit, y_train_fit, lookback
)


# validation sequences - using preceding training rows as historical context
X_val_with_context = np.concatenate(
    [X_train_fit[-(lookback - 1):], X_val],
    axis=0,
)

y_val_with_context = np.concatenate(
    [y_train_fit[-(lookback - 1):], y_val],
    axis=0,
)

X_val_seq, y_val_seq = create_sequences(
    X_val_with_context, y_val_with_context, lookback,
)

# test sequences - using final training rows as historical context
X_train_full = np.concatenate([X_train_fit, X_val], axis=0)

X_test_with_context = np.concatenate(
    [X_train_full[-(lookback - 1):], X_test],
    axis=0,
)

X_test_seq, _ = create_sequences(
    X_test_with_context,
    np.zeros((len(X_test_with_context), 1), dtype=np.float32),
    lookback,
)
assert len(X_val_seq) == len(X_val), "Validation sequence alignment error"
assert len(X_test_seq) == len(X_test), "Test sequence alignment error"


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(lookback, X_train.shape[1])),
    #tf.keras.layers.LSTM(64, return_sequences=True),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),
])


model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    #loss = tf.keras.losses.Huber(delta=1.0),
    loss = ["mse"],
    metrics = ["mae"],
)


history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (X_val_seq, y_val_seq),
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor= "val_loss",
            patience= 7,
            restore_best_weights= True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor= "val_loss",
            factor= 0.5,
            patience= 3,
            min_lr= 1e-6,
        ),
    ],
    shuffle = False,
    verbose = 1,
)


best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
best_val_loss = float(min(history.history["val_loss"]))


# generating predictions in scaled units
predictions_scaled = model.predict(X_test_seq, verbose=0)

# converting predictions back to the original units
predictions = y_scaler.inverse_transform(
    predictions_scaled
).ravel()

# saving the lstm model
model.save("models/lstm_model.keras")

joblib.dump(
    {
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "lookback": lookback,
        "feature_columns": feature_columns,
    },
    "models/lstm_preprocessing.joblib",
)

# adding LSTM predictions to the existing results file
results_path = Path("data/test_predictions.csv")
results = pd.read_csv(results_path)

# adding/replacing lstm column
results["lstm_predictions"] = pd.to_numeric(predictions)
results.to_csv(results_path, index=False)

print(f"Lookback window: {lookback} time steps"
      f"\nTraining rows: {len(X_train):,}"
      f"\nTesting rows: {len(X_test):,}"
      f"\nFeatures: {X_train.shape[1]}"
    )


# creating visuals
plot_dir = Path("data/plots")

epochs_ran = range(1, len(history.history["loss"]) + 1)

# training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(epochs_ran, history.history["loss"], label="Training Loss")
plt.plot(epochs_ran, history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss - scaled")
#plt.ylabel("Huber Loss - scaled")
plt.title("LSTM Training Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_dir / "lstm_loss_curves.png", dpi=150)
plt.close()

# training and validation mae
plt.figure(figsize=(10, 5))
plt.plot(epochs_ran, history.history["mae"], label="Training MAE")
plt.plot(epochs_ran, history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE - scaled")
plt.title("LSTM Training MAE curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_dir / "lstm_mae_curves.png", dpi=150)
plt.close()

print(f"Training plots saved in: {plot_dir.resolve()}")


# setting up residual analysis
actual = results["actual"]
predicted = results["lstm_predictions"]
residuals = actual - predicted

# 1. actual vs predicted demand
plt.figure(figsize=(10, 5))
plt.scatter(actual, predicted, alpha=0.4, s=10)

lower = min(actual.min(), predicted.min())
upper = max(actual.max(), predicted.max())
plt.plot([lower, upper], [lower, upper], linestyle="--")

plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("LSTM: Actual vs Predicted Demand")
plt.tight_layout()
plt.savefig(plot_dir / "lstm_actual_vs_predicted_demand.png", dpi=150)
plt.close()

# 2. residual distribution
plt.figure(figsize=(10,5))
plt.hist(residuals, bins=50, edgecolor="black")
plt.axvline(0, linestyle="--")
plt.xlabel("Residuals (actual - predicted)")
plt.ylabel("Frequency")
plt.title("LSTM Residual Distribution")
plt.savefig(plot_dir / "lstm_residual_distribution.png", dpi=150)
plt.close()

# 3. residuals over forecast time
plt.figure(figsize=(12, 5))
plt.plot(test_timestamps, residuals, linewidth=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Date/Time")
plt.ylabel("Residuals")
plt.title("LSTM Residuals Over Test Period")
plt.tight_layout()
plt.savefig(plot_dir / "lstm_residuals_over_time.png")
plt.close()

# 4. residuals over predicted time
plt.figure(figsize=(10,5))
plt.scatter(predicted, residuals, alpha=0.4, s=10)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Demand")
plt.ylabel("Residuals")
plt.title("LSTM Residuals vs Predicted Demand")
plt.tight_layout()
plt.savefig(plot_dir / "lstm_residuals_vs_predicted_time.png", dpi=150)
plt.close()


print(f"Residual plots saved in: {plot_dir.resolve()}"
      f"\nResidual mean: {residuals.mean():,.2f}"
      f"\nResidual standard deviation: {residuals.std():,.2f}"
      )

print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.5f}")