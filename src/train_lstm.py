import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from prepare_data import prepare_data

X_train, X_test, y_train, y_test, test_df = prepare_data()

np.random.seed(42)
tf.random.set_seed(42)

lookback = 144
epochs = 50
batch_size = 64


# creating sequences ending at each forecast-origin row
def create_sequences(X, y, lookback):
    X_sequences = []
    y_targets = []

    for i in range(lookback - 1, len(X)):
        X_sequences.append(X[i - lookback + 1: i + 1])
        y_targets.append(y[i])

    return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_targets, dtype=np.float32)

feature_columns = list(X_train.columns)

# converting to numpy arrays
X_train = X_train.to_numpy(dtype=np.float32)
X_test = X_test.to_numpy(dtype=np.float32)
y_train = y_train.to_numpy(dtype=np.float32).reshape(-1,1)
y_test = y_test.to_numpy(dtype=np.float32).reshape(-1,1)


# fitting scalers on training only
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train).astype(np.float32)
X_test = X_scaler.transform(X_test).astype(np.float32)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train).astype(np.float32)
y_test_scaled = y_scaler.transform(y_test).astype(np.float32)


X_train_seq, y_train_seq = create_sequences(
    X_train, y_train, lookback
    )


# adding historical context from end of training to test inputs
X_test_with_context = np.concatenate(
    [X_train[-(lookback - 1):], X_test],
    axis = 0,
)


X_test_seq, _ = create_sequences(
    X_test_with_context,
    np.zeros((len(X_test_with_context), 1), dtype=np.float32),
    lookback,
)

assert len(X_test_seq) == len(X_test), "Test sequence alignment error"


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(lookback, X_train.shape[1])),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),
])


model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = "mse",
    metrics = ["mae"],
)


model.fit(
    X_train_seq,
    y_train_seq,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.15,
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

# adding lightgbm predictions to the existing results file
results_path = Path("data/test_predictions.csv")
results = pd.read_csv(results_path)

# adding/replacing lightgbm column
results["lstm_predictions"] = pd.to_numeric(predictions)
results.to_csv(results_path, index=False)

print(f"Lookback window: {lookback} time steps"
      f"\nTraining rows: {len(X_train):,}"
      f"\nTesting rows: {len(X_test):,}"
      f"\nFeatures: {X_train.shape[1]}"
      )
