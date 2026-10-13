import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from prepare_data import prepare_data

X_train, X_test, y_train, y_test, test_df = prepare_data()

np.random_seed(42)
tf.random.set_seed(42)

lookback = 144
epochs = 50
batch_size = 64


# creating sequences ending at each forecast-origin row
def create_sequences(X, y, start, end, lookback):
    X_sequences = []
    y_targets = []

    for i in range(start, end):
        X_sequences.append(X[i - lookback + 1: i + 1])
        y_targets.append(y[i])

    return np.asarray(X_sequences), np.asarray(y_targets)


# converting to numpy arrays
X_train = X_train.to_numpy(dtype=np.float32)
X_test = X_test.to_numpy(dtype=np.float32)
y_train = y_train.to_numpy(dtype=np.float32).reshape(-1,1)
y_test = y_test.to_numpy(dtype=np.float32).reshape(-1,1)


# fitting scalers on training only
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train).astype(np.float32)
X_test = X_scaler.fit_transform(X_test).astype(np.float32)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train).astype(np.float32)





print(f"Lookback window: {lookback} time steps"
      f"\nTraining rows: {len(X_train):,}"
      f"\nTesting rows: {len(X_test):,}"
      f"\nFeatures: {X_train.shape[1]}"
      )

