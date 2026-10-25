import joblib
import pandas as pd
import tensorflow as tf

from pathlib import Path

from feature_engineering import (
    load_data, create_energy_features,
    create_time_features, create_lag_features,
    create_rolling_features,
)

# file paths
model_path = Path("models/lstm_model.keras")
preprocessing_path = Path("models/lstm_preprocessing.joblib")


def prepare_latest_features(feature_columns):
    df = load_data()
    df["measured_at"] = pd.datetime(df["measured_at"])
    df = (
        df.drop_duplicates(subset=["measured_at"])
        .sort_values("measured_at").reset_index(drop=True)
    )

    # reusing same features as training
    df = create_rolling_features
    df = create_time_features
    df = create_energy_features
    df = create_lag_features

    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    if len(df)


def load_artifacts():
    model = tf.keras.models.load_model(model_path)
    preprocessing = joblib.load(preprocessing_path)

    return model, preprocessing

def create_sequences(X, lookback):
    return np.asarray([
        X[i - lookback +1 : i + 1]
        for i in range(lookback - 1, len(X))
        ], dtype=np.float32,
    )


def predict_test_set():
    model, preprocessing = load_artifacts()

    X_scaler = preprocessing["X_scaler"]
    y_scaler = preprocessing["y_scaler"]
    lookback = preprocessing["lookback"]
    feature_columns = preprocessing["feature_columns"]

    X_train, X_test, _, _, test_df = prepare_data()

    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]

    # applying saved scalers
    X_train_scaled = X_scaler.transform(X_train).astype(np.float32)
    X_test_scaled = X_scaler.transform(X_test).astype(np.float32)

    # matching context used to construct sequences in training
    X_test_with_context = np.concatenate(
        [X_train_scaled[-(lookback - 1):], X_test_scaled],
        axis=0,
    )

    test_sequences = create_sequences(
        X_test_with_context, lookback,
    )

    predictions_scaled = model.predict(test_sequences, verbose=1)

    predictions = y_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).ravel()

    results_path = Path("data/test_predictions.csv")
    results = pd.read_csv(results_path)

    # adding/replacing lstm inference column
    results["inferenced_lstm_predictions"] = pd.to_numeric(predictions)
    results.to_csv(results_path, index=False)

    print(f"\nPredictions generated: {len(results):,}"
          f"\n{results.head()}"    
    )

if __name__ == "__main__":
    predict_test_set()