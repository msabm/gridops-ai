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
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    df = (
        df.drop_duplicates(subset=["measured_at"])
        .sort_values("measured_at").reset_index(drop=True)
    )

    # reusing same features as training
    df = create_time_features(df)
    df = create_energy_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    if len(df) < 144:
        raise ValueError(
            "Not enough feature rows to create input sequence"
        )
    
    latest_rows = df.tail(144).copy()

    if latest_rows[feature_columns].isna().any().any():
        raise ValueError(
            "Missing values remain in latest sequence"
        )
    
    return latest_rows


def main():
    model = tf.keras.models.load_model(model_path)
    preprocessing = joblib.load(preprocessing_path)
  
    X_scaler = preprocessing["X_scaler"]
    y_scaler = preprocessing["y_scaler"]
    lookback = preprocessing["lookback"]
    feature_columns = preprocessing["feature_columns"]

    if lookback != 144:
        raise ValueError(
            f"Expected 144 step lookback but found {lookback}"
        )

    # preparing the most recent valid input sequence
    latest_rows = prepare_latest_features(feature_columns)

    forecast_origin = latest_rows["measured_at"].iloc[-1]
    target_timestamp = forecast_origin + pd.Timedelta(hours=1)

    X_latest = latest_rows[feature_columns]

    # using same fitted scaler as training
    X_scaled = X_scaler.transform(X_latest)

    # LSTM input shape (samples, timestamps, features)
    X_sequence = X_scaled.reshape(
        1, lookback, len(feature_columns)
    )

    # predicting 1 hour ahead
    scaled_prediction = model.predict(X_sequence, verbose=1)
    prediction = y_scaler.inverse_transform(
        scaled_prediction.reshape(-1, 1)
    )[0, 0]

    # saving in a new file
    results = pd.DataFrame([{
        "forecast_origin": forecast_origin,
        "target_timestamp": target_timestamp,
        "predicted_total_power_consumption": float(prediction),
        "model": "LSTM",
        "lookback_steps":lookback,
    }])

    results.to_csv("data/saved_lstm_prediction_results.csv", index=False)

    print(
        f"\nLast observed timestamp: {forecast_origin}"
        f"\nForecasttarget time:     {target_timestamp}"
        f"\nPredicted total demand:  {prediction:,.2f}"
        f"\nInput sequence:          {lookback}"
        f"\nFeatures:                {len(feature_columns)}"
    )

if __name__ == "__main__":
    main()
