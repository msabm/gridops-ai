import joblib
import tensorflow as tf
<<<<<<< HEAD
=======
import pandas as pd

from src.feature_engineering import (
    load_data, create_energy_features,
    create_time_features, create_lag_features,
    create_rolling_features,
)
>>>>>>> feat/fastapi-integration

from pathlib import Path

model_path = Path("models/lstm_final_top10_model.keras")
preprocessing_path = Path("models/lstm_final_top10_preprocessing.joblib")

def load_model_artifacts():
    model = tf.keras.models.load_model(model_path)
    preprocessing = joblib.load(preprocessing_path)

<<<<<<< HEAD
    return model, preprocessing
=======
    return model, preprocessing

def prepare_latest_features(feature_columns, lookback):
    df = load_data()
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    df = (
        df.drop_duplicates(subset=["measured_at"])
        .sort_values("measured_at").reset_index(drop=True)
    )

    df = create_time_features(df)
    df = create_energy_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    if len(df) < lookback:
        raise ValueError(
            f"Not enough feature rows to create input sequence of {lookback} steps"
        )

    latest_rows = df.tail(lookback).copy()

    # ensuring the latest rows are in chronological order
    intervals = latest_rows["measured_at"].diff().dropna()

    if not (intervals == intervals.iloc[0]).all():
        raise ValueError(
            "Latest rows have irregular time intervals"
        )
    
    return latest_rows

def generate_forecast(model, preprocessing):
    X_scaler = preprocessing["X_scaler"]
    y_scaler = preprocessing["y_scaler"]
    lookback = preprocessing["lookback"]
    feature_columns = preprocessing["feature_columns"]

    latest_rows = prepare_latest_features(feature_columns, lookback)

    forecast_origin = latest_rows["measured_at"].iloc[-1]
    target_timestamp = forecast_origin + pd.Timedelta(hours=1)
    
    X_latest = latest_rows[feature_columns]
    X_scaled = X_scaler.transform(X_latest)

    # reshaping for LSTM input
    X_sequence = X_scaled.reshape(
        1, lookback, len(feature_columns) 
        )

    scaled_prediction = model.predict(X_sequence, verbose=0)

    prediction = y_scaler.inverse_transform(
        scaled_prediction.reshape(-1, 1)
    )[0, 0]

    return {
        "forecast_origin": forecast_origin,
        "target_timestamp": target_timestamp,
        "predicted_total_power_consumption": float(prediction),
        "model": "LSTM",
        "features": len(feature_columns),
        "lookback_steps": lookback,
    }
>>>>>>> feat/fastapi-integration
