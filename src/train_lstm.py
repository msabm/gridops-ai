import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, root_mean_squared_error,
    mean_absolute_percentage_error
)

from prepare_data import prepare_data


default_lookback = 144
default_epochs = 50
default_batch_size = 32


# creating sequences ending at each forecast-origin row
def create_sequences(X, y, lookback):
    X_sequences = []
    y_targets = []

    for i in range(lookback - 1, min(len(X), len(y))):
        X_sequences.append(X[i - lookback + 1 : i + 1])
        y_targets.append(y[i])

    return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_targets, dtype=np.float32)


def train_lstm(
        selected_features=None, run_name="lstm",
        lookback=default_lookback, epochs=default_epochs,
        batch_size=default_batch_size, save_outputs=True,
        make_plots=True,
):
    # training and evaluating lstm using selected feature subset
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    X_train, X_test, y_train, y_test, test_df = prepare_data(
        selected_features=selected_features
    )

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
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
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

    metrics = {
        "mae": float(mean_absolute_error(y_test.ravel(), predictions)),
        "mse": float(mean_squared_error(y_test.ravel(), predictions)),
        "rmse": float(root_mean_squared_error(y_test.ravel(), predictions)),
        "r2": float(r2_score(y_test.ravel(), predictions)),
        "mape": float(mean_absolute_percentage_error(y_test.ravel(), predictions)),
    }

    print(f"Run: {run_name}")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.5f}")

    # saving the lstm model
    if save_outputs:
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

    print(f"Lookback window: {lookback} time steps"
        f"\nTraining rows: {len(X_train):,}"
        f"\nTesting rows: {len(X_test):,}"
        f"\nFeatures: {X_train.shape[1]}"
        )


    # creating visuals
    
    # Create a dataframe aligned with the forecast target timestamps.
    pattern_df = pd.DataFrame({
        "timestamp": pd.to_datetime(test_df["target_timestamp"]),
        "actual": y_test.ravel(),
        "lstm": predictions,
    })

    # Compare errors during the highest-demand periods.
    peak_threshold = pattern_df["actual"].quantile(0.90)
    peak_df = pattern_df[pattern_df["actual"] >= peak_threshold]

    peak_mae = np.mean(
        np.abs(peak_df["actual"] - peak_df["lstm"])
    )

    print(f"LSTM peak-period MAE: {peak_mae:,.2f}")
    print(
        f"Peak threshold (90th percentile): {peak_threshold:,.2f}"
        f"\nPeak observations: {len(peak_df):,}"
    )

    if make_plots:
        plot_dir = Path("data/plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        # 1. Training and validation curves
        epochs_ran = range(1, len(history.history["loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].plot(
            epochs_ran, history.history["loss"], label="Training Loss"
        )
        axes[0].plot(
            epochs_ran, history.history["val_loss"], label="Validation Loss"
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss (scaled)")
        axes[0].set_title("LSTM Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            epochs_ran, history.history["mae"], label="Training MAE"
        )
        axes[1].plot(
            epochs_ran, history.history["val_mae"], label="Validation MAE"
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (scaled)")
        axes[1].set_title("LSTM Training MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / "lstm_training_curves.png", dpi=150)
        plt.close(fig)

        # 2. Actual versus predicted demand for the first test week
        start = pattern_df["timestamp"].min().normalize()
        end = start + pd.Timedelta(days=7)

        week_df = pattern_df[
            (pattern_df["timestamp"] >= start)
            & (pattern_df["timestamp"] < end)
        ]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            week_df["timestamp"], week_df["actual"],
            label="Actual", linewidth=2
        )
        ax.plot(
            week_df["timestamp"], week_df["lstm"],
            label="LSTM", linewidth=1
        )
        ax.set_xlabel("Target timestamp")
        ax.set_ylabel("Electricity demand")
        ax.set_title("Actual vs Forecast Demand — First Test Week")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "lstm_daily_pattern_week.png", dpi=150)
        plt.close(fig)

        # 3. Average demand profile by hour
        pattern_df["hour"] = pattern_df["timestamp"].dt.hour

        hourly_profile = pattern_df.groupby("hour")[
            ["actual", "lstm"]
        ].mean()

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(
            hourly_profile.index, hourly_profile["actual"],
            marker="o", label="Actual", linewidth=2
        )
        ax.plot(
            hourly_profile.index, hourly_profile["lstm"],
            marker="o", label="LSTM", linewidth=1
        )
        ax.set_xticks(range(24))
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Mean electricity demand")
        ax.set_title("Average Daily Demand Profile — Test Period")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "lstm_average_daily_profile.png", dpi=150)
        plt.close(fig)

        # 4. Residual analysis
        actual = y_test.ravel()
        predicted = predictions
        residuals = actual - predicted
        target_timestamps = pd.to_datetime(test_df["target_timestamp"])

        # Actual versus predicted
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(actual, predicted, alpha=0.4, s=10)

        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        ax.plot([lower, upper], [lower, upper], linestyle="--")

        ax.set_xlabel("Actual demand")
        ax.set_ylabel("Predicted demand")
        ax.set_title("LSTM: Actual vs Predicted Demand")
        fig.tight_layout()
        fig.savefig(
            plot_dir / "lstm_actual_vs_predicted_demand.png", dpi=150
        )
        plt.close(fig)

        # Residual distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(residuals, bins=50, edgecolor="black")
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Residual (actual - predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title("LSTM Residual Distribution")
        fig.tight_layout()
        fig.savefig(
            plot_dir / "lstm_residual_distribution.png", dpi=150
        )
        plt.close(fig)

        # Residuals over forecast time
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(target_timestamps, residuals, linewidth=0.7)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Target timestamp")
        ax.set_ylabel("Residual")
        ax.set_title("LSTM Residuals Over Test Period")
        fig.tight_layout()
        fig.savefig(
            plot_dir / "lstm_residuals_over_time.png", dpi=150
        )
        plt.close(fig)

        # Residuals versus predicted demand
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(predicted, residuals, alpha=0.4, s=10)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted demand")
        ax.set_ylabel("Residual")
        ax.set_title("LSTM Residuals vs Predicted Demand")
        fig.tight_layout()
        fig.savefig(
            plot_dir / "lstm_residuals_vs_predicted_demand.png", dpi=150
        )
        plt.close(fig)

        print(f"Plots saved in: {plot_dir.resolve()}")
        print(
            f"Residual mean: {residuals.mean():,.2f}"
            f"\nResidual standard deviation: {residuals.std():,.2f}"
        )

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.5f}")

    return {
        "run_name": run_name,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "predictions": predictions,
        "actual": y_test.ravel(),
        "feature_columns": feature_columns,
        "history": history.history,
    }