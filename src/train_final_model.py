# Separating the training of the final model from the rest of the code to avoid cluttering the main script

from train_lstm import train_lstm

top_10_features = [
    "lag_138", "total_power_consumption",
    "lag_144", "lag_1",
    "month", "rolling_6h_mean",
    "rolling_1h_mean", "zone_2_share",
    "hour_cos", "hour_sin",
]

if __name__ == "__main__":
    results = train_lstm(
        selected_features=top_10_features,
        run_name="lstm_final_top10",
        lookback=144,
        epochs=50,
        batch_size=32,
        save_outputs=True,
        make_plots=True,
    )

    print(f"Final features: {results['feature_columns']}"
        f"\nMetrics: {results['metrics']}"
        )
