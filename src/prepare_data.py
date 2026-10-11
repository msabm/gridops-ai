from feature_engineering import (
    load_data,
    create_time_features,
    create_energy_features,
    create_lag_features,
    create_rolling_features,
    create_forecasting_target
)

features = [
    "temperature",
    "humidity",
    "wind_speed",
    "general_diffuse_flows",
    "diffuse_flows",
    "hour",
    "day_of_week",
    "month",
    "year",
    "is_weekend",
    "total_power_consumption",
    "zone_1_share",
    "zone_2_share",
    "zone_3_share",
    "lag_1",
    "lag_6",
    "lag_144",
    "rolling_1h_mean",
    "rolling_6h_mean",
    "rolling_24h_mean",
]


def prepare_data():
    df = load_data()

    df = df.drop_duplicates(subset=["measured_at"])
    df = create_time_features(df)
    df = create_energy_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_forecasting_target(df)

    # timestamps before removing rows with missing values
    df["target_timestamp_naive"] = df["measured_at"].shift(-6)
    df["target_timestamp_seasonal_naive"] = df["measured_at"].shift(138)

    # Current demand predicts demand one hour ahead
    df["naive_predictions"] = df["total_power_consumption"]

    # 144 intervals per day - 6 intervals to target = 138
    df["seasonal_naive_predictions"] = (
        df["total_power_consumption"].shift(138)
    )

    # Remove rows with missing features, targets or timestamps
    df = df.dropna().reset_index(drop=True)

    # Chronological 80/20 split
    split_index = int(len(df) * 0.8)

    X = df[features]
    y = df["target_1h_ahead"]

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_df = df.iloc[split_index:].copy()

    print(
        f"\nTotal observations: {len(df):,}"
        f"\nTraining observations: {len(X_train):,}"
        f"\nTesting observations: {len(X_test):,}"
        f"\nNumber of features: {len(features)}"
        f"\nTraining period: {df['measured_at'].iloc[0]} "
        f"to {df['measured_at'].iloc[split_index - 1]}"
        f"\nTesting period: {test_df['measured_at'].iloc[0]} "
        f"to {test_df['measured_at'].iloc[-1]}"
    )

    return X_train, X_test, y_train, y_test, test_df    