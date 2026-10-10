import pandas as pd

from feature_engineering import (
    load_data,
    create_time_features,
    create_energy_features,
    create_lag_features,
    create_rolling_features,
    create_forecasting_target,
)

# loading data
df = load_data()

# removing duplicate rows
df = df.drop_duplicates(subset=["measured_at"])

# creating features
df = create_time_features(df)
df = create_energy_features(df)
df = create_lag_features(df)
df = create_rolling_features(df)
df = create_forecasting_target(df)

# removing rows with NaN values
df = df.dropna(subset=["target_1h_ahead"])

# removing rows with NaN values in lag/rolling features
df = df.dropna()

# feature set
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

target = "target_1h_ahead"
df["target_timestamp"] = df["measured_at"].shift(-6)


X = df[features]
y = df[target]

# splitting data into train and test sets
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Total observations: {len(df)}")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")
print(f"Number of features: {len(features)}")

print(f"\nTraining period:")
print(f"{df['measured_at'].iloc[:split_index].min()} to {df['measured_at'].iloc[split_index - 1]}")
print(f"\nTesting period:")
print(f"{df['measured_at'].iloc[split_index:].min()} to {df['measured_at'].iloc[-1]}")



# Training the baseline model
from sklearn.ensemble import RandomForestRegressor
import joblib

model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)

model.fit(X_train, y_train)

# random forest predictions
predictions = model.predict(X_test)

# saving model for evaluation
joblib.dump(model, "models/random_forest.joblib")

# naive predictions using current demand - current demand predicts demand one hour ahead
naive_predictions = X_test["total_power_consumption"].to_numpy()

# seasonal naive predictions - demand at t-23h predicts demand at t+1h (23*6=138)
seasonal_naive_predictions = df["total_power_consumption"].shift(138).iloc[split_index:].to_numpy()

results = pd.DataFrame({
    "measured_at": df["measured_at"].iloc[split_index:].to_numpy(),
    "target_timestamp": df["target_timestamp"].iloc[split_index:],
    "actual": y_test.to_numpy(),
    "naive_predictions": naive_predictions,
    "seasonal_naive_predictions": seasonal_naive_predictions,
    "random_forest_predictions": predictions
})

results.to_csv("data/test_predictions.csv", index=False)

print("Model saved to models/random_forest.joblib")
print("Test predictions saved to data/test_predictions.csv")