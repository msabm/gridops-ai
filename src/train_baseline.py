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
    "lag_5",
    "lag_144",
    "rolling_1h_mean",
    "rolling_6h_mean",
    "rolling_24h_mean",
]

target = "target_1h_ahead"


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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)

model.fit(X_train, y_train)

# predicting
predictions = model.predict(X_test)

# evaluating the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nBaseline Model Performance: "
    f"\nMAE: {mae:.2f}"
    f"\nMSE: {mse:.2f}"
    f"\nR2: {r2:.4f}"
    ) 