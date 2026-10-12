import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

from prepare_data import prepare_data

X_train, X_test, y_train, y_test, test_df = prepare_data()


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
seasonal_naive_predictions = test_df["seasonal_naive_predictions"].to_numpy()

results = pd.DataFrame({
    "forecast_origin": test_df["measured_at"].to_numpy(),
    "target_timestamp": test_df["target_timestamp"].to_numpy(),
    "actual": pd.to_numeric(y_test.to_numpy()),
    "naive_predictions": pd.to_numeric(naive_predictions),
    "seasonal_naive_predictions": pd.to_numeric(seasonal_naive_predictions),
    "random_forest_predictions": pd.to_numeric(predictions)
})

results.to_csv("data/test_predictions.csv", index=False)

print("Model saved to models/random_forest.joblib")
print("Test predictions saved to data/test_predictions.csv")