import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

results = pd.read_csv("data/test_predictions.csv")

# actual demand one hour ahead for test period
actual = results["actual"].to_numpy()

models = {
    "Naive Baseline":results["naive_predictions"],
    "Random Forest":results["random_forest_predictions"]
}

comparison = []

for name, predictions in models.items():
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = root_mean_squared_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions)

    comparison.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4)
    })

comparison_df = pd.DataFrame(comparison)

print(f"\nModel Comparison: "
    f"\n{comparison_df.to_string(index=False, justify='center')}"
)