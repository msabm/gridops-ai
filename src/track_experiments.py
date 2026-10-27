import mlflow
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, r2_score,
    root_mean_squared_error
)

# file paths
predictions_path = Path("data/test_predictions.csv")
comparison_path = Path("data/ml_flow_model_comparison.csv")

# local mlflow tracking database
mlflow.set_tracking_uri(
    f"sqlite:///{Path('mlflow.db').as_posix()}"
)
mlflow.set_experiment("GridOps AI - Energy Forecasting")

model_columns = {
    "Naive Baseline": "naive_predictions",
    "Seasonal Naive Baseline": "seasonal_naive_predictions",
    "Random Forest": "random_forest_predictions",
    "XGBoost": "xgboost_predictions",
    "LightGBM": "lightgbm_predictions",
    "LSTM": "lstm_predictions",
}

def main():
    df = pd.read_csv(predictions_path)

    required_columns = ["actual", *model_columns.values()]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # evaluating every model on the same rows for comparison
    evaluation_df = df[required_columns].dropna().copy()

    if evaluation_df.empty:
        raise ValueError("No complete rows available for evaluation")

    actual = evaluation_df["actual"]
    comparison = []

    for model_name, predictions_column in model_columns.items():
        predictions = evaluation_df[predictions_column]

        metrics = {
            "MAE": float(mean_absolute_error(actual, predictions)),
            "MSE": float(mean_squared_error(actual, predictions)),
            "RMSE": float(root_mean_squared_error(actual, predictions)),
            "R2": float(r2_score(actual, predictions)),
            "MAPE": float(mean_absolute_percentage_error(actual, predictions)),
        }

        mlflow.start_run(run_name=model_name)

        try:
            mlflow.set_tags({
                "project": "GridOps AI",
                "model_name": model_name,
                "tracking_type": "retrospective_evaluation",
                "evaluation_dataset": "test_predictions.csv",
            })

            mlflow.log_params({
                "model_name": model_name,
                "forecast_horizon_hours": 1,
                "evaluation_rows": len(evaluation_df),
                "prediction_column": predictions_column,
            })
            
            mlflow.log_metrics(metrics)

        finally:
            mlflow.end_run()

        comparison.append({
            "Model": model_name,
            **{key: round(value, 4) for key, value in metrics.items()},
            "Evaluation_Rows": len(evaluation_df),
        })

        print(f"Logged: {model_name} | MAE: {metrics["MAE"]:.2f}")

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(comparison_path, index=False)

    # attaching summary table ot the final run for inspection
    with mlflow.start_run(run_name="model_comparison_summary"):
        mlflow.set_tag("tracking_type", "comparison_summary")
        mlflow.log_artifact(str(comparison_path))

if __name__ == "__main__":
    main()