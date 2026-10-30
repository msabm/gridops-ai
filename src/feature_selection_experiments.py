import mlflow
import pandas as pd

from pathlib import Path
from train_lstm import train_lstm


# config
shap_path = Path("data/shap_feature_importance.csv")
results_path = Path("data/lstm_feature_selection_results.csv")

mlflow_tracking_uri = "sqlite:///mlflow.db"
experiment_name = "GridOps AI - Feature Selection"

# to retrieve 22 feature trained lstm
baseline_run_id = "f0fb680243004286bfc2ab0743c8caae"

lookback = 144
epochs = 50
batch_size = 32


# loding features ordered by SHAP importance
def load_shap_rankings(csv_path):
    shap_df = pd.read_csv(csv_path)

    normalised_columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in shap_df.columns
    }

    feature_candidates = [
        "feature", "feature_name",
        "features", "variable", 
    ]

    importance_candidates = [
        "mean_abs_shap", "mean_absolute_shap",
        "mean_abs_shap_value", "shap_importance",
        "importance", "mean_shap",
    ]

    feature_col = next((
        normalised_columns[name]
        for name in feature_candidates
        if name in normalised_columns
        ), None,
    )

    importance_col = next((
        normalised_columns[name]
        for name in importance_candidates
        if name in normalised_columns
        ), None,
    )

    if feature_col is None or importance_col is None:
        raise ValueError(
            "Could not find the feature and SHAP importance columns" 
        )

    shap_df = shap_df[[feature_col, importance_col]].copy()
    shap_df.columns = ["feature", "importance"]
    shap_df = shap_df.dropna()

    shap_df = shap_df.sort_values(
        "importance", ascending=False,
    )

    if shap_df["feature"].duplicated().any():
        raise ValueError("Duplicated feature names found in SHAP rankings")

    return shap_df["feature"].astype(str).to_list()


def main():
    # using existing mlflow tracking db
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    ranked_features = load_shap_rankings(shap_path)

    if len(ranked_features) < 15:
        raise ValueError(
            f"Found only {len(ranked_features)} features"
        )

    feature_sets = {
        "lstm_top20_features_v2": ranked_features[:20],
        "lstm_top17_features_v2": ranked_features[:17],       
        "lstm_top15_features_v2": ranked_features[:15],
        "lstm_top12_features_v2": ranked_features[:12],
        "lstm_top10_features_v2": ranked_features[:10],
    }

    for name, features in feature_sets.items():
        print(f"\n{name} ({len(features)} features): {', '.join(features)}")

    baseline_run = mlflow.get_run(baseline_run_id)

    baseline_metrics = {
        key.lower(): value
        for key, value in baseline_run.data.metrics.items()
    }

    metric_names = ["mae", "mse", "rmse", "r2", "mape"]

    results = [{
        "run_name": "lstm_baseline_22_features",
        "run_id": baseline_run_id,
        "feature_count": 22,
        **{
            metric: baseline_metrics[metric]
            for metric in metric_names
        }
    }]


    # running the controlled experiments
    for run_name, features in feature_sets.items():
        print(f"Starting experiment: {run_name}"
              f"{'='*50}"
              )

        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.set_tags({
                "model_family": "LSTM",
                "experiment_type": "SHAP feature selection",
            })

            mlflow.log_params({
                "feature_selection": "SHAP",
                "feature_count": len(features),
                "selected_features": ",".join(features),
                "lookback": lookback,
                "epoch": epochs,
                "batch_size": batch_size,
                "architecture": "LSTM128-dropout-LSTM64-dropout-Dense32",
                "learning_rate": 0.001,
                "loss_function": "MSE",
                "validation_split": 0.15,
                "random_seed": 42,
            })

            result = train_lstm(
                selected_features=features,
                run_name=run_name,
                lookback=lookback,
                epochs=epochs,
                batch_size=batch_size,
                save_outputs=False,
                make_plots=False,
            )

            mlflow.log_metrics(result["metrics"])
            mlflow.log_metrics({
                "best_epoch": result["best_epoch"],
                "best_val_loss_scaled": result["best_val_loss"],
                "epochs_run": len(result["history"]["loss"]),
            })

            results.append({
                "run_name": run_name,
                "run_id": active_run.info.run_id,
                "feature_count": len(features),
                **result["metrics"],
            })

            print(f"MLFlow run ID: {active_run.info.run_id}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    print(f"Final comparison"
            f"{results_df[
                ["run_name", "feature_count", "mae", "mse", "rmse", "r2", "mape"]
            ].to_string(index=False)}"
            )

if __name__ == "__main__":
    main()