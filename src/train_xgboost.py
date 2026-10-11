import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor


from prepare_data import prepare_data

X_train, X_test, y_train, y_test, test_df = prepare_data()

# XGB config
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="mae",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

joblib.dump(model, "models/xgboost.joblib")

# Add XGBoost predictions to the existing results file
results_path = Path("data/test_predictions.csv")
results = pd.read_csv(results_path)

# adding/replacing xgb column
results["xgboost_predictions"] = predictions

results.to_csv(results_path, index=False)

print(f"\nModel saved to models/xgboost.joblib"
      f"\nPredictions saved to data/xgboost_predictions.csv"
    )