from pathlib import Path
import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from prepare_data import prepare_data

X_train, X_test, y_train, y_test, test_df = prepare_data()

# lgbm config
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

joblib.dump(model, "models/lightgbm.joblib")

# adding lightgbm predictions to the existing results file
results_path = Path("data/test_predictions.csv")
results = pd.read_csv(results_path)

# adding/replacing lightgbm column
results["lightgbm_predictions"] = pd.to_numeric(predictions)
results.to_csv(results_path, index=False)

print(f"\nModel saved to models/lightgbm.joblib"
      f"\nPredictions added to data/test_predictions.csv"
      )