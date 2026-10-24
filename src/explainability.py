import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path

from prepare_data import prepare_data


# file paths and config
model_path = Path("models/lstm_model.keras")
preprocessing_path = Path("models/lstm_preprocessing.joblib")
predictions_path = Path("data/test_predictions.csv")
plots_directory = Path("data/plots")

background_size = 40
explain_size = 100
random_state = 42


# loading trained model and evaluation results
model = tf.keras.models.load_model(model_path)
preprocessing = joblib.load(preprocessing_path)

X_scaler = preprocessing["X_scaler"]
y_scaler = preprocessing["y_scaler"]
lookback = preprocessing["lookback"]
feature_columns = preprocessing["feature_columns"]


# loading same train/test split
X_train, X_test, y_train, y_test, test_df = prepare_data()

def as_feature_dataframe(data, columns):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data, columns=columns)

X_train = as_feature_dataframe(X_train, X_train.columns if isinstance(
    X_train, pd.DataFrame
) else feature_columns)

X_test = as_feature_dataframe(X_test, X_test.columns if isinstance(
    X_test, pd.DataFrame
) else feature_columns)

# checking pipeline has all features
missing_features = [
    col for col in feature_columns if col not in X_test.columns
    or col not in X_test.columns
]

if missing_features:
    raise ValueError(
        "Missing features required by the trained model:\n{missing_feaures}"
    )

# exact feature names and order from saved model
X_train = X_train[feature_columns]
X_test = X_test[feature_columns]


# applying scalers
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# building step sequences, 144
def make_sequences(features, window):
    sequences = []

    for i in range(window - 1, len(features)):
        sequences.append(features[i - window + 1:i + 1])
    return np.asarray(sequences,dtype=np.float32)


X_train_sequences = make_sequences(
    X_train_scaled, lookback
)

# preserving previous training observations as context for next test
context = X_train_scaled[-(lookback - 1):]

combined_test = np.concatenate(
    [context, X_test_scaled], axis=0
)

X_test_sequences = make_sequences(
    combined_test, lookback
)

assert len(X_test_sequences) == len(X_test), ("Test sequences do not align with test feature rows")

print(f"\nTraining sequences: {X_train_sequences.shape}"
      f"\nTest sequences: {X_test_sequences.shape}"
)


# setting background and explanation samples
rng = np.random.default_rng(random_state)

background_indices = rng.choice(
    len(X_train_sequences),
    size=min(background_size, len(X_train_sequences)),
    replace=False
)

explain_indices = rng.choice(
    len(X_test_sequences),
    size=min(explain_size, len(X_test_sequences)),
    replace=False
)

background = X_train_sequences[background_indices]
X_explain = X_test_sequences[explain_indices]


# SHAP values
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_explain)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
    shap_values = shap_values[..., 0]

if shap_values.shape != X_explain.shape:
    raise ValueError(
        print(f"\nUnexpected SHAP shape: {shap_values.shape}"
              f"\nExpected shape: {X_explain.shape}"
        ))

print(f"SHAP values calculated: {shap_values.shape}")


# Saving SHAP gloabl feaure importance 
mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))

importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_absolute_shap": mean_abs_shap
}).sort_values(
    "mean_absolute_shap", ascending=False
)

importance.to_csv("data/shap_feature_importance.csv", index=False)

print(f"\nTop 10 features: {importance.head(10).to_string(index=False)}")


# plotting global feature importance
top_features = importance.head(15).sort_values("mean_absolute_shap")

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["feature"],
    top_features["mean_absolute_shap"]
)    
plt.xlabel("Mean absolute SHAP value (scales prediction units)")
plt.ylabel("Input feature")
plt.title("LSTM Global Feature Importance")
plt.tight_layout()
plt.savefig("data/plots/shap_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

