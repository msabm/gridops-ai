import joblib
import tensorflow as tf

from pathlib import Path

model_path = Path("models/lstm_final_top10_model.keras")
preprocessing_path = Path("models/lstm_final_top10_preprocessing.joblib")

def load_model_artifacts():
    model = tf.keras.models.load_model(model_path)
    preprocessing = joblib.load(preprocessing_path)

    return model, preprocessing