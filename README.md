# GridOps AI — Electricity Demand Forecasting

An end-to-end electricity demand forecasting project combining PostgreSQL, SQL analytics, Python feature engineering, machine learning, LSTM modelling, explainability, experiment tracking, and a FastAPI inference service.

## Project Overview

GridOps AI uses historical electricity consumption and environmental measurements from the UCI Power Consumption of Tetouan City dataset to forecast total electricity demand one hour ahead.

The project is designed as an end-to-end workflow:

1. Ingest and store historical measurements in PostgreSQL.
2. Analyse consumption patterns using SQL.
3. Engineer time-series and energy-demand features in Python.
4. Benchmark statistical and machine-learning forecasting approaches.
5. Explain model behaviour using SHAP.
6. Track experiments with MLflow.
7. Serve the selected LSTM model through a REST API built with FastAPI.

## Dataset and Data Ingestion

- **Source:** UCI Power Consumption of Tetouan City
- **Records:** 52,417 in the source dataset
- **Frequency:** 10-minute intervals
- **Storage:** PostgreSQL
- **Database:** `gridops`
- **Schema:** `energy`
- **Main table:** `energy.power_measurements`

A Python ingestion pipeline loads the source measurements into PostgreSQL.

## Database

PostgreSQL stores the electricity and environmental measurements.

```text
gridops
└── energy
    └── power_measurements
```

- Created a structured PostgreSQL schema for energy measurements.
- Added an index on `measured_at` for time-based queries.
- Made the database schema reproducible through `sql/schema.sql`.

## SQL Analytics

The SQL analytics layer explores electricity consumption patterns across the three distribution zones.

- Implemented time-based aggregations.
- Analysed consumption by day and hour.
- Used Common Table Expressions (CTEs).
- Used window functions such as `LAG()` for period-over-period comparisons.
- Implemented rolling 24-hour consumption averages.
- Identified unusually high-demand periods using statistical thresholds.
- Ranked the three distribution zones by average consumption.
- Analysed changes in consumption and demand patterns over time.

## Feature Engineering

A Python feature-engineering pipeline loads measurements from PostgreSQL and prepares them for forecasting.

- Created time-based features, including hour, day of week, month, and weekend indicators.
- Calculated total power consumption across all three zones.
- Calculated individual zone consumption shares.
- Created lag features representing recent and historical demand.
- Created rolling mean features for 1-hour, 6-hour, and 24-hour demand windows.
- Prepared the data for one-hour-ahead electricity demand forecasting.

## Forecasting Dataset

- **Target:** Total power consumption across the three zones.
- **Forecast horizon:** One hour ahead.
- **Sampling interval:** 10 minutes.
- **Target construction:** A six-step forward shift represents one hour.
- **Prepared observations:** 52,266.
- **Training observations:** 41,812.
- **Test observations:** 10,454.
- **Split:** Chronological 80/20 train-test split, keeping later observations in the test set.

Rows without valid features, targets, or required baseline predictions are removed during preparation.

## Forecasting Models

The project evaluates six forecasting approaches:

1. **Persistence baseline** — uses current demand to predict demand one hour ahead.
2. **Seasonal-naive baseline** — uses demand from 23 hours before the forecast origin, corresponding to the same time of day as the one-hour-ahead target on the previous day.
3. **Random Forest Regressor** — uses engineered time, weather, and historical energy-demand features.
4. **XGBoost Regressor** — uses gradient boosting to model nonlinear relationships.
5. **LightGBM Regressor** — uses a tree-based gradient boosting approach.
6. **LSTM Neural Network** — uses sequences of 144 ten-minute observations (24 hours) to forecast demand one hour ahead.

The LSTM is implemented with TensorFlow/Keras and uses feature and target scaling, dropout regularisation, early stopping, and learning-rate reduction.

### LSTM Feature Selection

SHAP-based analysis was used to investigate which input features contributed most to the LSTM's predictions. Experiments compared different feature subsets.

The selected Top 10 feature set is:

```python
[
    "lag_138",
    "total_power_consumption",
    "lag_144",
    "lag_1",
    "month",
    "rolling_6h_mean",
    "rolling_1h_mean",
    "zone_2_share",
    "hour_cos",
    "hour_sin",
]
```

The final Top 10 model uses:

- 10 input features per time step.
- A lookback window of 144 observations (24 hours).
- A one-hour-ahead forecasting target.
- TensorFlow/Keras with dropout, early stopping, and learning-rate reduction.

### Model Comparison

The table below reports the benchmark results recorded in the project. The LSTM row reflects the later Top 10 feature-selection experiment.

| Model | MAE | MSE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| Persistence Baseline | 4,178.64 | 35,271,703.25 | 5,939.00 | 0.8232 | 6.46% |
| Seasonal-Naive Baseline | 2,117.48 | 10,350,769.25 | 3,217.26 | 0.9481 | 3.38% |
| Random Forest | 2,667.78 | 15,086,067.73 | 3,884.08 | 0.9244 | 4.34% |
| XGBoost | 2,277.73 | 12,682,337.26 | 3,561.23 | 0.9366 | 3.65% |
| LightGBM | 2,245.15 | 11,814,271.45 | 3,437.19 | 0.9408 | 3.66% |
| **LSTM (Top 10 features)** | **2,514.36** | **9,754,150.00** | **3,123.16** | **0.9511** | **4.31%** |

**Important evaluation note:** The Top 10 feature configuration was selected after comparing feature-selection experiments on the same chronological test period. Its reported test metrics are therefore useful for project comparison, but should not be presented as an untouched, independent estimate of generalisation performance.

### Evaluation Metrics

- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual demand.
- **Mean Squared Error (MSE):** Average squared prediction error, which penalises larger errors more heavily.
- **Root Mean Squared Error (RMSE):** Square root of MSE, expressed in the same units as demand.
- **Coefficient of Determination (R²):** Measures predictive performance relative to a test-set mean baseline.
- **Mean Absolute Percentage Error (MAPE):** Average absolute percentage error.

Lower MAE, MSE, RMSE, and MAPE indicate lower prediction error. Higher R² indicates better performance relative to the test-set mean baseline.

### Key Findings

- The Seasonal-Naive baseline is a strong benchmark, demonstrating the value of recurring daily demand patterns.
- LightGBM and XGBoost outperform Random Forest across the reported metrics.
- The Top 10 LSTM experiment improves on the earlier 22-feature LSTM result.
- The Top 10 LSTM achieves an MAE of 2,514.36, RMSE of 3,123.16, and R² of 0.9511.
- The LSTM's R² is higher than the seasonal-naive baseline, while its MAE and MAPE remain higher. Different metrics therefore give different perspectives on model performance.
- The results reinforce the importance of comparing machine-learning models against simple time-series baselines.

## LSTM Explainability with SHAP

SHAP was applied to investigate which input features contributed most to the LSTM's predictions.

Key findings from the feature-importance analysis:

- `lag_138` had the highest mean absolute SHAP value (0.004801), highlighting the importance of demand observed 23 hours before the forecast origin.
- `total_power_consumption` was the second most influential feature (0.003715).
- `lag_144` and `lag_1` also ranked highly, indicating that both daily historical patterns and recent demand are important inputs.
- Rolling demand statistics and cyclical hour features provided additional predictive information.
- Weather features had lower mean absolute SHAP values than the leading demand-history features.

The results suggest that the model relies primarily on historical demand patterns and current demand levels. SHAP explains model-specific predictive behaviour; it does not establish causal relationships.

## Experiment Tracking with MLflow

MLflow is used to track and compare forecasting experiments.

Runs record model parameters, evaluation metrics, and tags describing the model, forecast horizon, evaluation dataset, and number of evaluation rows.

### Running Experiment Tracking

From the repository root, activate the virtual environment, install the dependencies, and run:

```bash
pip install -r requirements.txt
python src/track_experiments.py
```

The tracking script evaluates existing predictions in `data/test_predictions.csv` and records results in a local MLflow tracking database. It also generates a model-comparison CSV and logs it as an MLflow artifact.

Launch the MLflow UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open `http://127.0.0.1:5000` to inspect experiment runs, metrics, parameters, and artifacts.

**Note:** The tracking script records existing evaluation results; it does not retrain the models.

## FastAPI Model Serving

The selected Top 10 LSTM model is served through a REST API built with FastAPI.

The API loads the trained model and its preprocessing artefacts when the application starts. When a forecast is requested, the service:

1. Loads the latest available measurements from PostgreSQL.
2. Applies the same feature-engineering steps used by the forecasting pipeline.
3. Selects the 10 features expected by the saved model.
4. Extracts the latest 144 observations (24 hours).
5. Applies the saved feature and target scalers.
6. Generates a one-hour-ahead demand forecast.
7. Returns the forecast and associated metadata as JSON.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns the API welcome message and service status. |
| GET | `/health` | Returns the API health status and number of model features. |
| GET | `/forecast` | Generates a one-hour-ahead electricity demand forecast. |
| GET | `/docs` | Opens the interactive Swagger API documentation. |

### Run the API Locally

From the repository root, activate the project's virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Start the development server:

```bash
uvicorn src.api.main:app --reload
```

The API is available at:

- Base URL: `http://127.0.0.1:8000`
- Interactive documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Forecast endpoint: `http://127.0.0.1:8000/forecast`

The `/forecast` endpoint returns fields including the forecast origin, target timestamp, predicted total power consumption, model name, feature count, and lookback length.

### Automated API Tests

Automated endpoint tests are implemented with `pytest` and FastAPI's `TestClient`. They check the root endpoint, health endpoint, and forecast response.

Run the tests from the repository root:

```bash
python -m pytest src/api/test_api.py -v
```

The current test suite contains three passing tests.

**Data scope:** The current API forecasts from the latest measurements available in the project's PostgreSQL dataset. Since the source dataset is historical, this is an inference demonstration on the latest stored observations, not a live connection to an electricity meter or real-time grid feed.

### Example API Response

A successful request to `/forecast` returns:

```json
{
    "forecast_origin": "2017-12-30T23:50:00",
    "target_timestamp": "2017-12-31T00:50:00",
    "predicted_total_power_consumption": 56445.41,
    "model": "LSTM",
    "features": 10,
    "lookback_steps": 144
}
```

### Project Structure

```text
gridops-ai/
├── data/
│   ├── plots/
│   ├── raw/
│   ├── ml_flow_model_comparison.csv
│   ├── lstm_feature_selection_results.csv
│   ├── experiment_logs.xlsx
│   ├── test_predictions.csv
│   ├── shap_feature_importance.csv
│   └── saved_lstm_prediction_results.csv
├── models/
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lightgbm.joblib
│   ├── lstm_model.keras
│   ├── lstm_preprocessing.joblib
│   ├── lstm_final_top10_model.keras
│   └── lstm_final_top10_preprocessing.joblib
├── sql/
│   ├── schema.sql
│   ├── README.md
│   └── analytics.sql
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── services.py
│   │   └── test_api.py
│   ├── data_ingestion.py
│   ├── load_to_postgres.py
│   ├── feature_engineering.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── train_lstm.py
│   ├── train_lstm_backup.py
│   ├── train_final_model.py
│   ├── prediction.py
│   ├── explainability.py
│   ├── evaluation.py
│   ├── feature_selection_experiments.py
│   └── track_experiments.py
├── README.md
├── requirements.txt
└── pytest.ini
```

The tree lists the main project files and omits generated folders such as `.venv`, `__pycache__`, `.pytest_cache`, `mlruns`, and local database files where appropriate. Adjust the tree if your current repository differs.

## Project Status

### Completed

- [x] Project setup
- [x] Dataset ingestion
- [x] PostgreSQL database setup
- [x] Data loading pipeline
- [x] Initial and advanced SQL analytics
- [x] Feature engineering
- [x] Forecasting target and chronological train-test split
- [x] Forecasting baselines and machine-learning benchmarks
- [x] LSTM model development
- [x] SHAP-based model explainability
- [x] Feature-selection experiments
- [x] MLflow experiment tracking
- [x] FastAPI model-serving endpoints
- [x] Automated tests for API endpoints

### Next

- [ ] Agentic AI integration
- [ ] Dockerisation
- [ ] Deployment
- [ ] Further API robustness and error-handling tests
