## Current Progress

### Data Ingestion

* Source: UCI Power Consumption of Tetouan City
* Records: 52,417
* Frequency: 10-minute intervals
* Storage: PostgreSQL
* Schema: `energy`
* Main table: `energy.power_measurements`
* Built a Python ingestion pipeline to load the dataset into PostgreSQL

### Database

The project uses PostgreSQL to store the electricity and environmental measurements.

Current database structure:

```
gridops
└── energy
    └── power_measurements
```

* Created a structured PostgreSQL schema for energy measurements
* Added an index on `measured_at` for time-based queries
* Database schema is reproducible through `sql/schema.sql`

### Analytics

* Added an initial SQL analytics layer
* Implemented time-based aggregations
* Analysed consumption by day and hour
* Used Common Table Expressions (CTEs)
* Used window functions such as `LAG()` for period-over-period comparisons
* Implemented rolling 24-hour consumption averages
* Identified unusually high-demand periods using statistical thresholds
* Ranked the three distribution zones by average consumption
* Added analysis of consumption changes and demand patterns over time

### Feature Engineering

* Built a Python feature-engineering pipeline using data loaded from PostgreSQL
* Created time-based features including hour, day of week, month, and weekend indicators
* Created total power consumption across all three zones
* Calculated individual zone consumption shares
* Created lag features for 10-minute, 1-hour, and 24-hour historical demand
* Created rolling mean features for 1-hour, 6-hour, and 24-hour demand windows
* Prepared the dataset for one-hour-ahead electricity demand forecasting

### Forecasting Dataset

* Defined total power consumption as the forecasting target
* Created a one-hour-ahead target using the 10-minute measurement frequency
* Used a 6-step forward shift to represent one hour of future demand
* Removed rows without valid features, targets or baseline predictions.
* Used a chronological 80/20 train-test split to avoid randomly mixing historical and future observations.
* Current prepared dataset: 52,266 observations, comprising 41,812 training observations and 10,454 test observations.

### Forecasting Models

Three forecasting approaches are currently evaluated:

1. **Persistence baseline** — uses current demand to predict demand one hour ahead.
2. **Seasonal-naive baseline** — uses demand from 23 hours before the forecast origin, corresponding to the same time of day as the one-hour-ahead target on the previous day.
3. **Random Forest Regressor** — uses engineered time, weather and historical energy-demand features.
4. **XGBoost Regressor** — uses the engineered features to model nonlinear relationships and improve forecasting accuracy.
5. **LightGBM Regressor** — uses gradient boosting with a tree-based learning approach for demand forecasting.
6. **LSTM Neural Network** — uses sequences of 144 ten-minute observations (24 hours) and 20 engineered features per time step to predict demand one hour ahead.

The LSTM uses TensorFlow/Keras, with feature and target scaling, dropout regularisation, early stopping and learning-rate reduction.

Models are evaluated using the same chronological test period.

### Model Comparison

| Model | MAE | MSE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| Persistence Baseline | 4,178.64 | 35,271,703.25 | 5,939.00 | 0.8232 | 6.46% |
| Seasonal-Naive Baseline | 2,117.48 | 10,350,769.25 | 3,217.26 | 0.9481 | 3.38% |
| Random Forest | 2,667.78 | 15,086,067.73 | 3,884.08 | 0.9244 | 4.34% |
| XGBoost | 2,277.73 | 12,682,337.26 | 3,561.23 | 0.9366 | 3.65% |
| LightGBM | 2,245.15 | 11,814,271.45 | 3,437.19 | 0.9408 | 3.66% |
| LSTM | 3,768.86 | 22,150,121.68 | 4,706.39 | 0.8889 | 6.29% |



### Evaluation metrics

* **Mean Absolute Error**: Average absolute difference between predicted and actual demand.
* **Mean Squared Error**: Average squared prediction error, which penalises larger errors more heavily.
* **Root Mean Squared Error**: Square root of MSE, expressed in the same units as demand.
* **Coefficient of Determination**: Measures predictive performance relative to a test-set mean baseline; higher is generally better.
* **Mean Absolute Percentage Error**: Average absolute percentage error.

Lower MAE, MSE, RMSE and MAPE indicate lower prediction error. Higher R² indicates better performance relative to the test-set mean baseline.

### Key Findings

* The Seasonal Naive baseline achieved the best overall forecasting performance, with an MAE of 2,117.48 and an R² of 0.9481.
* LightGBM marginally outperformed XGBoost on MAE, RMSE, and R², while XGBoost achieved a slightly lower MAPE.
* Both boosting models outperformed Random Forest across the reported metrics.
* The results highlight the importance of evaluating machine-learning models against simple time-series baselines. Daily demand patterns provide a strong benchmark for short-term electricity demand forecasting.
* The LSTM currently underperforms the simpler forecasting baselines and requires further investigation and tuning.
* New results indicate adding previous-day demand provides useful information for this forecasting. However, the LSTM has not yet outperformed the seasonal-naive baseline, which remains an important benchmark.

### Predictions Dataset: data/test_predictions.csv

The predictions CSV contains the actual test-set demand, forecast timestamps, baseline predictions and machine-learning predictions.


| Column |	Meaning |
|----|:----| 
| forecast_origin	| Timestamp when the prediction is made |
| target_timestamp |	Timestamp one hour after the forecast origin; the time being predicted |
| actual	| Actual demand recorded at target_timestamp |
| naive_predictions	| Persistence forecast, using current demand to predict demand one hour ahead |
| seasonal_naive_predictions |	Seasonal-naive forecast based on historical demand at the corresponding time on the previous day |
| random_forest_predictions |	Demand forecast produced by the Random Forest model |
| xgboost_predictions |	Demand forecast produced by the XGBoost model |


### Project Structure

```
gridops-ai/
├── data/
│   ├── plots/
│   ├── raw/
│   ├── experiment_logs.xlsx
│   └── test_predictions.csv
├── models/
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lightgbm.joblib
│   └── lstm_model.keras
├── sql/
│   ├── schema.sql
│   ├── README.md
│   └── analytics.sql
├── src/
│   ├── load_to_postgres.py
│   ├── feature_engineering.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── train_xgboost.py
|   ├── train_lightgbm.py
|   ├── train_lstm.py
│   └── evaluate.py
└── README.md
```

### Project Status

**Completed**

* [x] Project setup
* [x] Dataset ingestion
* [x] PostgreSQL database setup
* [x] Data loading pipeline
* [x] Initial SQL analytics
* [x] Advanced SQL analytics
* [x] Feature engineering
* [x] Forecasting target
* [x] First machine learning benchmark
* [x] Build naive forecasting baselines
* [x] Compare naive and machine learning baselines
* [x] Compare machine learning models

**Next**

* [ ] Model explainability with SHAP
* [ ] Model tracking with MLflow
* [ ] FastAPI model serving
* [ ] Agentic AI integration
* [ ] Dockerisation
* [ ] Deployment
