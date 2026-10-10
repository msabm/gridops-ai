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

```text
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
* Removed rows without a valid forecasting target
* Verified the final dataset contains 52,411 forecasting observations

### Machine Learning Baselines

Three forecasting approaches are currently evaluated:

1. **Persistence baseline** — uses current demand to predict demand
   one hour ahead.
2. **Seasonal-naive baseline** — uses demand from 23 hours before
   the forecast origin, corresponding to the same time of day as
   the one-hour-ahead target on the previous day.
3. **Random Forest Regressor** — uses engineered time, weather
   and historical energy-demand features.

Models are evaluated using a chronological 80/20 train-test split.

### Model Comparison

| Model | MAE | MSE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| Persistence Baseline | 4,178.64 | 35,271,703.25 | 5,939.00 | 0.8232 | 6.46% |
| Seasonal-Naive Baseline | 2,117.48 | 10,350,769.25 | 3,217.26 | 0.9481 | 3.38% |
| Random Forest | 2,667.78 | 15,086,067.73 | 3,884.08 | 0.9244 | 4.34% |

### Key Findings

* Lower MAE, MSE, RMSE and MAPE indicate lower prediction error. Higher R² indicates a better fit relative to predicting the test-set mean.

* The seasonal-naive baseline currently outperforms the Random Forest on this test set. This highlights the importance of daily demand patterns and establishes a benchmark that more advanced models must improve upon.

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


**Next**

* [ ] Compare machine learning models
* [ ] Model explainability with SHAP
* [ ] Model tracking with MLflow
* [ ] FastAPI model serving
* [ ] Agentic AI integration
* [ ] Dockerisation
* [ ] Deployment
