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
* Added initial analysis of consumption changes over time

### Project Status

**Completed**

* [x] Project setup
* [x] Dataset ingestion
* [x] PostgreSQL database setup
* [x] Data loading pipeline
* [x] Initial SQL analytics

**Next**

* [ ] Advanced SQL analytics
* [ ] Feature engineering
* [ ] Machine learning models
* [ ] Model tracking with MLflow
* [ ] FastAPI model serving
* [ ] Agentic AI integration
* [ ] Dockerisation
* [ ] Deployment
