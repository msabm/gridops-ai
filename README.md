## Current Progress

### Data Ingestion
- Source: UCI Power Consumption of Tetouan City
- Records: 52,417
- Frequency: 10-minute intervals
- Storage: PostgreSQL
- Schema: `energy`
- Main table: `energy.power_measurements`

### Database
The project uses PostgreSQL to store the electricity and environmental measurements.

Current database structure:

```text
gridops
└── energy
    └── power_measurements

 ### Project Structure

```text
gridops-ai/
├── data/
│   └── raw/
├── sql/
│   └── schema.sql
├── src/
│   └── load_to_postgres.py
├── README.md
├── .gitignore
└── requirements.txt