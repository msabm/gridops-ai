CREATE SCHEMA IF NOT EXISTS energy;

CREATE TABLE IF NOT EXISTS energy.power_measurements (
    measurement_id BIGSERIAL PRIMARY KEY,
    measured_at TIMESTAMP NOT NULL,

    temperature NUMERIC(6,2),
    humidity NUMERIC(6,2),
    wind_speed NUMERIC(8,4),
    general_diffuse_flows NUMERIC(10,4),
    diffuse_flows NUMERIC(10,4),

    zone_1_power_consumption NUMERIC(12,4),
    zone_2_power_consumption NUMERIC(12,4),
    zone_3_power_consumption NUMERIC(12,4),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_power_measurements_measured_at
ON energy.power_measurements (measured_at);