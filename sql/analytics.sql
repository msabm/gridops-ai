-- ============================================================
-- GridOps AI - Energy Analytics
-- ============================================================
SELECT *
FROM energy.power_measurements;

-- 1. Basic dataset overview
SELECT
    COUNT(*) AS total_measurements,
    MIN(measured_at) AS first_measurement,
    MAX(measured_at) AS last_measurement
FROM energy.power_measurements;


-- 2. Daily energy consumption by zone
SELECT
	DATE(measured_at) AS measurement_date,
	SUM(zone_1_power_consumption) AS zone_1_total,
	SUM(zone_2_power_consumption) AS zone_2_total,
	SUM(zone_3_power_consumption) AS zone_3_total
FROM energy.power_measurements
GROUP BY DATE(measured_at)
ORDER BY measurement_date;


-- 3. Average consumption by hour
SELECT
	EXTRACT(HOUR FROM measured_at) AS hour_of_day,
	ROUND(AVG(zone_1_power_consumption), 2) AS zone_1_average,
	ROUND(AVG(zone_2_power_consumption), 2) AS zone_2_average,
	ROUND(AVG(zone_3_power_consumption), 2) AS zone_3_average
FROM energy.power_measurements
GROUP BY EXTRACT(HOUR FROM measured_at)
ORDER BY hour_of_day;


-- 4. Total consumption and average temperature by day
WITH daily_metrics AS (
	SELECT 
		DATE(measured_at) AS measurement_date,
		ROUND(AVG(temperature), 2) AS average_temperature,
		SUM(zone_1_power_consumption) AS zone_1_total,
		SUM(zone_2_power_consumption) AS zone_2_total,
		SUM(zone_3_power_consumption) AS zone_3_total
	FROM energy.power_measurements
	GROUP BY DATE(measured_at)
)

SELECT 
	measurement_date,
	average_temperature,
	ROUND(zone_1_total + zone_2_total + zone_3_total, 2) AS total_consumption
FROM daily_metrics
ORDER BY measurement_date;


-- 5. Hourly consumption with previous-hour comparison
WITH hourly_consumption AS (
	SELECT
	DATE_TRUNC('hour', measured_at) AS hour,
	SUM(zone_1_power_consumption + zone_2_power_consumption + zone_3_power_consumption) AS total_consumption
	FROM energy.power_measurements
	GROUP BY DATE_TRUNC('hour', measured_at)
)

SELECT
	hour,
	ROUND(total_consumption, 2) as total_consumption,
	ROUND(
		LAG(total_consumption) OVER (ORDER BY hour), 2
	) AS previous_hour_consumption,
	ROUND(
		total_consumption - LAG(total_consumption) OVER (ORDER BY hour), 2
	) AS change_from_previous_hour
FROM hourly_consumption
ORDER BY hour;