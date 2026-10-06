from multiprocessing import connection

import pandas as pd
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "database": os.getenv("POSTGRES_DATABASE"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}

CSV_PATH = Path("data/raw/tetouan_power_consumption.csv")

def main():
    df = pd.read_csv(CSV_PATH)
    
    print(f"Read {len(df):,} rows from {CSV_PATH}")

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()

    insert_query = """
        INSERT INTO energy.power_measurements (
            measured_at,
            temperature,
            humidity,
            wind_speed,
            general_diffuse_flows,
            diffuse_flows,
            zone_1_power_consumption,
            zone_2_power_consumption,
            zone_3_power_consumption
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    rows = [
        (
                row["DateTime"],
                row["Temperature"],
                row["Humidity"],
                row["Wind Speed"],
                row["general diffuse flows"],
                row["diffuse flows"],
                row["Zone 1 Power Consumption"],
                row["Zone 2  Power Consumption"],
                row["Zone 3  Power Consumption"],
            )
            for _, row in df.iterrows()
        ]

    cursor.executemany(insert_query, rows)
    connection.commit()

    print(f"Inserted {len(rows):,} rows into the database.")

    cursor.close()
    connection.close()

if __name__ == "__main__":
    main()