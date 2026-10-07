import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT")),
    "database": os.getenv("POSTGRES_DATABASE"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD")
}


def load_data():
    query = """
    SELECT
        measured_at,
        temperature,
        humidity,
        wind_speed,
        general_diffuse_flows,
        diffuse_flows,
        zone_1_power_consumption,
        zone_2_power_consumption,
        zone_3_power_consumption
    FROM energy.power_measurements
    ORDER BY measured_at;
    """

    connection = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql(query, connection)
    df = df.drop_duplicates(subset=["measured_at"])
    connection.close()

    return df


def create_time_features(df):
    df["hour"] = df["measured_at"].dt.hour
    df["day_of_week"] = df["measured_at"].dt.dayofweek
    df["month"] = df["measured_at"].dt.month
    df["year"] = df["measured_at"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def create_energy_features(df):
    df["total_power_consumption"] = (
        df["zone_1_power_consumption"] +
        df["zone_2_power_consumption"] +
        df["zone_3_power_consumption"]
    )

    df["zone_1_share"] = ( 
        df["zone_1_power_consumption"] 
        / df["total_power_consumption"]
    )

    df["zone_2_share"] = (
        df["zone_2_power_consumption"]
        / df["total_power_consumption"]
    )

    df["zone_3_share"] = (
        df["zone_3_power_consumption"]
        / df["total_power_consumption"]
    )
    
    return df


def create_lag_features(df):
    df["lag_1"] = df["total_power_consumption"].shift(1)
    df["lag_5"] = df["total_power_consumption"].shift(6)
    df["lag_144"] = df["total_power_consumption"].shift(144)
    
    return df


def create_rolling_features(df):
    df["rolling_1h_mean"] = (
        df["total_power_consumption"].rolling(window=6).mean()
    )
    return df

    df["rolling_6h_mean"] = (
        df["total_power_consumption"].rolling(window=36).mean()
    )

    df["rolling_24h_mean"] = (
        df["total_power_consumption"].rolling(window=144).mean()
    )

    return df


def create_forecasting_target(df):
    # Creating 1 hour ahead electricity demand target
    # data is 10 minutes frequency, so 6 rows = 1 hour

    df["target_1h_ahead"] = df["total_power_consumption"].shift(-6)
    
    return df


def main():
    df = load_data()
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns before feature engineering: {len(df.columns):,}")

    df = create_time_features(df)
    df = create_energy_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_forecasting_target(df)

    print(f"Columns after feature engineering: {len(df.columns):,}")
    print(f"\nFeature Engineering Results:\n{df.head()}")

    print("\nNew features added:")
    
    new_features = ["hour", "day_of_week", "month", "year", "is_weekend", 
                    "total_power_consumption", "zone_1_share", "zone_2_share", "zone_3_share",
                    "lag_1", "lag_5", "lag_144", 
                    "rolling_1h_mean", "rolling_6h_mean", "rolling_24h_mean"]
    
    for feature in new_features:
        print(f" - {feature}")

    df = df.dropna(subset=["target_1h_ahead"])
    
    print(df[["measured_at", "total_power_consumption", "target_1h_ahead"]].head(10))
    print(f"\nRows after target creation: {len(df)}")
    print(f"Missing targets: {df['target_1h_ahead'].isna().sum()}")


if __name__ == "__main__":
    main() 