from ucimlrepo import fetch_ucirepo
from pathlib import Path

def main():
    dataset = fetch_ucirepo(id=849)
    data = dataset.data.original
    output_path = Path("data/raw/tetouan_power_consumption.csv")
    data.to_csv(output_path, index=False)
    
    print(f"Saved {len(data):,} rows to {output_path}")
    print("\nColumns:")
    print(data.columns.to_list())

if __name__ == "__main__":
    main()