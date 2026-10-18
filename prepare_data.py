from src.data_preparation import (
    load_data,
    prepare_data,
    save_processed_data_to_device,
)

def main():
    input_path = "data/raw/superstore_v1.csv"
    output_path = "data/processed/superstore_v1.parquet"

    df, _ = load_data(input_path)
    df = prepare_data(df)
    save_processed_data_to_device(df, output_path)

if __name__ == "__main__":
    main()