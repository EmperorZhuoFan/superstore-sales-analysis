import pandas as pd
import numpy as np
from IPython.display import display


def load_data(file_path):
    df = pd.read_csv(file_path)
    df_original = df.copy()
    print(f"Dataset shape: {df.shape}")
    display(df.head())
    return df, df_original


def inspect_data(df):
    print(" --- the first 5 rows ---")
    print(df.head())
    print("=" * 70)

    print(" --- Rows and Columns ---")
    print(f"Rows are {df.shape[0]} and Columns are {df.shape[1]}")
    print("=" * 70)

    print(" --- What columns in Dataset ---")
    print(df.columns.tolist())
    print("=" * 70)

    print(" --- Dataset types  ---")
    print(df.dtypes.to_string())
    print("=" * 70)

    print(" --- Summary Statistics  ---")
    print(df.describe())
    print("=" * 70)

    print(" --- Missing Data  ---")
    print(df.isnull().sum())
    print("=" * 70)

    print(" --- Missing Data Percentage  ---")
    print(f"{((df.isnull().sum() / len(df)) * 100).round(2)}")
    print("=" * 70)

    missing = df.isnull().sum()
    missing_report = pd.DataFrame({
        "Missing Values": missing,
        "Missing %": ((missing / df.shape[0]) * 100).round(2)
    })
    print(missing_report)

    print(" --- Duplicated Rows ---")
    print(df.duplicated().sum())
    print("=" * 70)

    return df


def prepare_dates(df):
    print(" --- DateTime Columns ---")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    print(f"Invalid Order Dates: {df['Order Date'].isna().sum()}")
    print("=" * 70)

    print(f"Invalid Ship Dates: {df['Ship Date'].isna().sum()}")
    print("=" * 70)

    invalid_shipping_dates = (df["Ship Date"] < df["Order Date"]).sum()
    print(f"Ship dates before order dates: {invalid_shipping_dates}")

    return df


def create_target(df):
    df["Profit Status"] = (df["Profit"] > 0).astype(int)
    df["Profit Status Label"] = df["Profit Status"].map({
        0: "Unprofitable",
        1: "Profitable"
    })

    print("Target distribution:")
    display(df["Profit Status Label"].value_counts())

    print("\nTarget percentage:")
    display(df["Profit Status Label"].value_counts(normalize=True).mul(100).round(2))

    return df


def engineer_features(df):
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Order Day"] = df["Order Date"].dt.day
    df["Order Day of Week"] = df["Order Date"].dt.dayofweek

    df["Sales Per Unit"] = (
        df["Sales"] / df["Quantity"].replace(0, np.nan)
    )

    df["Sales Per Unit"] = (
        df["Sales Per Unit"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df


def prepare_data(df):
    df = prepare_dates(df)
    df = create_target(df)
    df = engineer_features(df)
    return df
