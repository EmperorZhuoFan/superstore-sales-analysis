import numpy as np
import pandas as pd
from IPython.display import display


# ============================================================
# LOAD DATA
# ============================================================
def load_data(file_path):
    df = pd.read_csv(file_path)
    df_original = df.copy()  # Keep an untouched copy of the original dataset
    print(f"Dataset shape: {df.shape}")
    display(df.head())
    return df, df_original


# ============================================================
# DATA UNDERSTANDING
# ============================================================
def inspect_data(df):
    print(" --- First 5 Rows ---")
    display(df.head())
    print("=" * 70)

    print(f" --- Rows and Columns ---\nRows: {df.shape[0]}\nColumns: {df.shape[1]}")
    print("=" * 70)

    print(f" --- Dataset Columns ---\n{df.columns.tolist()}")
    print("=" * 70)

    print(f" --- Dataset Data Types ---\n{df.dtypes}")
    print("=" * 70)

    print(" --- Summary Statistics ---")
    display(df.describe())
    print("=" * 70)

    print(" --- Missing Data ---")
    missing_values = df.isnull().sum()
    display(missing_values)
    print("=" * 70)

    print(" --- Missing Data Percentage ---")
    missing_percentage = (df.isnull().sum().div(len(df)).mul(100).round(2))
    display(missing_percentage)
    print("=" * 70)

    print(" --- Missing Data Report ---")
    missing_report = pd.DataFrame({"Missing Values": missing_values, "Missing %": missing_percentage})
    display(missing_report)
    print("=" * 70)

    print(f" --- Duplicated Rows ---\nDuplicated rows: {df.duplicated().sum()}")
    print("=" * 70)
    return df


# ============================================================
# DATE PREPARATION
# ============================================================
def prepare_dates(df):
    print(" --- Date Preparation ---")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    invalid_order_dates = df["Order Date"].isna().sum()
    invalid_ship_dates = df["Ship Date"].isna().sum()
    invalid_shipping_dates = (df["Ship Date"] < df["Order Date"]).sum()

    print(f"Invalid Order Dates: {invalid_order_dates}\nInvalid Ship Dates: {invalid_ship_dates}\nShip Dates Before Order Dates: {invalid_shipping_dates}")
    print("=" * 70)
    return df


# ============================================================
# CREATE TARGET VARIABLE
# ============================================================
def create_target(df):
    # 1 = Profitable | 0 = Unprofitable
    df["Profit Status"] = (df["Profit"] > 0).astype(int)
    df["Profit Status Label"] = df["Profit Status"].map({0: "Unprofitable", 1: "Profitable"})

    print(" --- Target Distribution ---")
    display(df["Profit Status Label"].value_counts())

    print("\nTarget Percentage:")
    display(df["Profit Status Label"].value_counts(normalize=True).mul(100).round(2))
    print("=" * 70)
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    # Date Features
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Order Day"] = df["Order Date"].dt.day
    df["Order Day of Week"] = df["Order Date"].dt.dayofweek

    # Sales Per Unit
    df["Sales Per Unit"] = df["Sales"] / df["Quantity"].replace(0, np.nan)
    df["Sales Per Unit"] = df["Sales Per Unit"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ============================================================
# COMPLETE DATA PREPARATION PIPELINE
# ============================================================
def prepare_data(df):
    df = prepare_dates(df)
    df = create_target(df)
    df = engineer_features(df)

    print(f" --- Data Preparation Completed ---\nPrepared Dataset Shape: {df.shape}")
    print("=" * 70)
    
    return df