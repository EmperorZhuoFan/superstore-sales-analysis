import pandas as pd

from src.data_preparation import prepare_data


def test_prepare_data_creates_expected_columns():
    df = pd.DataFrame({
        "Order Date": ["2024-01-01"],
        "Ship Date": ["2024-01-03"],
        "Sales": [100.0],
        "Quantity": [2],
        "Profit": [20.0],
    })

    prepared_df = prepare_data(df)

    assert "Profit Status" in prepared_df.columns
    assert "Order Year" in prepared_df.columns
    assert "Sales Per Unit" in prepared_df.columns

    # The original input should remain unchanged.
    assert "Profit Status" not in df.columns