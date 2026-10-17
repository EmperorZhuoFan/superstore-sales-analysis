
import great_expectations as gx
import pandas as pd


def validate_data_quality(df: pd.DataFrame):
   
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(
        name="superstore_data_source"
    )
    data_asset = data_source.add_dataframe_asset(
        name="prepared_superstore"
    )
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="prepared_superstore_batch"
    )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    suite = gx.ExpectationSuite(name="superstore_quality_suite")

    required_columns = [
        "Row ID", "Order ID", "Order Date", "Ship Date",
        "Sales", "Quantity", "Discount", "Profit",
        "Profit Status", "Profit Status Label",
        "Order Year", "Order Month", "Order Quarter",
        "Order Day", "Order Day of Week", "Sales Per Unit",
    ]

    for column in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=column)
        )

    for column in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=column
            )
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Sales", min_value=0, strict_min=True
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Quantity", min_value=1
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Discount", min_value=0, max_value=1
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Order Month", min_value=1, max_value=12
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Order Quarter", min_value=1, max_value=4
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Order Day of Week", min_value=0, max_value=6
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Profit Status", value_set={0, 1}
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Profit Status Label",
            value_set={"Profitable", "Unprofitable"},
        )
    )

    results = batch.validate(expect=suite)

    return {
        "success": results.success,
        "results": results,
    }
