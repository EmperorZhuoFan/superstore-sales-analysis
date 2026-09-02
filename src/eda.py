import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def setup_visualization():
    sns.set_theme(style="whitegrid")
    plt.style.use("Solarize_Light2")
    plt.rcParams["figure.figsize"] = (10, 6)


def plot_profit_distribution(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Profit",
        bins=50,
        kde=True
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2
    )

    plt.title("Distribution of Transaction Profit")
    plt.xlabel("Profit")
    plt.ylabel("Number of Transactions")
    plt.show()


def plot_discount_vs_profit(df):
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="Discount",
        y="Profit",
        alpha=0.35
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=2
    )

    plt.title("Relationship Between Discount and Profit")
    plt.xlabel("Discount")
    plt.ylabel("Profit")
    plt.show()


def plot_profitability_distribution(df):
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Profit Status Label"
    )

    plt.title("Transaction Profitability Distribution")
    plt.xlabel("Transaction Outcome")
    plt.ylabel("Number of Transactions")
    plt.show()


def plot_discount_by_profitability(df):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Profit Status Label",
        y="Discount"
    )

    plt.title("Discount Distribution by Profitability")
    plt.xlabel("Transaction Outcome")
    plt.ylabel("Discount")
    plt.show()


def create_profit_summary(df):

    profit_summary = (
        df.groupby("Profit Status Label")
        .agg(
            Average_Sales=("Sales", "mean"),
            Average_Quantity=("Quantity", "mean"),
            Average_Discount=("Discount", "mean"),
            Average_Profit=("Profit", "mean")
        )
        .round(2)
    )

    display(profit_summary)

    return profit_summary


# ============================================================
# BUSINESS-FOCUSED EDA
# ============================================================


def analyze_profit_by_category(df):

    category_profit = (
        df.groupby("Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    print("Total Profit by Category:")
    display(category_profit)

    plt.figure(figsize=(10, 6))

    category_profit.plot(kind="bar")

    plt.title("Total Profit by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Profit")
    plt.xticks(rotation=0)

    plt.show()

    return category_profit


def analyze_profit_by_region(df):

    region_profit = (
        df.groupby("Region")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    print("Total Profit by Region:")
    display(region_profit)

    plt.figure(figsize=(10, 6))

    region_profit.plot(kind="bar")

    plt.title("Total Profit by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Profit")
    plt.xticks(rotation=0)

    plt.show()

    return region_profit


def analyze_profit_by_segment(df):

    segment_profit = (
        df.groupby("Segment")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    print("Total Profit by Segment:")
    display(segment_profit)

    plt.figure(figsize=(10, 6))

    segment_profit.plot(kind="bar")

    plt.title("Total Profit by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Total Profit")
    plt.xticks(rotation=0)

    plt.show()

    return segment_profit


def analyze_profit_by_subcategory(df):

    subcategory_profit = (
        df.groupby("Sub-Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    print("Total Profit by Sub-Category:")
    display(subcategory_profit)

    plt.figure(figsize=(10, 7))

    subcategory_profit.plot(kind="bar")

    plt.title("Total Profit by Sub-Category")
    plt.xlabel("Sub-Category")
    plt.ylabel("Total Profit")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.show()

    return subcategory_profit


def run_business_eda(df):

    category_profit = analyze_profit_by_category(df)

    region_profit = analyze_profit_by_region(df)

    segment_profit = analyze_profit_by_segment(df)

    subcategory_profit = analyze_profit_by_subcategory(df)

    return {
        "category_profit": category_profit,
        "region_profit": region_profit,
        "segment_profit": segment_profit,
        "subcategory_profit": subcategory_profit
    }


def run_eda(df):

    plot_profit_distribution(df)

    plot_discount_vs_profit(df)

    plot_profitability_distribution(df)

    plot_discount_by_profitability(df)

    profit_summary = create_profit_summary(df)

    business_results = run_business_eda(df)

    return {
        "profit_summary": profit_summary,
        "business_results": business_results
    }