import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def setup_visualization():
    sns.set_theme(style="whitegrid")
    plt.style.use("Solarize_Light2")
    plt.rcParams["figure.figsize"] = (10, 6)


def plot_profit_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Profit", bins=50, kde=True)
    plt.axvline(0, linestyle="--", linewidth=2)
    plt.title("Distribution of Transaction Profit")
    plt.xlabel("Profit")
    plt.ylabel("Number of Transactions")
    plt.show()


def plot_discount_vs_profit(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Discount", y="Profit", alpha=0.35)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.title("Relationship Between Discount and Profit")
    plt.xlabel("Discount")
    plt.ylabel("Profit")
    plt.show()


def plot_profitability_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Profit Status Label")
    plt.title("Transaction Profitability Distribution")
    plt.xlabel("Transaction Outcome")
    plt.ylabel("Number of Transactions")
    plt.show()


def plot_discount_by_profitability(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Profit Status Label", y="Discount")
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


def run_eda(df):
    plot_profit_distribution(df)
    plot_discount_vs_profit(df)
    plot_profitability_distribution(df)
    plot_discount_by_profitability(df)
    profit_summary = create_profit_summary(df)
    return profit_summary
