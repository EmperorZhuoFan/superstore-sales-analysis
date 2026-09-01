# ================================================================
# SAMPLE SUPERSTORE — COMPLETE EDA PROJECT
# Phase 2 — Data Science / ML Preparation
# ================================================================


# ================================================================
# STEP 1: DATA READING
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

plt.style.use("Solarize_Light2")

df = pd.read_csv(Path(__file__).resolve().parent / "data" / "samplesuperstore.csv")
df_original = df.copy()

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# ================================================================
# STEP 2: DATA UNDERSTANDING
# ================================================================

print("\n" + "=" * 70)
print("DATA UNDERSTANDING")
print("=" * 70)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Dataset Information ---")
df.info()

print("\n--- Numerical Summary ---")
print(df.describe())

print("\n--- Categorical Summary ---")
print(df.describe(include="object"))

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())


# ================================================================
# STEP 3: DATA CLEANING
# ================================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)


# ------------------------------------------------
# 3.1 Missing Values
# ------------------------------------------------

missing = df.isnull().sum()
missing_percentage = (missing / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Values": missing,
    "Missing %": missing_percentage.round(2)
})

print("\n--- Missing Data Report ---")
print(missing_report)


# ------------------------------------------------
# 3.2 Duplicate Rows
# ------------------------------------------------

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate rows after cleaning:", df.duplicated().sum())


# ------------------------------------------------
# 3.3 Correct Data Types
# ------------------------------------------------

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

print("\n--- Date Data Types ---")
print(df[["Order Date", "Ship Date"]].dtypes)


# ------------------------------------------------
# 3.4 Invalid Dates
# ------------------------------------------------

print("\nInvalid Order Dates:", df["Order Date"].isna().sum())
print("Invalid Ship Dates:", df["Ship Date"].isna().sum())


# ------------------------------------------------
# 3.5 Numerical Data Check
# ------------------------------------------------

print("\n--- Numerical Summary ---")
print(df[["Sales", "Quantity", "Discount", "Profit"]].describe())


# ------------------------------------------------
# 3.6 Logical Checks
# ------------------------------------------------

print("\n--- Logical Checks ---")

print("Negative Sales:", (df["Sales"] < 0).sum())
print("Negative Quantity:", (df["Quantity"] < 0).sum())
print("Discount below 0:", (df["Discount"] < 0).sum())
print("Discount above 1:", (df["Discount"] > 1).sum())
print("Ship Date before Order Date:", (df["Ship Date"] < df["Order Date"]).sum())


# ================================================================
# STEP 3.5: OUTLIER DETECTION
# ================================================================

print("\n" + "=" * 70)
print("OUTLIER DETECTION")
print("=" * 70)


def detect_outliers(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"\n--- {column} ---")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)
    print("Outliers:", len(outliers))

    return outliers


sales_outliers = detect_outliers("Sales")
profit_outliers = detect_outliers("Profit")


# ================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ================================================================
# 4.1 UNIVARIATE ANALYSIS
# ================================================================

print("\n" + "-" * 70)
print("UNIVARIATE ANALYSIS")
print("-" * 70)


# ------------------------------------------------
# Numerical Statistics
# ------------------------------------------------

print("\n--- Sales ---")
print(df["Sales"].describe())

print("\n--- Quantity ---")
print(df["Quantity"].describe())

print("\n--- Discount ---")
print(df["Discount"].describe())

print("\n--- Profit ---")
print(df["Profit"].describe())


# ------------------------------------------------
# Sales Distribution
# ------------------------------------------------

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Sales", bins=30)
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()


# ------------------------------------------------
# Profit Distribution
# ------------------------------------------------

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Profit", bins=30)
plt.title("Distribution of Profit")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.show()


# ------------------------------------------------
# Sales Outliers
# ------------------------------------------------

plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x="Sales")
plt.title("Sales Distribution and Outliers")
plt.show()


# ------------------------------------------------
# Profit Outliers
# ------------------------------------------------

plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x="Profit")
plt.title("Profit Distribution and Outliers")
plt.show()


# ------------------------------------------------
# Categorical Variables
# ------------------------------------------------

print("\n--- Category ---")
print(df["Category"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Category")
plt.title("Category Distribution")
plt.show()


print("\n--- Region ---")
print(df["Region"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Region")
plt.title("Region Distribution")
plt.show()


print("\n--- Segment ---")
print(df["Segment"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Segment")
plt.title("Segment Distribution")
plt.show()


print("\n--- Ship Mode ---")
print(df["Ship Mode"].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Ship Mode")
plt.title("Ship Mode Distribution")
plt.show()


# ================================================================
# 4.2 BIVARIATE ANALYSIS
# ================================================================

print("\n" + "-" * 70)
print("BIVARIATE ANALYSIS")
print("-" * 70)


# ------------------------------------------------
# Discount vs Profit
# ------------------------------------------------

discount_profit_corr = df["Discount"].corr(df["Profit"])

print("Discount vs Profit Correlation:", round(discount_profit_corr, 3))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Discount", y="Profit", alpha=0.4)
sns.regplot(data=df, x="Discount", y="Profit", scatter=False)
plt.title(f"Discount vs Profit — Correlation: {discount_profit_corr:.2f}")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.show()


# ------------------------------------------------
# Sales vs Profit
# ------------------------------------------------

sales_profit_corr = df["Sales"].corr(df["Profit"])

print("Sales vs Profit Correlation:", round(sales_profit_corr, 3))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Sales", y="Profit", alpha=0.4)
sns.regplot(data=df, x="Sales", y="Profit", scatter=False)
plt.title(f"Sales vs Profit — Correlation: {sales_profit_corr:.2f}")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()


# ------------------------------------------------
# Category vs Sales
# ------------------------------------------------

category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

print("\nSales by Category:")
print(category_sales)

plt.figure(figsize=(9, 5))
sns.barplot(data=category_sales.reset_index(), x="Category", y="Sales")
plt.title("Total Sales by Category")
plt.ylabel("Sales")
plt.show()


# ------------------------------------------------
# Category vs Profit
# ------------------------------------------------

category_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)

print("\nProfit by Category:")
print(category_profit)

plt.figure(figsize=(9, 5))
sns.barplot(data=category_profit.reset_index(), x="Category", y="Profit")
plt.title("Total Profit by Category")
plt.ylabel("Profit")
plt.show()


# ------------------------------------------------
# Category vs Sales Distribution
# ------------------------------------------------

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Category", y="Sales")
plt.title("Sales Distribution by Category")
plt.show()


# ------------------------------------------------
# Category vs Profit Distribution
# ------------------------------------------------

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Category", y="Profit")
plt.title("Profit Distribution by Category")
plt.show()


# ------------------------------------------------
# Categorical vs Categorical
# ------------------------------------------------

category_region = pd.crosstab(df["Category"], df["Region"])

print("\nCategory vs Region:")
print(category_region)

plt.figure(figsize=(9, 5))
sns.heatmap(category_region, annot=True, fmt="d")
plt.title("Category vs Region")
plt.xlabel("Region")
plt.ylabel("Category")
plt.show()


# ================================================================
# 4.3 MULTIVARIATE ANALYSIS
# ================================================================

print("\n" + "-" * 70)
print("MULTIVARIATE ANALYSIS")
print("-" * 70)


# ------------------------------------------------
# Sales + Profit + Category
# ------------------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Sales", y="Profit", hue="Category", alpha=0.6)
plt.title("Sales vs Profit by Category")
plt.show()


# ------------------------------------------------
# Sales + Profit + Discount
# ------------------------------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Sales", y="Profit", hue="Discount", alpha=0.6)
plt.title("Sales vs Profit with Discount")
plt.show()


# ------------------------------------------------
# Region + Category + Profit
# ------------------------------------------------

region_category_profit = df.groupby(["Region", "Category"])["Profit"].sum().unstack()

print("\nProfit by Region and Category:")
print(region_category_profit)

plt.figure(figsize=(10, 6))
sns.heatmap(region_category_profit, annot=True, fmt=".0f")
plt.title("Profit by Region and Category")
plt.xlabel("Category")
plt.ylabel("Region")
plt.show()


# ------------------------------------------------
# Pivot Table
# ------------------------------------------------

pivot = df.pivot_table(
    values="Profit",
    index="Region",
    columns="Category",
    aggfunc="sum"
)

print("\nPivot Table:")
print(pivot)


# ------------------------------------------------
# Pairplot
# ------------------------------------------------

sns.pairplot(df[["Sales", "Quantity", "Discount", "Profit"]])
plt.show()


# ================================================================
# STEP 5: FEATURE ENGINEERING
# ================================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------
# 5.1 Date Features
# ------------------------------------------------

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Month Name"] = df["Order Date"].dt.month_name()
df["Quarter"] = df["Order Date"].dt.quarter
df["Day of Week"] = df["Order Date"].dt.day_name()

print("\nDate Features Created:")
print(df[["Order Date", "Year", "Month", "Month Name", "Quarter", "Day of Week"]].head())


# ------------------------------------------------
# 5.2 Shipping Days
# ------------------------------------------------

df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

print("\nShipping Days:")
print(df["Shipping Days"].describe())


# ------------------------------------------------
# 5.3 Profit Margin
# ------------------------------------------------

df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)
df["Profit Margin %"] = df["Profit Margin"] * 100

print("\nProfit Margin:")
print(df[["Sales", "Profit", "Profit Margin", "Profit Margin %"]].head())


# ------------------------------------------------
# 5.4 Sales Per Unit
# ------------------------------------------------

df["Sales Per Unit"] = np.where(df["Quantity"] != 0, df["Sales"] / df["Quantity"], np.nan)

print("\nSales Per Unit:")
print(df[["Sales", "Quantity", "Sales Per Unit"]].head())


# ------------------------------------------------
# 5.5 Customer Total Sales
# ------------------------------------------------

df["Customer Total Sales"] = df.groupby("Customer Name")["Sales"].transform("sum")

print("\nCustomer Total Sales:")
print(df[["Customer Name", "Customer Total Sales"]].head())


# ------------------------------------------------
# 5.6 Profit Status
# ------------------------------------------------

df["Profit Status"] = np.where(df["Profit"] >= 0, "Profit", "Loss")

print("\nProfit Status:")
print(df["Profit Status"].value_counts())


# ================================================================
# STEP 6: FEATURE ENGINEERING ANALYSIS
# ================================================================


# ------------------------------------------------
# Sales by Quarter
# ------------------------------------------------

quarter_sales = df.groupby("Quarter")["Sales"].sum().sort_index()

print("\nSales by Quarter:")
print(quarter_sales.round(0))

plt.figure(figsize=(8, 5))
sns.barplot(data=quarter_sales.reset_index(), x="Quarter", y="Sales")
plt.title("Total Sales by Quarter")
plt.show()


# ------------------------------------------------
# Average Shipping Days by Ship Mode
# ------------------------------------------------

shipping_by_mode = df.groupby("Ship Mode")["Shipping Days"].mean().sort_values()

print("\nAverage Shipping Days by Ship Mode:")
print(shipping_by_mode.round(2))

plt.figure(figsize=(9, 5))
sns.barplot(data=shipping_by_mode.reset_index(), x="Ship Mode", y="Shipping Days")
plt.title("Average Shipping Days by Ship Mode")
plt.ylabel("Average Shipping Days")
plt.show()


# ------------------------------------------------
# Profit Margin by Category
# ------------------------------------------------

margin_by_category = df.groupby("Category")["Profit Margin"].mean().sort_values(ascending=False)

print("\nAverage Profit Margin by Category:")
print((margin_by_category * 100).round(2))

plt.figure(figsize=(9, 5))
sns.barplot(data=margin_by_category.reset_index(), x="Category", y="Profit Margin")
plt.title("Average Profit Margin by Category")
plt.ylabel("Profit Margin")
plt.show()


# ================================================================
# STEP 7: PROFIT & LOSS ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("PROFIT & LOSS ANALYSIS")
print("=" * 70)

profits = df.loc[df["Profit"] >= 0, "Profit"].sum()
losses = df.loc[df["Profit"] < 0, "Profit"].sum()

print(f"Total Profits: ${profits:,.0f}")
print(f"Total Losses: ${-losses:,.0f}")
print(f"Net Profit: ${df['Profit'].sum():,.0f}")

print("Profitable Orders:", (df["Profit"] >= 0).sum())
print("Loss-Making Orders:", (df["Profit"] < 0).sum())


# ================================================================
# STEP 8: BUSINESS QUESTIONS
# ================================================================

print("\n" + "=" * 70)
print("BUSINESS QUESTIONS")
print("=" * 70)


# ------------------------------------------------
# Q1: Which category generates the most sales?
# ------------------------------------------------

category_sales_result = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

print("\n1. Sales by Category:")
print(category_sales_result)


# ------------------------------------------------
# Q2: Which category generates the most profit?
# ------------------------------------------------

category_profit_result = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)

print("\n2. Profit by Category:")
print(category_profit_result)


# ------------------------------------------------
# Q3: Which region is most profitable?
# ------------------------------------------------

region_profit = df.groupby("Region")["Profit"].sum().sort_values(ascending=False)

print("\n3. Profit by Region:")
print(region_profit)


# ------------------------------------------------
# Q4: Which sub-categories are losing money?
# ------------------------------------------------

subcategory_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values()

print("\n4. Sub-Categories by Profit:")
print(subcategory_profit)


# ------------------------------------------------
# Q5: Which customer segment generates the most profit?
# ------------------------------------------------

segment_profit = df.groupby("Segment")["Profit"].sum().sort_values(ascending=False)

print("\n5. Profit by Segment:")
print(segment_profit)


# ------------------------------------------------
# Q6: Top and Bottom States
# ------------------------------------------------

state_profit = df.groupby("State/Province")["Profit"].sum().sort_values()

print("\n6. Bottom 10 States:")
print(state_profit.head(10))

print("\nTop 10 States:")
print(state_profit.tail(10).sort_values(ascending=False))


# ------------------------------------------------
# Q7: Does discounting affect profitability?
# ------------------------------------------------

discount_analysis = df.groupby("Discount")["Profit"].agg(
    Average_Profit="mean",
    Total_Profit="sum",
    Orders="count"
)

print("\n7. Discount vs Profit:")
print(discount_analysis)


# ------------------------------------------------
# Q8: High Sales but Weak Profit
# ------------------------------------------------

subcategory_performance = df.groupby("Sub-Category").agg(
    Sales=("Sales", "sum"),
    Profit=("Profit", "sum"),
    Orders=("Order ID", "count")
).sort_values("Sales", ascending=False)

print("\n8. Sub-Category Performance:")
print(subcategory_performance)


# ================================================================
# STEP 9: OVERALL BUSINESS PERFORMANCE
# ================================================================

print("\n" + "=" * 70)
print("OVERALL BUSINESS PERFORMANCE")
print("=" * 70)

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
overall_margin = total_profit / total_sales

print(f"Total Sales: ${total_sales:,.0f}")
print(f"Total Profit: ${total_profit:,.0f}")
print(f"Overall Profit Margin: {overall_margin:.2%}")
print("Unique Orders:", df["Order ID"].nunique())
print("Unique Customers:", df["Customer Name"].nunique())
print("Unique Products:", df["Product Name"].nunique())


# ================================================================
# STEP 10: FINAL PERFORMANCE TABLES
# ================================================================

print("\n" + "=" * 70)
print("FINAL PERFORMANCE TABLES")
print("=" * 70)


# ------------------------------------------------
# Category Performance
# ------------------------------------------------

category_performance = df.groupby("Category").agg(
    Sales=("Sales", "sum"),
    Profit=("Profit", "sum"),
    Orders=("Order ID", "count")
)

category_performance["Profit Margin"] = (
    category_performance["Profit"] /
    category_performance["Sales"]
)

print("\nCategory Performance:")
print(category_performance)


# ------------------------------------------------
# Regional Performance
# ------------------------------------------------

region_performance = df.groupby("Region").agg(
    Sales=("Sales", "sum"),
    Profit=("Profit", "sum"),
    Orders=("Order ID", "count")
)

region_performance["Profit Margin"] = (
    region_performance["Profit"] /
    region_performance["Sales"]
)

print("\nRegional Performance:")
print(region_performance)


# ================================================================
# STEP 11: FINAL VISUAL STORY
# ================================================================


# ------------------------------------------------
# Sales by Category
# ------------------------------------------------

plt.figure(figsize=(9, 5))
sns.barplot(data=category_performance.reset_index(), x="Category", y="Sales")
plt.title("Total Sales by Category")
plt.ylabel("Sales")
plt.show()


# ------------------------------------------------
# Profit by Category
# ------------------------------------------------

plt.figure(figsize=(9, 5))
sns.barplot(data=category_performance.reset_index(), x="Category", y="Profit")
plt.title("Total Profit by Category")
plt.ylabel("Profit")
plt.show()


# ------------------------------------------------
# Profit by Region
# ------------------------------------------------

plt.figure(figsize=(9, 5))
sns.barplot(data=region_performance.reset_index(), x="Region", y="Profit")
plt.title("Total Profit by Region")
plt.ylabel("Profit")
plt.show()


# ------------------------------------------------
# Profit Margin by Category
# ------------------------------------------------

plt.figure(figsize=(9, 5))
sns.barplot(data=category_performance.reset_index(), x="Category", y="Profit Margin")
plt.title("Profit Margin by Category")
plt.ylabel("Profit Margin")
plt.show()


# ================================================================
# STEP 12: FINAL CONCLUSIONS
# ================================================================

best_category = category_performance["Profit"].idxmax()
worst_category = category_performance["Profit"].idxmin()

best_region = region_performance["Profit"].idxmax()
worst_region = region_performance["Profit"].idxmin()

best_subcategory = subcategory_profit.idxmax()
worst_subcategory = subcategory_profit.idxmin()

print("\n" + "=" * 70)
print("FINAL CONCLUSIONS")
print("=" * 70)

print(f"Most profitable category: {best_category}")
print(f"Least profitable category: {worst_category}")
print(f"Most profitable region: {best_region}")
print(f"Least profitable region: {worst_region}")
print(f"Most profitable sub-category: {best_subcategory}")
print(f"Least profitable sub-category: {worst_subcategory}")
print(f"Total Sales: ${total_sales:,.0f}")
print(f"Total Profit: ${total_profit:,.0f}")
print(f"Overall Profit Margin: {overall_margin:.2%}")


# ================================================================
# PROJECT COMPLETE
# ================================================================

print("\n" + "=" * 70)
print("SAMPLE SUPERSTORE EDA PROJECT COMPLETED")
print("=" * 70)