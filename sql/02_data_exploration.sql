-- ============================================================
-- SUPERSTORE SALES ANALYSIS
-- 02 - DATA EXPLORATION
-- ============================================================


-- ------------------------------------------------------------
-- 1. Number of transactions
-- ------------------------------------------------------------

SELECT
    COUNT(*) AS total_transactions
FROM superstore;


-- ------------------------------------------------------------
-- 2. Number of customers
-- ------------------------------------------------------------

SELECT
    COUNT(DISTINCT "Customer ID") AS total_customers
FROM superstore;


-- ------------------------------------------------------------
-- 3. Number of products
-- ------------------------------------------------------------

SELECT
    COUNT(DISTINCT "Product ID") AS total_products
FROM superstore;


-- ------------------------------------------------------------
-- 4. Date range
-- ------------------------------------------------------------

SELECT
    MIN("Order Date") AS first_order,
    MAX("Order Date") AS last_order
FROM superstore;


-- ------------------------------------------------------------
-- 5. Basic sales and profit statistics
-- ------------------------------------------------------------

SELECT
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit,
    ROUND(AVG("Sales"), 2) AS average_sales,
    ROUND(AVG("Profit"), 2) AS average_profit,
    ROUND(AVG("Discount"), 3) AS average_discount
FROM superstore;


-- ------------------------------------------------------------
-- 6. Categories
-- ------------------------------------------------------------

SELECT DISTINCT
    "Category"
FROM superstore
ORDER BY "Category";


-- ------------------------------------------------------------
-- 7. Regions
-- ------------------------------------------------------------

SELECT DISTINCT
    "Region"
FROM superstore
ORDER BY "Region";


-- ------------------------------------------------------------
-- 8. Check missing values in important columns
-- ------------------------------------------------------------

SELECT
    COUNT(*) FILTER (WHERE "Sales" IS NULL) AS missing_sales,
    COUNT(*) FILTER (WHERE "Profit" IS NULL) AS missing_profit,
    COUNT(*) FILTER (WHERE "Quantity" IS NULL) AS missing_quantity,
    COUNT(*) FILTER (WHERE "Discount" IS NULL) AS missing_discount,
    COUNT(*) FILTER (WHERE "Category" IS NULL) AS missing_category,
    COUNT(*) FILTER (WHERE "Customer ID" IS NULL) AS missing_customer_id
FROM superstore;


-- ------------------------------------------------------------
-- 9. Check duplicate rows
-- ------------------------------------------------------------

SELECT
    COUNT(*) AS duplicate_rows
FROM (
    SELECT
        "Row ID"
    FROM superstore
    GROUP BY "Row ID"
    HAVING COUNT(*) > 1
) AS duplicates;
