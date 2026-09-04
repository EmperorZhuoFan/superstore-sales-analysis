-- ============================================================
-- SUPERSTORE SALES ANALYSIS
-- 03 - SALES & PROFIT ANALYSIS
-- ============================================================


-- ------------------------------------------------------------
-- 1. Sales and profit by category
-- ------------------------------------------------------------

SELECT
    "Category",
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit,
    ROUND(AVG("Profit"), 2) AS average_profit
FROM superstore
GROUP BY "Category"
ORDER BY total_sales DESC;


-- ------------------------------------------------------------
-- 2. Sales and profit by sub-category
-- ------------------------------------------------------------

SELECT
    "Sub-Category",
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit,
    ROUND(AVG("Profit"), 2) AS average_profit
FROM superstore
GROUP BY "Sub-Category"
ORDER BY total_profit DESC;


-- ------------------------------------------------------------
-- 3. Sales and profit by region
-- ------------------------------------------------------------

SELECT
    "Region",
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit,
    ROUND(AVG("Profit"), 2) AS average_profit
FROM superstore
GROUP BY "Region"
ORDER BY total_profit DESC;


-- ------------------------------------------------------------
-- 4. Sales and profit by customer segment
-- ------------------------------------------------------------

SELECT
    "Segment",
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit,
    COUNT(DISTINCT "Customer ID") AS customers
FROM superstore
GROUP BY "Segment"
ORDER BY total_sales DESC;


-- ------------------------------------------------------------
-- 5. Monthly sales and profit
-- ------------------------------------------------------------

SELECT
    DATE_TRUNC('month', "Order Date") AS month,
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit
FROM superstore
GROUP BY DATE_TRUNC('month', "Order Date")
ORDER BY month;


-- ------------------------------------------------------------
-- 6. Yearly sales and profit
-- ------------------------------------------------------------

SELECT
    EXTRACT(YEAR FROM "Order Date") AS order_year,
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit
FROM superstore
GROUP BY EXTRACT(YEAR FROM "Order Date")
ORDER BY order_year;


-- ------------------------------------------------------------
-- 7. Most profitable products
-- ------------------------------------------------------------

SELECT
    "Product ID",
    "Product Name",
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit
FROM superstore
GROUP BY
    "Product ID",
    "Product Name"
ORDER BY total_profit DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 8. Least profitable products
-- ------------------------------------------------------------

SELECT
    "Product ID",
    "Product Name",
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit
FROM superstore
GROUP BY
    "Product ID",
    "Product Name"
ORDER BY total_profit ASC
LIMIT 10;
