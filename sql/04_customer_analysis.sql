
-- ============================================================
-- SUPERSTORE SALES ANALYSIS
-- 04 - CUSTOMER ANALYSIS
-- ============================================================


-- ------------------------------------------------------------
-- 1. Customer sales and profit
-- ------------------------------------------------------------

SELECT
    "Customer ID",
    "Customer Name",
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit,
    SUM("Quantity") AS total_quantity
FROM superstore
GROUP BY
    "Customer ID",
    "Customer Name"
ORDER BY total_sales DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 2. Customer order frequency
-- ------------------------------------------------------------

SELECT
    "Customer ID",
    "Customer Name",
    COUNT(DISTINCT "Order ID") AS number_of_orders,
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit
FROM superstore
GROUP BY
    "Customer ID",
    "Customer Name"
ORDER BY number_of_orders DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 3. Average discount by customer
-- ------------------------------------------------------------

SELECT
    "Customer ID",
    "Customer Name",
    ROUND(AVG("Discount"), 3) AS average_discount,
    ROUND(SUM("Sales"), 2) AS total_sales,
    ROUND(SUM("Profit"), 2) AS total_profit
FROM superstore
GROUP BY
    "Customer ID",
    "Customer Name"
ORDER BY average_discount DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 4. Customer-level features
--
-- These are closely related to the features used by
-- the Python K-Means clustering workflow.
-- ------------------------------------------------------------

SELECT
    "Customer ID",

    ROUND(SUM("Sales"), 2) AS total_sales,

    ROUND(SUM("Profit"), 2) AS total_profit,

    SUM("Quantity") AS total_quantity,

    COUNT(DISTINCT "Order ID") AS number_of_orders,

    ROUND(AVG("Discount"), 3) AS average_discount,

    ROUND(
        SUM("Sales") /
        NULLIF(COUNT(DISTINCT "Order ID"), 0),
        2
    ) AS average_order_value

FROM superstore

GROUP BY "Customer ID"

ORDER BY total_sales DESC;


-- ------------------------------------------------------------
-- 5. Customer profitability
-- ------------------------------------------------------------

SELECT
    "Customer ID",
    "Customer Name",

    ROUND(SUM("Sales"), 2) AS total_sales,

    ROUND(SUM("Profit"), 2) AS total_profit,

    CASE
        WHEN SUM("Profit") > 0
            THEN 'Profitable'
        ELSE 'Unprofitable'
    END AS profit_status

FROM superstore

GROUP BY
    "Customer ID",
    "Customer Name"

ORDER BY total_profit DESC;
