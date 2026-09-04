
-- ============================================================
-- SUPERSTORE SALES ANALYSIS
-- 05 - ML SUPPORTING ANALYSIS
-- ============================================================


-- ------------------------------------------------------------
-- 1. Create a profitability label
--
-- This mirrors the Python ML target:
-- Profit > 0 = Profitable
-- Profit <= 0 = Unprofitable
-- ------------------------------------------------------------

SELECT
    "Row ID",
    "Sales",
    "Quantity",
    "Discount",
    "Category",
    "Sub-Category",
    "Segment",
    "Region",
    "Ship Mode",
    "Profit",

    CASE
        WHEN "Profit" > 0
            THEN 1
        ELSE 0
    END AS profit_status

FROM superstore
LIMIT 20;


-- ------------------------------------------------------------
-- 2. Overall profitability distribution
-- ------------------------------------------------------------

SELECT
    CASE
        WHEN "Profit" > 0
            THEN 'Profitable'
        ELSE 'Unprofitable'
    END AS profit_status,

    COUNT(*) AS transactions,

    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        2
    ) AS percentage

FROM superstore

GROUP BY
    CASE
        WHEN "Profit" > 0
            THEN 'Profitable'
        ELSE 'Unprofitable'
    END

ORDER BY transactions DESC;


-- ------------------------------------------------------------
-- 3. Profitability by category
-- ------------------------------------------------------------

SELECT
    "Category",

    COUNT(*) AS transactions,

    SUM(
        CASE
            WHEN "Profit" > 0 THEN 1
            ELSE 0
        END
    ) AS profitable_transactions,

    SUM(
        CASE
            WHEN "Profit" <= 0 THEN 1
            ELSE 0
        END
    ) AS unprofitable_transactions,

    ROUND(
        AVG(
            CASE
                WHEN "Profit" > 0 THEN 1.0
                ELSE 0.0
            END
        ) * 100,
        2
    ) AS profitability_rate

FROM superstore

GROUP BY "Category"

ORDER BY profitability_rate DESC;


-- ------------------------------------------------------------
-- 4. Profitability by sub-category
-- ------------------------------------------------------------

SELECT
    "Sub-Category",

    COUNT(*) AS transactions,

    ROUND(
        AVG(
            CASE
                WHEN "Profit" > 0 THEN 1.0
                ELSE 0.0
            END
        ) * 100,
        2
    ) AS profitability_rate,

    ROUND(AVG("Discount"), 3) AS average_discount,

    ROUND(AVG("Profit"), 2) AS average_profit

FROM superstore

GROUP BY "Sub-Category"

ORDER BY profitability_rate ASC;


-- ------------------------------------------------------------
-- 5. Discount vs profitability
-- ------------------------------------------------------------

SELECT
    "Discount",

    COUNT(*) AS transactions,

    ROUND(AVG("Profit"), 2) AS average_profit,

    ROUND(
        AVG(
            CASE
                WHEN "Profit" > 0 THEN 1.0
                ELSE 0.0
            END
        ) * 100,
        2
    ) AS profitability_rate

FROM superstore

GROUP BY "Discount"

ORDER BY "Discount";


-- ------------------------------------------------------------
-- 6. Profitability by region
-- ------------------------------------------------------------

SELECT
    "Region",

    COUNT(*) AS transactions,

    ROUND(
        AVG(
            CASE
                WHEN "Profit" > 0 THEN 1.0
                ELSE 0.0
            END
        ) * 100,
        2
    ) AS profitability_rate,

    ROUND(AVG("Profit"), 2) AS average_profit

FROM superstore

GROUP BY "Region"

ORDER BY profitability_rate DESC;


-- ------------------------------------------------------------
-- 7. Profitability by customer segment
-- ------------------------------------------------------------

SELECT
    "Segment",

    COUNT(*) AS transactions,

    ROUND(
        AVG(
            CASE
                WHEN "Profit" > 0 THEN 1.0
                ELSE 0.0
            END
        ) * 100,
        2
    ) AS profitability_rate,

    ROUND(AVG("Discount"), 3) AS average_discount,

    ROUND(AVG("Profit"), 2) AS average_profit

FROM superstore

GROUP BY "Segment"

ORDER BY profitability_rate DESC;


-- ------------------------------------------------------------
-- 8. Profitability by shipping mode
-- ------------------------------------------------------------

SELECT
    "Ship Mode",

    COUNT(*) AS transactions,

    ROUND(
        AVG(
            CASE
                WHEN "Profit" > 0 THEN 1.0
                ELSE 0.0
            END
        ) * 100,
        2
    ) AS profitability_rate,

    ROUND(AVG("Profit"), 2) AS average_profit

FROM superstore

GROUP BY "Ship Mode"

ORDER BY profitability_rate DESC;


-- ------------------------------------------------------------
-- 9. Sales and quantity compared with profitability
-- ------------------------------------------------------------

SELECT
    CASE
        WHEN "Profit" > 0
            THEN 'Profitable'
        ELSE 'Unprofitable'
    END AS profit_status,

    ROUND(AVG("Sales"), 2) AS average_sales,

    ROUND(AVG("Quantity"), 2) AS average_quantity,

    ROUND(AVG("Discount"), 3) AS average_discount,

    ROUND(AVG("Profit"), 2) AS average_profit

FROM superstore

GROUP BY
    CASE
        WHEN "Profit" > 0
            THEN 'Profitable'
        ELSE 'Unprofitable'
    END

ORDER BY profit_status;


-- ------------------------------------------------------------
-- 10. High-discount transactions
-- ------------------------------------------------------------

SELECT
    "Category",
    "Sub-Category",

    COUNT(*) AS transactions,

    ROUND(AVG("Discount"), 3) AS average_discount,

    ROUND(AVG("Profit"), 2) AS average_profit,

    ROUND(
        AVG(
            CASE
                WHEN "Profit" > 0 THEN 1.0
                ELSE 0.0
            END
        ) * 100,
        2
    ) AS profitability_rate

FROM superstore

WHERE "Discount" >= 0.30

GROUP BY
    "Category",
    "Sub-Category"

ORDER BY profitability_rate ASC;
