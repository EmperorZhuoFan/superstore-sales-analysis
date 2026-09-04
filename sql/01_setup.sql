-- ============================================================
-- SUPERSTORE SALES ANALYSIS
-- 01 - DATABASE SETUP
-- PostgreSQL
-- ============================================================

-- Create the main Superstore table
CREATE TABLE IF NOT EXISTS superstore (
    "Row ID" INTEGER,
    "Order ID" VARCHAR(50),
    "Order Date" DATE,
    "Ship Date" DATE,
    "Ship Mode" VARCHAR(50),
    "Customer ID" VARCHAR(50),
    "Customer Name" VARCHAR(150),
    "Segment" VARCHAR(50),
    "Country" VARCHAR(100),
    "City" VARCHAR(100),
    "State" VARCHAR(100),
    "Postal Code" VARCHAR(20),
    "Region" VARCHAR(50),
    "Product ID" VARCHAR(50),
    "Category" VARCHAR(50),
    "Sub-Category" VARCHAR(50),
    "Product Name" VARCHAR(255),
    "Sales" NUMERIC(12, 2),
    "Quantity" INTEGER,
    "Discount" NUMERIC(5, 2),
    "Profit" NUMERIC(12, 2)
);

-- Check table structure
SELECT *
FROM superstore
LIMIT 5;

-- Check number of rows
SELECT COUNT(*) AS total_rows
FROM superstore;

