
-- RetailPulse: SQL Analytics
-- Source: public.feature_engineered_sales

-- 1. Overall sales KPIs
SELECT
    COUNT(*) AS transaction_lines,
    COUNT(DISTINCT "Invoice") AS distinct_invoices,
    COUNT(DISTINCT "Customer ID") AS distinct_customers,
    COUNT(DISTINCT "StockCode") AS distinct_products,
    SUM("Quantity") AS total_quantity,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue,
    ROUND(AVG("Price")::numeric, 2) AS average_unit_price
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE;

-- 2. Monthly revenue
SELECT
    "YearMonth",
    COUNT(DISTINCT "Invoice") AS total_orders,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "YearMonth"
ORDER BY "YearMonth";

-- 3. Top 10 products by revenue
SELECT
    "StockCode",
    MAX("Description") AS product_description,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "StockCode"
ORDER BY SUM("Revenue") DESC
LIMIT 10;

-- 4. Revenue by country
SELECT
    "Country",
    COUNT(DISTINCT "Invoice") AS total_orders,
    COUNT(DISTINCT "Customer ID") AS unique_customers,
    ROUND(SUM("Revenue")::numeric, 2) AS revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "Country"
ORDER BY SUM("Revenue") DESC;

-- 5. Cancellation overview
SELECT
    "IsCancelled",
    COUNT(*) AS transaction_lines,
    COUNT(DISTINCT "Invoice") AS distinct_invoices,
    SUM("Quantity") AS total_quantity,
    ROUND(SUM("Revenue")::numeric, 2) AS revenue
FROM public.feature_engineered_sales
GROUP BY "IsCancelled"
ORDER BY "IsCancelled";
