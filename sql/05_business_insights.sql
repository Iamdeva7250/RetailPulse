
/* ============================================================
   PROJECT: RetailPulse
   FILE: 05_business_insights.sql
   STEP 5.29: Business Insights
   TABLE: public.feature_engineered_sales
   ============================================================ */


-- QUERY 1: TOP 10 PRODUCTS BY UNITS SOLD
SELECT
    "StockCode",
    MAX("Description") AS product_description,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "StockCode"
ORDER BY SUM("Quantity") DESC
LIMIT 10;


-- QUERY 2: LOW-REVENUE PRODUCTS
-- Products with the lowest positive revenue.
SELECT
    "StockCode",
    MAX("Description") AS product_description,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "StockCode"
HAVING SUM("Revenue") > 0
ORDER BY SUM("Revenue") ASC
LIMIT 10;


-- QUERY 3: AVERAGE ORDER VALUE BY MONTH
WITH invoice_totals AS (
    SELECT
        "YearMonth",
        "Invoice",
        SUM("Revenue") AS order_revenue
    FROM public.feature_engineered_sales
    WHERE "IsCancelled" = FALSE
    GROUP BY "YearMonth", "Invoice"
)
SELECT
    "YearMonth",
    COUNT(*) AS total_orders,
    ROUND(AVG(order_revenue)::numeric, 2)
        AS average_order_value,
    ROUND(SUM(order_revenue)::numeric, 2)
        AS total_revenue
FROM invoice_totals
GROUP BY "YearMonth"
ORDER BY "YearMonth";


-- QUERY 4: CUSTOMER REVENUE SEGMENTS
WITH customer_totals AS (
    SELECT
        "Customer ID",
        SUM("Revenue") AS customer_revenue
    FROM public.feature_engineered_sales
    WHERE "IsCancelled" = FALSE
      AND "Customer ID" IS NOT NULL
    GROUP BY "Customer ID"
)
SELECT
    CASE
        WHEN customer_revenue >= 10000 THEN 'High Value'
        WHEN customer_revenue >= 1000 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS customer_segment,
    COUNT(*) AS customer_count,
    ROUND(SUM(customer_revenue)::numeric, 2)
        AS segment_revenue
FROM customer_totals
GROUP BY
    CASE
        WHEN customer_revenue >= 10000 THEN 'High Value'
        WHEN customer_revenue >= 1000 THEN 'Medium Value'
        ELSE 'Low Value'
    END
ORDER BY segment_revenue DESC;


-- QUERY 5: REVENUE BY YEAR AND QUARTER
SELECT
    "Year",
    "Quarter",
    COUNT(DISTINCT "Invoice") AS total_orders,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "Year", "Quarter"
ORDER BY "Year", "Quarter";


-- QUERY 6: SALES BY HOUR OF DAY
SELECT
    "Hour",
    COUNT(DISTINCT "Invoice") AS total_orders,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "Hour"
ORDER BY "Hour";


-- QUERY 7: PRODUCTS PURCHASED BY THE MOST CUSTOMERS
SELECT
    "StockCode",
    MAX("Description") AS product_description,
    COUNT(DISTINCT "Customer ID") AS unique_customers,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
  AND "Customer ID" IS NOT NULL
GROUP BY "StockCode"
ORDER BY COUNT(DISTINCT "Customer ID") DESC
LIMIT 10;


/* ===================== END OF FILE ======================== */



-- RetailPulse: Analytics View Validation

-- 1. Overall sales KPIs
SELECT *
FROM public.vw_retailpulse_sales_kpis;

-- 2. Monthly sales
SELECT *
FROM public.vw_retailpulse_monthly_sales
ORDER BY "YearMonth";

-- 3. Top 10 products by revenue
SELECT *
FROM public.vw_retailpulse_product_sales
ORDER BY total_revenue DESC
LIMIT 10;

-- 4. Top 10 customers by revenue
SELECT *
FROM public.vw_retailpulse_customer_sales
ORDER BY total_revenue DESC
LIMIT 10;

-- 5. Top 10 countries by revenue
SELECT *
FROM public.vw_retailpulse_country_sales
ORDER BY total_revenue DESC
LIMIT 10;

-- 6. Recent daily sales records
SELECT *
FROM public.vw_retailpulse_daily_sales
ORDER BY sales_date DESC
LIMIT 10;
