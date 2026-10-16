
/* ============================================================
   PROJECT: RetailPulse
   FILE: 03_analytics_views.sql
   STEP 5.19: Reusable SQL Analytics Views
   DATABASE: retailpulse
   SOURCE TABLE: public.feature_engineered_sales

   VIEWS:
   1. Overall Sales KPIs
   2. Monthly Sales Summary
   3. Product Performance

   PURPOSE:
   Create reusable views for sales reporting and future
   dashboard integration.
   ============================================================ */


-- ============================================================
-- VIEW 1: OVERALL SALES KPIs
-- Purpose:
-- Calculate transaction lines, invoices, customers, products,
-- total quantity, total revenue, and average unit price.
-- ============================================================

CREATE OR REPLACE VIEW public.vw_retailpulse_sales_kpis AS
SELECT
    COUNT(*) AS transaction_lines,
    COUNT(DISTINCT "Invoice") AS total_invoices,
    COUNT(DISTINCT "Customer ID") AS total_customers,
    COUNT(DISTINCT "StockCode") AS total_products,
    SUM("Quantity") AS total_quantity,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue,
    ROUND(AVG("Price")::numeric, 2) AS average_unit_price
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE;


-- ============================================================
-- VIEW 2: MONTHLY SALES SUMMARY
-- Purpose:
-- Summarize monthly orders, units sold, and revenue.
-- ============================================================

CREATE OR REPLACE VIEW public.vw_retailpulse_monthly_sales AS
SELECT
    "YearMonth",
    COUNT(DISTINCT "Invoice") AS total_orders,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "YearMonth";


-- ============================================================
-- VIEW 3: PRODUCT PERFORMANCE
-- Purpose:
-- Calculate product-level quantity, orders, and revenue.
-- ============================================================

CREATE OR REPLACE VIEW public.vw_retailpulse_product_sales AS
SELECT
    "StockCode",
    MAX("Description") AS product_description,
    SUM("Quantity") AS units_sold,
    COUNT(DISTINCT "Invoice") AS total_orders,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "StockCode";


-- ============================================================
-- VERIFICATION 1: OVERALL SALES KPIs
-- Expected: One row of summary metrics.
-- ============================================================

SELECT *
FROM public.vw_retailpulse_sales_kpis;


-- ============================================================
-- VERIFICATION 2: MONTHLY SALES
-- Expected: One row per month.
-- ============================================================

SELECT *
FROM public.vw_retailpulse_monthly_sales
ORDER BY "YearMonth";


-- ============================================================
-- VERIFICATION 3: TOP 10 PRODUCTS
-- Expected: Up to 10 products sorted by revenue.
-- ============================================================

SELECT *
FROM public.vw_retailpulse_product_sales
ORDER BY total_revenue DESC
LIMIT 10;


-- ============================================================
-- VERIFICATION 4: LIST ALL RETAILPULSE VIEWS
-- Expected: Three view names.
-- ============================================================

SELECT table_name
FROM information_schema.views
WHERE table_schema = 'public'
  AND table_name LIKE 'vw_retailpulse_%'
ORDER BY table_name;



-- ============================================================
-- VIEW 4: CUSTOMER ANALYTICS
-- Purpose:
-- Analyze customer orders, quantity, revenue, average order
-- value, and first/last purchase dates.
-- ============================================================

CREATE OR REPLACE VIEW public.vw_retailpulse_customer_sales AS
SELECT
    "Customer ID",
    COUNT(DISTINCT "Invoice") AS total_orders,
    SUM("Quantity") AS total_quantity,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue,
    ROUND(
        (
            SUM("Revenue") /
            NULLIF(COUNT(DISTINCT "Invoice"), 0)
        )::numeric,
        2
    ) AS average_order_value,
    MIN("InvoiceDate") AS first_purchase,
    MAX("InvoiceDate") AS last_purchase
FROM public.feature_engineered_sales
WHERE
    "IsCancelled" = FALSE
    AND "Customer ID" IS NOT NULL
GROUP BY "Customer ID";


-- ============================================================
-- VIEW 5: COUNTRY ANALYTICS
-- Purpose:
-- Compare countries by orders, customers, units sold,
-- and total revenue.
-- ============================================================

CREATE OR REPLACE VIEW public.vw_retailpulse_country_sales AS
SELECT
    "Country",
    COUNT(DISTINCT "Invoice") AS total_orders,
    COUNT(DISTINCT "Customer ID") AS total_customers,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE
    "IsCancelled" = FALSE
    AND "Country" IS NOT NULL
GROUP BY "Country";


-- ============================================================
-- VERIFICATION 5: TOP 10 CUSTOMERS BY REVENUE
-- ============================================================

SELECT *
FROM public.vw_retailpulse_customer_sales
ORDER BY total_revenue DESC
LIMIT 10;


-- ============================================================
-- VERIFICATION 6: TOP 10 COUNTRIES BY REVENUE
-- ============================================================

SELECT *
FROM public.vw_retailpulse_country_sales
ORDER BY total_revenue DESC
LIMIT 10;


-- ============================================================
-- VERIFICATION 7: LIST ALL RETAILPULSE ANALYTICS VIEWS
-- Expected: Five view names.
-- ============================================================

SELECT table_name
FROM information_schema.views
WHERE table_schema = 'public'
  AND table_name LIKE 'vw_retailpulse_%'
ORDER BY table_name;



-- ============================================================
-- VIEW 6: DAILY SALES ANALYTICS
-- Purpose:
-- Summarize daily orders, units sold, customers, and revenue.
-- ============================================================

CREATE OR REPLACE VIEW public.vw_retailpulse_daily_sales AS
SELECT
    "Date" AS sales_date,
    COUNT(DISTINCT "Invoice") AS total_orders,
    COUNT(DISTINCT "Customer ID") AS total_customers,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "Date";


-- ============================================================
-- VERIFY VIEW 6: LATEST 30 DAYS IN THE DATASET
-- Uses the latest available dates in the dataset.
-- ============================================================

SELECT *
FROM public.vw_retailpulse_daily_sales
ORDER BY sales_date DESC
LIMIT 30;


-- ============================================================
-- TOP 10 DAYS BY REVENUE
-- ============================================================

SELECT *
FROM public.vw_retailpulse_daily_sales
ORDER BY total_revenue DESC
LIMIT 10;

