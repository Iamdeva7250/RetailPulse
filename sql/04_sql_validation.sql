
/* ============================================================
   PROJECT: RetailPulse
   FILE: 04_sql_validation.sql
   STEP 5.28: SQL Analytics Validation
   ============================================================ */


-- 1. Verify the main sales table exists
SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name = 'feature_engineered_sales';


-- 2. Validate total rows and data quality
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (
        WHERE "IsCancelled" = TRUE
    ) AS cancelled_lines,
    COUNT(*) FILTER (
        WHERE "IsCancelled" = FALSE
    ) AS non_cancelled_lines,
    COUNT(*) FILTER (
        WHERE "Revenue" IS NULL
    ) AS missing_revenue,
    COUNT(*) FILTER (
        WHERE "Customer ID" IS NULL
    ) AS missing_customer_id
FROM public.feature_engineered_sales;


-- 3. Verify the analytics views
SELECT
    table_name AS view_name
FROM information_schema.views
WHERE table_schema = 'public'
  AND table_name LIKE 'vw_retailpulse_%'
ORDER BY table_name;


-- 4. Verify overall sales KPIs
SELECT *
FROM public.vw_retailpulse_sales_kpis;


-- 5. Verify monthly sales totals
SELECT *
FROM public.vw_retailpulse_monthly_sales
ORDER BY "YearMonth";


-- 6. Verify the 10 highest-revenue products
SELECT *
FROM public.vw_retailpulse_product_sales
ORDER BY total_revenue DESC
LIMIT 10;


-- 7. Verify the 10 highest-revenue customers
SELECT *
FROM public.vw_retailpulse_customer_sales
ORDER BY total_revenue DESC
LIMIT 10;


-- 8. Verify the 10 highest-revenue countries
SELECT *
FROM public.vw_retailpulse_country_sales
ORDER BY total_revenue DESC
LIMIT 10;


-- 9. Verify the 10 highest-revenue days
SELECT *
FROM public.vw_retailpulse_daily_sales
ORDER BY total_revenue DESC
LIMIT 10;


/* ===================== END OF FILE ======================== */
