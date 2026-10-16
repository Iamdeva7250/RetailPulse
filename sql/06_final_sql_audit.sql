
/* ============================================================
   RetailPulse - Final SQL Data Quality Audit
   Database: retailpulse
   Main table: public.feature_engineered_sales
   ============================================================ */

-- QUERY 1: Total records and cancellation status
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (
        WHERE "IsCancelled" = TRUE
    ) AS cancelled_rows,
    COUNT(*) FILTER (
        WHERE "IsCancelled" = FALSE
    ) AS active_rows,
    COUNT(*) FILTER (
        WHERE "IsCancelled" IS NULL
    ) AS unknown_cancellation_status
FROM public.feature_engineered_sales;


-- QUERY 2: Missing values in important columns
SELECT
    COUNT(*) FILTER (
        WHERE "Invoice" IS NULL
    ) AS missing_invoice,
    COUNT(*) FILTER (
        WHERE "StockCode" IS NULL
    ) AS missing_stock_code,
    COUNT(*) FILTER (
        WHERE "InvoiceDate" IS NULL
    ) AS missing_invoice_date,
    COUNT(*) FILTER (
        WHERE "Quantity" IS NULL
    ) AS missing_quantity,
    COUNT(*) FILTER (
        WHERE "Price" IS NULL
    ) AS missing_price,
    COUNT(*) FILTER (
        WHERE "Revenue" IS NULL
    ) AS missing_revenue,
    COUNT(*) FILTER (
        WHERE "Customer ID" IS NULL
    ) AS missing_customer_id
FROM public.feature_engineered_sales;


-- QUERY 3: Check duplicate invoice-product-date lines
SELECT
    "Invoice",
    "StockCode",
    "InvoiceDate",
    COUNT(*) AS duplicate_count
FROM public.feature_engineered_sales
GROUP BY
    "Invoice",
    "StockCode",
    "InvoiceDate"
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 20;


-- QUERY 4: Compare stored revenue with quantity multiplied by price
SELECT
    COUNT(*) AS checked_rows,
    COUNT(*) FILTER (
        WHERE "Revenue" IS DISTINCT FROM
              ("Quantity" * "Price")
    ) AS revenue_mismatch_rows
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
  AND "Quantity" IS NOT NULL
  AND "Price" IS NOT NULL
  AND "Revenue" IS NOT NULL;


-- QUERY 5: Check the six analytics views
SELECT
    table_name AS view_name
FROM information_schema.views
WHERE table_schema = 'public'
  AND table_name IN (
      'vw_retailpulse_sales_kpis',
      'vw_retailpulse_monthly_sales',
      'vw_retailpulse_product_sales',
      'vw_retailpulse_customer_sales',
      'vw_retailpulse_country_sales',
      'vw_retailpulse_daily_sales'
  )
ORDER BY table_name;


/* ================= END OF AUDIT ================= */
