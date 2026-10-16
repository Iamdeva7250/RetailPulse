

/* ============================================================
   RetailPulse - 02_advanced_analytics.sql
   Table: public.feature_engineered_sales
   Queries 6 to 11: Advanced SQL Analytics
   ============================================================ */

-- QUERY 6: MONTHLY REVENUE GROWTH
WITH monthly_sales AS (
    SELECT "YearMonth", SUM("Revenue") AS revenue
    FROM public.feature_engineered_sales
    WHERE "IsCancelled" = FALSE
    GROUP BY "YearMonth"
),
sales_with_previous AS (
    SELECT
        "YearMonth",
        revenue,
        LAG(revenue) OVER (ORDER BY "YearMonth") AS previous_month_revenue
    FROM monthly_sales
)
SELECT
    "YearMonth",
    ROUND(revenue::numeric, 2) AS revenue,
    ROUND(previous_month_revenue::numeric, 2) AS previous_month_revenue,
    ROUND(
        (100.0 * (revenue - previous_month_revenue)
        / NULLIF(previous_month_revenue, 0))::numeric, 2
    ) AS growth_percentage
FROM sales_with_previous
ORDER BY "YearMonth";


-- QUERY 7: TOP 10 CUSTOMERS BY REVENUE
SELECT
    "Customer ID",
    COUNT(DISTINCT "Invoice") AS total_orders,
    SUM("Quantity") AS total_quantity,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
  AND "Customer ID" IS NOT NULL
GROUP BY "Customer ID"
ORDER BY SUM("Revenue") DESC
LIMIT 10;


-- QUERY 8: ONE-TIME VS REPEAT CUSTOMERS
WITH customer_orders AS (
    SELECT
        "Customer ID",
        COUNT(DISTINCT "Invoice") AS order_count
    FROM public.feature_engineered_sales
    WHERE "IsCancelled" = FALSE
      AND "Customer ID" IS NOT NULL
    GROUP BY "Customer ID"
)
SELECT
    CASE
        WHEN order_count = 1 THEN 'One-time customer'
        ELSE 'Repeat customer'
    END AS customer_type,
    COUNT(*) AS customers,
    ROUND(AVG(order_count)::numeric, 2) AS average_orders
FROM customer_orders
GROUP BY
    CASE
        WHEN order_count = 1 THEN 'One-time customer'
        ELSE 'Repeat customer'
    END
ORDER BY customers DESC;


-- QUERY 9: SALES BY DAY OF THE WEEK
SELECT
    "DayName",
    "DayOfWeek",
    COUNT(DISTINCT "Invoice") AS total_orders,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
GROUP BY "DayName", "DayOfWeek"
ORDER BY "DayOfWeek";


-- QUERY 10: TOP 20 PRODUCTS BY REVENUE CONTRIBUTION
WITH product_sales AS (
    SELECT
        "StockCode",
        MAX("Description") AS product_description,
        SUM("Quantity") AS units_sold,
        SUM("Revenue") AS revenue
    FROM public.feature_engineered_sales
    WHERE "IsCancelled" = FALSE
    GROUP BY "StockCode"
)
SELECT
    "StockCode",
    product_description,
    units_sold,
    ROUND(revenue::numeric, 2) AS revenue,
    ROUND(
        (100.0 * revenue / NULLIF(SUM(revenue) OVER (), 0))::numeric,
        2
    ) AS revenue_share_percentage
FROM product_sales
ORDER BY revenue DESC
LIMIT 20;


-- QUERY 11: TOP 10 COUNTRIES BY REVENUE
SELECT
    "Country",
    COUNT(DISTINCT "Invoice") AS total_orders,
    COUNT(DISTINCT "Customer ID") AS total_customers,
    SUM("Quantity") AS units_sold,
    ROUND(SUM("Revenue")::numeric, 2) AS total_revenue,
    ROUND(AVG("Revenue")::numeric, 2) AS average_line_revenue
FROM public.feature_engineered_sales
WHERE "IsCancelled" = FALSE
  AND "Country" IS NOT NULL
GROUP BY "Country"
ORDER BY SUM("Revenue") DESC
LIMIT 10;

/* ================= END OF QUERIES 6–11 ================= */

-- Step 5.18: Validate RetailPulse database
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
    ) AS missing_customer_id,
    COUNT(DISTINCT "Invoice") AS distinct_invoices,
    COUNT(DISTINCT "StockCode") AS distinct_products
FROM public.feature_engineered_sales;