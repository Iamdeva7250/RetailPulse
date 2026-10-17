
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

DB_TABLE = "feature_engineered_sales"
DB_SCHEMA = "public"

_engine: Engine | None = None

app = FastAPI(
    title="RetailPulse API",
    description=(
        "REST API for RetailPulse sales analytics, customer "
        "analytics, churn predictions, revenue forecasting, "
        "and inventory optimization."
    ),
    version="1.0.0",
)


# ============================================================
# 2. DATABASE HELPERS
# ============================================================

def get_database_engine() -> Engine | None:
    """Create a PostgreSQL engine when DATABASE_URL is configured."""
    global _engine

    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        return None

    if _engine is None:
        # Support Render URLs and normalize them for psycopg 3.
        if database_url.startswith("postgres://"):
            database_url = database_url.replace(
                "postgres://",
                "postgresql+psycopg://",
                1,
            )
        elif database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://",
                "postgresql+psycopg://",
                1,
            )

        _engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=3,
            pool_recycle=300,
        )

    return _engine


def database_table_exists() -> bool:
    """Return True if the configured sales table exists."""
    engine = get_database_engine()

    if engine is None:
        return False

    return inspect(engine).has_table(
        DB_TABLE,
        schema=DB_SCHEMA,
    )


def database_columns() -> list[str]:
    """Return the PostgreSQL sales table's column names."""
    engine = get_database_engine()

    if engine is None:
        return []

    inspector = inspect(engine)

    if not inspector.has_table(DB_TABLE, schema=DB_SCHEMA):
        return []

    return [
        column["name"]
        for column in inspector.get_columns(
            DB_TABLE,
            schema=DB_SCHEMA,
        )
    ]


def quote_column(column_name: str) -> str:
    """Safely quote a database column identifier."""
    return '"' + column_name.replace('"', '""') + '"'


def find_database_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    """Find a database column using case-insensitive matching."""
    def normalize(value: str) -> str:
        return "".join(
            character.lower()
            for character in str(value)
            if character.isalnum()
        )

    normalized_columns = {
        normalize(column): column
        for column in columns
    }

    for candidate in candidates:
        match = normalized_columns.get(normalize(candidate))
        if match is not None:
            return match

    return None


def database_row_count() -> int:
    """Count sales rows without downloading the dataset."""
    engine = get_database_engine()

    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured.")

    query = text(
        'SELECT COUNT(*) FROM "public"."feature_engineered_sales"'
    )

    with engine.connect() as connection:
        return int(connection.execute(query).scalar_one())


def database_preview(limit: int) -> dict[str, Any]:
    """Fetch only a limited number of sales records."""
    engine = get_database_engine()

    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured.")

    count_query = text(
        'SELECT COUNT(*) FROM "public"."feature_engineered_sales"'
    )
    preview_query = text(
        'SELECT * FROM "public"."feature_engineered_sales" LIMIT :limit'
    )

    with engine.connect() as connection:
        total_rows = int(
            connection.execute(count_query).scalar_one()
        )

        result = connection.execute(
            preview_query,
            {"limit": limit},
        )

        columns = list(result.keys())
        rows = result.mappings().all()

    records = [
        {
            str(column): clean_value(value)
            for column, value in row.items()
        }
        for row in rows
    ]

    return {
        "dataset": DB_TABLE,
        "source": "postgresql",
        "rows_in_file": total_rows,
	"rows_in_dataset": total_rows,
        "column_count": len(columns),
        "columns": columns,
        "preview_limit": limit,
        "data": records,
    }


def database_analytics_summary() -> dict[str, Any]:
    """Calculate the sales KPIs inside PostgreSQL."""
    engine = get_database_engine()

    if engine is None:
        raise RuntimeError("DATABASE_URL is not configured.")

    columns = database_columns()

    if not columns:
        raise RuntimeError(
            f"Table {DB_SCHEMA}.{DB_TABLE} was not found."
        )

    revenue_column = find_database_column(
        columns,
        ["Revenue", "TotalRevenue", "Sales", "Amount"],
    )
    quantity_column = find_database_column(
        columns,
        ["Quantity", "TotalQuantity"],
    )
    customer_column = find_database_column(
        columns,
        ["Customer_ID", "CustomerID", "Customer ID", "Customer"],
    )
    invoice_column = find_database_column(
        columns,
        ["Invoice", "InvoiceNo", "InvoiceID", "OrderID"],
    )
    product_column = find_database_column(
        columns,
        ["StockCode", "ProductID", "Product", "SKU"],
    )

    expressions = [
        "COUNT(*) AS row_count",
    ]

    if revenue_column:
        expressions.append(
            f"SUM({quote_column(revenue_column)}) AS total_revenue"
        )

    if quantity_column:
        expressions.append(
            f"SUM({quote_column(quantity_column)}) AS total_quantity"
        )

    if customer_column:
        expressions.append(
            f"COUNT(DISTINCT {quote_column(customer_column)}) "
            "AS unique_customers"
        )

    if invoice_column:
        expressions.append(
            f"COUNT(DISTINCT {quote_column(invoice_column)}) "
            "AS unique_invoices"
        )

    if product_column:
        expressions.append(
            f"COUNT(DISTINCT {quote_column(product_column)}) "
            "AS unique_products"
        )

    query = text(
        'SELECT '
        + ", ".join(expressions)
        + ' FROM "public"."feature_engineered_sales"'
    )

    with engine.connect() as connection:
        result = connection.execute(query).mappings().one()

    return {
        "dataset_used": DB_TABLE,
        "source": "postgresql",
        "row_count": int(result["row_count"]),
        "column_count": len(columns),
        "total_revenue": (
            float(result["total_revenue"])
            if "total_revenue" in result
            and result["total_revenue"] is not None
            else None
        ),
        "total_quantity": (
            float(result["total_quantity"])
            if "total_quantity" in result
            and result["total_quantity"] is not None
            else None
        ),
        "unique_customers": (
            int(result["unique_customers"])
            if "unique_customers" in result
            else None
        ),
        "unique_invoices": (
            int(result["unique_invoices"])
            if "unique_invoices" in result
            else None
        ),
        "unique_products": (
            int(result["unique_products"])
            if "unique_products" in result
            else None
        ),
    }


# ============================================================
# 3. LOCAL CSV HELPERS
# ============================================================

def available_csv_files() -> dict[str, Path]:
    """Return CSV files available in the processed data folder."""
    if not DATA_DIR.exists():
        return {}

    return {
        path.stem: path
        for path in DATA_DIR.glob("*.csv")
        if path.is_file()
    }


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a local CSV by its filename stem."""
    files = available_csv_files()

    if dataset_name not in files:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset '{dataset_name}' was not found locally. "
                "Use GET /datasets to see available datasets."
            ),
        )

    try:
        return pd.read_csv(
            files[dataset_name],
            low_memory=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read dataset '{dataset_name}'.",
        ) from exc


def clean_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-compatible values."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def dataframe_records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-compatible records."""
    return [
        {
            str(column): clean_value(value)
            for column, value in record.items()
        }
        for record in df.to_dict(orient="records")
    ]


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find a DataFrame column using case-insensitive matching."""
    def normalize(value: str) -> str:
        return "".join(
            character.lower()
            for character in str(value)
            if character.isalnum()
        )

    columns = {
        normalize(column): column
        for column in df.columns
    }

    for candidate in candidates:
        match = columns.get(normalize(candidate))
        if match is not None:
            return match

    return None


def numeric_sum(
    df: pd.DataFrame,
    candidates: list[str],
) -> float | None:
    column = find_column(df, candidates)

    if column is None:
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.sum())


# ============================================================
# 4. ROOT ENDPOINT
# ============================================================

@app.get("/", tags=["General"])
def root():
    return {
        "project": "RetailPulse",
        "message": "RetailPulse API is running.",
        "version": app.version,
        "documentation": "/docs",
        "health_check": "/health",
        "available_datasets": "/datasets",
    }


# ============================================================
# 5. HEALTH CHECK
# ============================================================

@app.get("/health", tags=["General"])
def health_check():
    files = available_csv_files()
    db_configured = bool(
        os.getenv("DATABASE_URL", "").strip()
    )

    db_status = "not_configured"

    if db_configured:
        try:
            engine = get_database_engine()

            if engine is not None:
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))

                db_status = (
                    "connected"
                    if database_table_exists()
                    else "connected_table_missing"
                )
        except Exception:
            db_status = "unavailable"

    database_healthy = db_status in {
        "connected",
        "not_configured",
    }

    return {
        "status": (
            "healthy"
            if database_healthy
            else "degraded"
        ),
        "project_directory_exists": PROJECT_DIR.exists(),
        "processed_data_directory_exists": DATA_DIR.exists(),
        "available_csv_count": len(files),
        "database_configured": db_configured,
        "database_status": db_status,
        "database_table": f"{DB_SCHEMA}.{DB_TABLE}",
    }


# ============================================================
# 6. LIST AVAILABLE DATASETS
# ============================================================

@app.get("/datasets", tags=["Datasets"])
def list_datasets():
    files = available_csv_files()
    datasets = []

    for name, path in sorted(files.items()):
        datasets.append({
            "name": name,
            "source": "csv",
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "preview_endpoint": f"/datasets/{name}",
        })

    # Include the hosted database dataset if it is available.
    try:
        if database_table_exists():
            columns = database_columns()
            datasets = [
                item for item in datasets
                if item["name"] != DB_TABLE
            ]
            datasets.append({
                "name": DB_TABLE,
                "source": "postgresql",
                "rows": database_row_count(),
                "column_count": len(columns),
                "columns": columns,
                "preview_endpoint": f"/datasets/{DB_TABLE}",
            })
    except SQLAlchemyError:
        # Keep CSV datasets available if PostgreSQL is temporarily down.
        pass

    return {
        "count": len(datasets),
        "datasets": datasets,
    }


# ============================================================
# 7. PREVIEW A DATASET
# ============================================================

@app.get("/datasets/{dataset_name}", tags=["Datasets"])
def dataset_preview(
    dataset_name: str,
    limit: int = Query(default=20, ge=1, le=500),
):
    # Use PostgreSQL for the migrated sales dataset.
    if dataset_name == DB_TABLE:
        try:
            if database_table_exists():
                return database_preview(limit)
        except SQLAlchemyError as exc:
            raise HTTPException(
                status_code=503,
                detail="The PostgreSQL dataset is temporarily unavailable.",
            ) from exc

    # Other datasets continue to use local CSV files.
    df = load_dataset(dataset_name)

    return {
        "dataset": dataset_name,
        "source": "csv",
        "rows_in_file": len(df),
	"rows_in_dataset": len(df),
        "column_count": len(df.columns),
        "columns": [str(column) for column in df.columns],
        "preview_limit": limit,
        "data": dataframe_records(df.head(limit)),
    }


# ============================================================
# 8. SALES ANALYTICS SUMMARY
# ============================================================

@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary():
    # Prefer PostgreSQL for the main sales dataset.
    try:
        if database_table_exists():
            return database_analytics_summary()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL analytics are temporarily unavailable.",
        ) from exc

    # Fallback to a local CSV during development.
    files = available_csv_files()
    selected_name = None
    df = None

    for candidate in [
        "feature_engineered_sales",
        "retail_sales",
        "retail_clean",
    ]:
        if candidate in files:
            df = load_dataset(candidate)
            selected_name = candidate
            break

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No recognized sales dataset was found in PostgreSQL "
                "or the local processed-data folder."
            ),
        )

    customer_column = find_column(
        df,
        ["CustomerID", "Customer_ID", "Customer ID", "Customer"],
    )
    invoice_column = find_column(
        df,
        ["InvoiceNo", "Invoice", "InvoiceID", "OrderID"],
    )
    product_column = find_column(
        df,
        ["StockCode", "ProductID", "Product", "SKU"],
    )

    return {
        "dataset_used": selected_name,
        "source": "csv",
        "row_count": len(df),
        "column_count": len(df.columns),
        "total_revenue": numeric_sum(
            df,
            ["Revenue", "TotalRevenue", "Sales", "Amount"],
        ),
        "total_quantity": numeric_sum(
            df,
            ["Quantity", "TotalQuantity"],
        ),
        "unique_customers": (
            int(df[customer_column].nunique())
            if customer_column else None
        ),
        "unique_invoices": (
            int(df[invoice_column].nunique())
            if invoice_column else None
        ),
        "unique_products": (
            int(df[product_column].nunique())
            if product_column else None
        ),
    }


# ============================================================
# 9. REVENUE FORECAST
# ============================================================


@app.get("/analytics/customers", tags=["Analytics"])
def customer_analytics(
    limit: int = Query(default=500, ge=1, le=500),
):
    """Aggregate full-dataset customer metrics inside PostgreSQL."""
    engine = get_database_engine()

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not configured.",
        )

    try:
        columns = database_columns()

        customer_col = find_database_column(
            columns, ["Customer ID", "CustomerID", "Customer_ID"]
        )
        invoice_col = find_database_column(
            columns, ["Invoice", "InvoiceNo", "InvoiceID", "OrderID"]
        )
        date_col = find_database_column(
            columns, ["InvoiceDate", "Invoice Date", "OrderDate", "Date"]
        )
        revenue_col = find_database_column(
            columns, ["Revenue", "TotalRevenue", "Sales", "Amount"]
        )

        missing = [
            name for name, value in [
                ("customer ID", customer_col),
                ("invoice ID", invoice_col),
                ("invoice date", date_col),
                ("revenue", revenue_col),
            ] if value is None
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Customer analytics requires these columns: "
                    + ", ".join(missing)
                ),
            )

        table = (
            f"{quote_column(DB_SCHEMA)}."
            f"{quote_column(DB_TABLE)}"
        )
        customer = quote_column(customer_col)
        invoice = quote_column(invoice_col)
        invoice_date = quote_column(date_col)
        revenue = quote_column(revenue_col)

        # Both queries aggregate in PostgreSQL. Only the requested
        # customer preview is transferred to the dashboard.
        grouped_cte = f"""
            SELECT
                {customer} AS customer_id,
                MAX({invoice_date}) AS last_purchase_date,
                COUNT(DISTINCT {invoice}) AS frequency,
                SUM(COALESCE({revenue}, 0)) AS monetary,
                COUNT(*) AS transaction_rows
            FROM {table}
            WHERE {customer} IS NOT NULL
            GROUP BY {customer}
        """

        summary_query = text(f"""
            WITH customer_totals AS ({grouped_cte})
            SELECT
                COUNT(*) AS total_customers,
                COALESCE(SUM(monetary), 0) AS total_customer_revenue,
                COALESCE(AVG(monetary), 0) AS average_customer_revenue
            FROM customer_totals
        """)

        preview_query = text(f"""
            WITH bounds AS (
                SELECT MAX({invoice_date}) AS max_date
                FROM {table}
            ),
            customer_totals AS ({grouped_cte})
            SELECT
                c.customer_id AS "CustomerID",
                c.last_purchase_date AS "LastPurchaseDate",
                (CAST(b.max_date AS DATE)
                 - CAST(c.last_purchase_date AS DATE)) AS "Recency",
                c.frequency AS "Frequency",
                c.monetary AS "Monetary",
                c.transaction_rows AS "TransactionRows"
            FROM customer_totals c
            CROSS JOIN bounds b
            ORDER BY c.monetary DESC
            LIMIT :limit
        """)

        with engine.connect() as connection:
            summary_row = connection.execute(
                summary_query
            ).mappings().one()

            result = connection.execute(
                preview_query,
                {"limit": limit},
            )
            records = [
                {
                    str(key): clean_value(value)
                    for key, value in row.items()
                }
                for row in result.mappings().all()
            ]

        return {
            "source": "postgresql",
            "dataset": DB_TABLE,
            "total_customers": int(summary_row["total_customers"]),
            "total_customer_revenue": clean_value(
                summary_row["total_customer_revenue"]
            ),
            "average_customer_revenue": clean_value(
                summary_row["average_customer_revenue"]
            ),
            "preview_limit": limit,
            "returned_customers": len(records),
            "data": records,
        }

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL customer analytics are temporarily unavailable.",
        ) from exc


@app.get("/forecast/revenue", tags=["Forecasting"])
def revenue_forecast(
    limit: int = Query(default=90, ge=1, le=500),
):
    files = available_csv_files()

    candidates = [
        "lstm_revenue_forecast_90_days",
        "revenue_forecast_90_days",
        "monthly_revenue_forecast",
    ]

    selected_name = next(
        (name for name in candidates if name in files),
        None,
    )

    if selected_name is None:
        raise HTTPException(
            status_code=404,
            detail="No saved revenue forecast dataset was found.",
        )

    df = load_dataset(selected_name)

    return {
        "dataset_used": selected_name,
        "source": "csv",
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "forecast": dataframe_records(df.head(limit)),
    }


# ============================================================
# 10. INVENTORY PRIORITY PRODUCTS
# ============================================================

@app.get("/inventory/priority", tags=["Inventory"])
def inventory_priority(
    limit: int = Query(default=50, ge=1, le=500),
):
    files = available_csv_files()

    candidates = [
        "inventory_priority_products",
        "inventory_optimization",
    ]

    selected_name = next(
        (name for name in candidates if name in files),
        None,
    )

    if selected_name is None:
        raise HTTPException(
            status_code=404,
            detail="No inventory priority dataset was found.",
        )

    df = load_dataset(selected_name)

    return {
        "dataset_used": selected_name,
        "source": "csv",
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "products": dataframe_records(df.head(limit)),
    }


# ============================================================
# 11. CHURN PREDICTIONS
# ============================================================

@app.get("/analytics/churn", tags=["Analytics"])
def churn_predictions(
    limit: int = Query(default=50, ge=1, le=500),
):
    dataset_name = "customer_churn_predictions"
    engine = get_database_engine()

    # Prefer saved predictions in PostgreSQL when available.
    if (
        engine is not None
        and inspect(engine).has_table(dataset_name, schema="public")
    ):
        summary_query = text("""
            SELECT
                COUNT(*) AS total_predictions,
                COUNT(DISTINCT "Customer ID") AS unique_customers,
                COALESCE(
                    SUM(
                        CASE
                            WHEN LOWER(CAST("PredictedChurn" AS TEXT))
                                 IN ('1', 'true', 'yes', 'churn', 'churned')
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS predicted_churners,
                AVG("ChurnProbability") AS average_churn_probability
            FROM public.customer_churn_predictions
        """)

        preview_query = text("""
            SELECT *
            FROM public.customer_churn_predictions
            ORDER BY "ChurnProbability" DESC NULLS LAST
            LIMIT :limit
        """)

        with engine.connect() as connection:
            summary_row = connection.execute(
                summary_query
            ).mappings().one()

            predictions_df = pd.read_sql_query(
                preview_query,
                connection,
                params={"limit": limit},
            )

        total_predictions = int(summary_row["total_predictions"] or 0)
        predicted_churners = int(summary_row["predicted_churners"] or 0)

        return {
            "dataset_used": dataset_name,
            "source": "postgresql",
            "rows_available": total_predictions,
            "columns": [str(column) for column in predictions_df.columns],
            "summary": {
                "total_predictions": total_predictions,
                "unique_customers": int(
                    summary_row["unique_customers"] or 0
                ),
                "predicted_churners": predicted_churners,
                "predicted_churn_rate": (
                    predicted_churners / total_predictions * 100
                    if total_predictions else 0.0
                ),
                "average_churn_probability": (
                    float(summary_row["average_churn_probability"])
                    if summary_row["average_churn_probability"] is not None
                    else None
                ),
            },
            "predictions": dataframe_records(predictions_df),
        }

    # Preserve the original local CSV fallback.
    if dataset_name not in available_csv_files():
        raise HTTPException(
            status_code=404,
            detail=(
                "Customer churn predictions were not found "
                "in PostgreSQL or the local CSV datasets."
            ),
        )

    df = load_dataset(dataset_name)

    return {
        "dataset_used": dataset_name,
        "source": "csv",
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "predictions": dataframe_records(df.head(limit)),
    }


# 12. RFM CUSTOMER SEGMENTS
# ============================================================

@app.get("/analytics/rfm", tags=["Analytics"])
def rfm_segments(
    limit: int = Query(default=100, ge=1, le=500),
):
    dataset_name = "rfm_customer_segments"

    if dataset_name not in available_csv_files():
        raise HTTPException(
            status_code=404,
            detail="RFM customer segments were not found.",
        )

    df = load_dataset(dataset_name)

    return {
        "dataset_used": dataset_name,
        "source": "csv",
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "segments": dataframe_records(df.head(limit)),
    }


# ============================================================
# 13. PRODUCT ABC ANALYSIS
# ============================================================

@app.get("/analytics/abc-products", tags=["Analytics"])
def abc_product_analysis(
    limit: int = Query(default=100, ge=1, le=500),
):
    dataset_name = "abc_product_analysis"

    if dataset_name not in available_csv_files():
        raise HTTPException(
            status_code=404,
            detail="ABC product analysis was not found.",
        )

    df = load_dataset(dataset_name)

    return {
        "dataset_used": dataset_name,
        "source": "csv",
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "products": dataframe_records(df.head(limit)),
    }
