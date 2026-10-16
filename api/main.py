
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

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
# 2. DATASET HELPERS
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
    """Load a CSV using its filename stem, without a file extension."""
    files = available_csv_files()

    if dataset_name not in files:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset '{dataset_name}' was not found. "
                "Use GET /datasets to see available datasets."
            ),
        )

    try:
        return pd.read_csv(files[dataset_name], low_memory=False)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read dataset '{dataset_name}': {exc}",
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

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into clean JSON records."""
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
    """Find a column using case-insensitive alphanumeric matching."""
    def normalize(value: str) -> str:
        return "".join(
            character.lower()
            for character in str(value)
            if character.isalnum()
        )

    columns = {normalize(column): column for column in df.columns}

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

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return None

    return float(values.sum())


def numeric_mean(
    df: pd.DataFrame,
    candidates: list[str],
) -> float | None:
    column = find_column(df, candidates)

    if column is None:
        return None

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return None

    return float(values.mean())


# ============================================================
# 3. ROOT ENDPOINT
# ============================================================

@app.get("/", tags=["General"])
def root():
    return {
        "project": "RetailPulse",
        "message": "RetailPulse API is running.",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "available_datasets": "/datasets",
    }


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.get("/health", tags=["General"])
def health_check():
    files = available_csv_files()

    return {
        "status": "healthy" if DATA_DIR.exists() else "warning",
        "project_directory_exists": PROJECT_DIR.exists(),
        "processed_data_directory_exists": DATA_DIR.exists(),
        "available_csv_count": len(files),
        "data_directory": str(DATA_DIR),
    }


# ============================================================
# 5. LIST ALL AVAILABLE DATASETS
# ============================================================

@app.get("/datasets", tags=["Datasets"])
def list_datasets():
    files = available_csv_files()

    datasets = []

    for name, path in sorted(files.items()):
        datasets.append({
            "name": name,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "preview_endpoint": f"/datasets/{name}",
        })

    return {
        "count": len(datasets),
        "datasets": datasets,
    }


# ============================================================
# 6. PREVIEW A DATASET
# ============================================================

@app.get("/datasets/{dataset_name}", tags=["Datasets"])
def dataset_preview(
    dataset_name: str,
    limit: int = Query(default=20, ge=1, le=500),
):
    df = load_dataset(dataset_name)

    return {
        "dataset": dataset_name,
        "rows_in_file": len(df),
        "column_count": len(df.columns),
        "columns": [str(column) for column in df.columns],
        "preview_limit": limit,
        "data": dataframe_records(df.head(limit)),
    }


# ============================================================
# 7. SALES ANALYTICS SUMMARY
# ============================================================

@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary():
    # Prefer the feature-engineered dataset; fall back to sales data.
    selected_name = None
    df = None

    for candidate in [
        "feature_engineered_sales",
        "retail_sales",
        "retail_clean",
    ]:
        if candidate in available_csv_files():
            df = load_dataset(candidate)
            selected_name = candidate
            break

    if df is None:
        raise HTTPException(
            status_code=404,
            detail="No recognized sales dataset was found.",
        )

    customer_column = find_column(
        df, ["CustomerID", "Customer ID", "Customer"]
    )
    invoice_column = find_column(
        df, ["InvoiceNo", "Invoice", "InvoiceID", "OrderID"]
    )
    product_column = find_column(
        df, ["StockCode", "ProductID", "Product", "SKU"]
    )

    return {
        "dataset_used": selected_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "total_revenue": numeric_sum(
            df, ["Revenue", "TotalRevenue", "Sales", "Amount"]
        ),
        "total_quantity": numeric_sum(
            df, ["Quantity", "TotalQuantity"]
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
# 8. REVENUE FORECAST
# ============================================================

@app.get("/forecast/revenue", tags=["Forecasting"])
def revenue_forecast(
    limit: int = Query(default=90, ge=1, le=500),
):
    files = available_csv_files()

    # Prefer the 90-day LSTM forecast if available.
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
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "forecast": dataframe_records(df.head(limit)),
    }


# ============================================================
# 9. INVENTORY PRIORITY PRODUCTS
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
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "products": dataframe_records(df.head(limit)),
    }


# ============================================================
# 10. CHURN PREDICTIONS
# ============================================================

@app.get("/analytics/churn", tags=["Analytics"])
def churn_predictions(
    limit: int = Query(default=50, ge=1, le=500),
):
    dataset_name = "customer_churn_predictions"

    if dataset_name not in available_csv_files():
        raise HTTPException(
            status_code=404,
            detail="Customer churn predictions were not found.",
        )

    df = load_dataset(dataset_name)

    return {
        "dataset_used": dataset_name,
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "predictions": dataframe_records(df.head(limit)),
    }


# ============================================================
# 11. RFM CUSTOMER SEGMENTS
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
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "segments": dataframe_records(df.head(limit)),
    }


# ============================================================
# 12. PRODUCT ABC ANALYSIS
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
        "rows_available": len(df),
        "columns": [str(column) for column in df.columns],
        "products": dataframe_records(df.head(limit)),
    }
