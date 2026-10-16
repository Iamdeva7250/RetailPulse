
import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

DATASET = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "feature_engineered_sales.csv"
)

# Actual column names in the processed dataset.
REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "Price",
]


@pytest.fixture(scope="module")
def sales_sample():
    if not DATASET.is_file():
        pytest.fail(f"Dataset not found: {DATASET}")

    return pd.read_csv(DATASET, nrows=5000)


@pytest.fixture(scope="module")
def db_engine():
    keys = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    missing = [key for key in keys if not os.getenv(key)]

    if missing:
        pytest.fail(
            "Missing PostgreSQL settings in .env: "
            + ", ".join(missing)
        )

    url = URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        database=os.getenv("POSTGRES_DB"),
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )

    try:
        yield engine
    finally:
        engine.dispose()


def test_sales_dataset_has_required_columns(sales_sample):
    missing = set(REQUIRED_COLUMNS) - set(sales_sample.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"


def test_sales_dataset_is_not_empty(sales_sample):
    assert not sales_sample.empty


def test_quantity_is_numeric(sales_sample):
    quantity = pd.to_numeric(
        sales_sample["Quantity"], errors="coerce"
    )
    assert quantity.notna().all(), "Quantity contains invalid values"


def test_price_is_numeric(sales_sample):
    price = pd.to_numeric(
        sales_sample["Price"], errors="coerce"
    )
    assert price.notna().all(), "Price contains invalid values"


def test_invoice_date_is_parseable(sales_sample):
    dates = pd.to_datetime(
        sales_sample["InvoiceDate"], errors="coerce"
    )
    assert dates.notna().all(), "InvoiceDate contains invalid dates"


def test_postgres_required_columns_have_no_nulls(db_engine):
    """Check essential transaction fields across the full table."""
    with db_engine.connect() as connection:
        result = connection.execute(text("""
            SELECT
                COUNT(*) FILTER (
                    WHERE "Invoice" IS NULL
                ) AS invoice_nulls,
                COUNT(*) FILTER (
                    WHERE "StockCode" IS NULL
                ) AS product_nulls,
                COUNT(*) FILTER (
                    WHERE "Quantity" IS NULL
                ) AS quantity_nulls,
                COUNT(*) FILTER (
                    WHERE "InvoiceDate" IS NULL
                ) AS date_nulls,
                COUNT(*) FILTER (
                    WHERE "Price" IS NULL
                ) AS price_nulls
            FROM public.feature_engineered_sales
        """)).mappings().one()

    null_counts = dict(result)
    assert sum(null_counts.values()) == 0, (
        f"Null counts in essential fields: {null_counts}"
    )


def test_postgres_has_no_negative_prices(db_engine):
    with db_engine.connect() as connection:
        count = connection.execute(text("""
            SELECT COUNT(*)
            FROM public.feature_engineered_sales
            WHERE "Price" < 0
        """)).scalar_one()

    assert count == 0, f"Found {count} rows with negative prices"


def test_postgres_has_no_zero_quantities(db_engine):
    with db_engine.connect() as connection:
        count = connection.execute(text("""
            SELECT COUNT(*)
            FROM public.feature_engineered_sales
            WHERE "Quantity" = 0
        """)).scalar_one()

    assert count == 0, f"Found {count} rows with zero quantity"
