
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

EXPECTED_VIEWS = [
    "vw_retailpulse_country_sales",
    "vw_retailpulse_customer_sales",
    "vw_retailpulse_daily_sales",
    "vw_retailpulse_monthly_sales",
    "vw_retailpulse_product_sales",
    "vw_retailpulse_sales_kpis",
]


@pytest.fixture(scope="module")
def db_engine():
    required = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    missing = [key for key in required if not os.getenv(key)]

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


def test_postgres_connection(db_engine):
    with db_engine.connect() as connection:
        database = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

    assert database == "retailpulse"


def test_feature_engineered_sales_table_exists(db_engine):
    with db_engine.connect() as connection:
        exists = connection.execute(
            text("""
                SELECT to_regclass(
                    'public.feature_engineered_sales'
                ) IS NOT NULL
            """)
        ).scalar_one()

    assert exists is True


def test_feature_engineered_sales_has_rows(db_engine):
    with db_engine.connect() as connection:
        row_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM public.feature_engineered_sales
            """)
        ).scalar_one()

    assert row_count > 0


def test_feature_engineered_sales_expected_row_count(db_engine):
    with db_engine.connect() as connection:
        row_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM public.feature_engineered_sales
            """)
        ).scalar_one()

    assert row_count == 779425


def test_feature_engineered_sales_has_48_columns(db_engine):
    with db_engine.connect() as connection:
        column_count = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'feature_engineered_sales'
            """)
        ).scalar_one()

    assert column_count == 48


@pytest.mark.parametrize("view_name", EXPECTED_VIEWS)
def test_analytics_view_exists(db_engine, view_name):
    with db_engine.connect() as connection:
        exists = connection.execute(
            text("""
                SELECT to_regclass(
                    'public.' || :view_name
                ) IS NOT NULL
            """),
            {"view_name": view_name},
        ).scalar_one()

    assert exists is True, f"Missing analytics view: {view_name}"
