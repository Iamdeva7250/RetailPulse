from pathlib import Path
from getpass import getpass

import pandas as pd
from sqlalchemy import URL, create_engine, inspect, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_sales.csv"

DATABASE = "retailpulse"
USERNAME = "postgres"
HOST = "localhost"
PORT = 5432
TABLE_NAME = "feature_engineered_sales"


def main():
    # Check CSV
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    print(f"Reading: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # PostgreSQL password
    password = getpass("Enter your PostgreSQL password (not shown): ")

    # Create database URL
    url = URL.create(
        "postgresql+psycopg",
        username=USERNAME,
        password=password,
        host=HOST,
        port=PORT,
        database=DATABASE,
    )

    engine = create_engine(url)

    try:
        # Test connection
        with engine.connect() as connection:
            print(
                "Connected to:",
                connection.execute(
                    text("SELECT current_database()")
                ).scalar_one()
            )

        # Check table
        inspector = inspect(engine)

        if inspector.has_table(TABLE_NAME, schema="public"):

            with engine.connect() as connection:
                existing_rows = connection.execute(
                    text(
                        "SELECT COUNT(*) "
                        "FROM public.feature_engineered_sales"
                    )
                ).scalar_one()

            # Stop if table already has data
            if existing_rows > 0:
                print(
                    f"STOP: table already contains {existing_rows:,} rows. "
                    "No data was imported."
                )
                return

            print("Existing table is empty; proceeding with import.")

        else:
            print(
                f"Table '{TABLE_NAME}' does not exist. "
                "It will be created automatically."
            )

        # Import CSV into PostgreSQL
        print("Importing data in batches. Please wait...")

        df.to_sql(
            TABLE_NAME,
            engine,
            schema="public",
            if_exists="append",
            index=False,
            chunksize=1000,
            method=None,
        )

        # Verify import
        with engine.connect() as connection:
            row_count = connection.execute(
                text(
                    "SELECT COUNT(*) "
                    "FROM public.feature_engineered_sales"
                )
            ).scalar_one()

        print(f"Import complete: {row_count:,} rows loaded.")
        print("PostgreSQL table: public.feature_engineered_sales")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()