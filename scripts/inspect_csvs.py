
from pathlib import Path
import pandas as pd
import json

# RetailPulse project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "retail_clean.csv",
    "retail_sales.csv",
    "feature_engineered_sales.csv",
]

report = {}

for filename in files:
    path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    if not path.exists():
        print("STATUS: File not found")
        report[filename] = {"status": "File not found"}
        continue

    try:
        # Inspect only; do not modify the original CSV.
        df = pd.read_csv(path, low_memory=False)

        columns = [str(col) for col in df.columns]
        missing = {
            str(col): int(count)
            for col, count in df.isna().sum().items()
            if count > 0
        }

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print("\nColumn names:")
        print(columns)

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nMissing values:")
        print(missing if missing else "None")

        print(f"\nDuplicate rows: {df.duplicated().sum():,}")
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

        report[filename] = {
            "status": "success",
            "rows": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": columns,
            "data_types": {
                str(col): str(dtype)
                for col, dtype in df.dtypes.items()
            },
            "missing_values": missing,
            "duplicate_rows": int(df.duplicated().sum()),
            "file_size_bytes": path.stat().st_size,
        }

    except Exception as exc:
        print(f"ERROR: {exc}")
        report[filename] = {
            "status": "error",
            "message": str(exc),
        }

report_path = OUTPUT_DIR / "sql_csv_inspection.json"
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 70)
print(f"Inspection report saved to:\n{report_path}")
print("Original CSV files were not modified.")
