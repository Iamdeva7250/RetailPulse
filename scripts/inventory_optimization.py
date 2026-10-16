
"""
RetailPulse - Step 12: Inventory Optimization
---------------------------------------------
Input:
    data/processed/feature_engineered_sales.csv

Outputs:
    data/processed/abc_product_analysis.csv
    data/processed/inventory_optimization.csv
    data/processed/inventory_priority_products.csv
    data/processed/inventory_risk_summary.csv

Features:
    1. Product demand analysis
    2. ABC classification using revenue
    3. Demand variability analysis
    4. Lead-time demand
    5. Safety stock and reorder point
    6. Recommended target stock level
    7. Inventory risk scoring and prioritization

Note:
    Recommendations use estimated demand and assumed lead time.
    Actual stock-on-hand, supplier lead times, and open purchase orders
    are not available in the input dataset.
"""

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_sales.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ABC_OUTPUT = OUTPUT_DIR / "abc_product_analysis.csv"
INVENTORY_OUTPUT = OUTPUT_DIR / "inventory_optimization.csv"
PRIORITY_OUTPUT = OUTPUT_DIR / "inventory_priority_products.csv"
RISK_SUMMARY_OUTPUT = OUTPUT_DIR / "inventory_risk_summary.csv"


# Planning assumptions. Adjust these when actual supplier data is available.
LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL_Z = 1.65  # Approx. 95% one-sided service-level factor

# ABC thresholds based on cumulative revenue contribution.
ABC_A_THRESHOLD = 80.0
ABC_B_THRESHOLD = 95.0

# Risk-score weights. These are configurable heuristics, not learned parameters.
ABC_RISK_WEIGHT = 35.0
VARIABILITY_RISK_WEIGHT = 25.0
DEMAND_RISK_WEIGHT = 40.0

# Stabilizes variability scoring for low-volume products.
# This constant is used only for scoring; it does not alter raw demand metrics.
VARIABILITY_STABILIZER = 5.0
VARIABILITY_SCORE_CAP = 2.0


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates, required=True):
    """Find a column by case-insensitive name or known alias."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        original = lookup.get(candidate.strip().lower())
        if original is not None:
            return original

    if required:
        raise ValueError(
            f"Could not find a required column. "
            f"Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def to_numeric(series):
    """Convert values to numbers, replacing invalid values with NaN."""
    return pd.to_numeric(series, errors="coerce")


def classify_abc(revenue):
    """
    Classify products by cumulative revenue:
    A: first 80% of cumulative revenue
    B: next 15%
    C: remaining revenue
    """
    revenue = revenue.clip(lower=0)

    total_revenue = revenue.sum()

    if total_revenue <= 0:
        return pd.Series(
            ["C"] * len(revenue),
            index=revenue.index,
            dtype="object"
        )

    revenue_share = revenue / total_revenue * 100
    cumulative_share = revenue_share.cumsum()

    # Assign each product according to its cumulative revenue position.
    # The highest-revenue product is included in Class A.
    classes = np.select(
        [
            cumulative_share - revenue_share < ABC_A_THRESHOLD,
            cumulative_share - revenue_share < ABC_B_THRESHOLD,
        ],
        ["A", "B"],
        default="C"
    )

    return pd.Series(classes, index=revenue.index, dtype="object")


def save_csv(df, path):
    """Save a DataFrame to CSV and report its location."""
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ============================================================
# 3. LOAD AND VALIDATE THE DATASET
# ============================================================

def main():
    print("=" * 65)
    print("RETAILPULSE - INVENTORY OPTIMIZATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        print(f"\nERROR: Input dataset not found:\n{INPUT_FILE}")
        print(
            "\nGenerate or restore "
            "data/processed/feature_engineered_sales.csv first."
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nReading dataset:\n{INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    if df.empty:
        raise ValueError("The input dataset contains no rows.")

    # Resolve column names used by different versions of the dataset.
    date_col = find_column(
        df,
        ["InvoiceDate", "Invoice Date", "Date", "TransactionDate"]
    )

    product_col = find_column(
        df,
        ["StockCode", "Stock Code", "ProductID", "Product ID", "ProductCode"]
    )

    quantity_col = find_column(
        df,
        ["Quantity", "Qty", "UnitsSold"]
    )

    price_col = find_column(
        df,
        ["UnitPrice", "Unit Price", "Price", "UnitCost"],
        required=False
    )

    revenue_col = find_column(
        df,
        ["Revenue", "TotalRevenue", "SalesAmount"],
        required=False
    )

    invoice_col = find_column(
        df,
        ["InvoiceNo", "Invoice No", "Invoice", "InvoiceNumber"],
        required=False
    )

    cancelled_col = find_column(
        df,
        ["IsCancelled", "IsCanceled", "Cancelled", "Canceled"],
        required=False
    )

    print("\nResolved columns:")
    print(f"  Date:       {date_col}")
    print(f"  Product:    {product_col}")
    print(f"  Quantity:   {quantity_col}")
    print(f"  Unit price: {price_col}")
    print(f"  Revenue:    {revenue_col}")
    print(f"  Invoice:    {invoice_col}")
    print(f"  Cancelled:  {cancelled_col}")

    # Parse dates and numeric fields.
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df[quantity_col] = to_numeric(df[quantity_col])

    if price_col is not None:
        df[price_col] = to_numeric(df[price_col])

    if revenue_col is not None:
        df[revenue_col] = to_numeric(df[revenue_col])

    df[product_col] = df[product_col].astype("string").str.strip()

    # Remove invalid dates, products, and quantities.
    before = len(df)

    df = df.dropna(
        subset=[date_col, product_col, quantity_col]
    ).copy()

    df = df[
        df[product_col].notna()
        & df[product_col].ne("")
        & df[quantity_col].notna()
    ].copy()

    print(f"Rows removed for invalid date/product/quantity: {before - len(df):,}")

    # Exclude explicit cancellations if a cancellation flag is available.
    if cancelled_col is not None:
        cancellation_values = (
            df[cancelled_col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        cancelled_mask = cancellation_values.isin(
            ["true", "1", "yes", "y", "cancelled", "canceled"]
        )

        removed = int(cancelled_mask.sum())
        df = df.loc[~cancelled_mask].copy()

        print(f"Explicitly cancelled rows removed: {removed:,}")

    # Also exclude credit-note invoices, e.g. C536379.
    if invoice_col is not None:
        invoice_values = (
            df[invoice_col]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        credit_note_mask = invoice_values.str.startswith(
            "C",
            na=False
        )

        removed = int(credit_note_mask.sum())
        df = df.loc[~credit_note_mask].copy()

        print(f"Credit-note invoice rows removed: {removed:,}")

    # Demand planning uses positive sold quantities only.
    before = len(df)

    df = df[df[quantity_col] > 0].copy()

    print(f"Rows removed for non-positive quantity: {before - len(df):,}")

    if df.empty:
        raise ValueError(
            "No positive-quantity sales remain after cleaning."
        )

    # Use unit price to identify positive-price sales when available.
    if price_col is not None:
        df = df[df[price_col].notna()].copy()
        df = df[df[price_col] > 0].copy()

    if df.empty:
        raise ValueError(
            "No usable positive-price sales remain after cleaning."
        )

    # Prefer revenue from quantity * unit price when a unit-price column
    # exists. This avoids relying on a possibly stale precomputed field.
    if price_col is not None:
        df["_RevenueForInventory"] = (
            df[quantity_col] * df[price_col]
        )
    elif revenue_col is not None:
        df["_RevenueForInventory"] = to_numeric(df[revenue_col])
    else:
        raise ValueError(
            "Cannot calculate revenue. Supply a UnitPrice/Price column "
            "or a Revenue column."
        )

    df = df[
        df["_RevenueForInventory"].notna()
        & (df["_RevenueForInventory"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No positive-revenue sales remain.")

    # Normalize transaction dates to calendar days.
    df["_SaleDate"] = df[date_col].dt.normalize()

    print(f"\nUsable sales rows: {len(df):,}")
    print(f"Products: {df[product_col].nunique():,}")
    print(f"Date range: {df['_SaleDate'].min().date()} to "
          f"{df['_SaleDate'].max().date()}")

    # ========================================================
    # 4. PRODUCT-LEVEL DEMAND AND REVENUE
    # ========================================================

    print("\n[1/6] Calculating product demand and revenue...")

    product_summary = (
        df.groupby(product_col, as_index=False)
        .agg(
            TotalDemand=(quantity_col, "sum"),
            TotalRevenue=("_RevenueForInventory", "sum")
        )
        .rename(columns={product_col: "StockCode"})
    )

    # Consistent product identifiers in outputs.
    product_summary["StockCode"] = (
        product_summary["StockCode"].astype(str)
    )

    product_summary["TotalDemand"] = (
        product_summary["TotalDemand"].astype(float)
    )

    product_summary["TotalRevenue"] = (
        product_summary["TotalRevenue"].astype(float)
    )

    # ========================================================
    # 5. DAILY PRODUCT DEMAND INCLUDING ZERO-SALES DAYS
    # ========================================================

    print("[2/6] Calculating daily demand and variability...")

    start_date = df["_SaleDate"].min()
    end_date = df["_SaleDate"].max()

    calendar = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    calendar_days = len(calendar)

    if calendar_days == 0:
        raise ValueError("Unable to create a daily demand calendar.")

    daily_demand = (
        df.groupby(
            [product_col, "_SaleDate"],
            as_index=False
        )[quantity_col]
        .sum()
        .rename(
            columns={
                product_col: "StockCode",
                "_SaleDate": "Date",
                quantity_col: "DailyDemand"
            }
        )
    )

    daily_demand["StockCode"] = (
        daily_demand["StockCode"].astype(str)
    )

    # Calculate statistics on the observed daily demand values.
    # Missing product/date combinations are zero-demand days.
    product_codes = product_summary["StockCode"].tolist()

    full_index = pd.MultiIndex.from_product(
        [product_codes, calendar],
        names=["StockCode", "Date"]
    )

    daily_series = (
        daily_demand
        .set_index(["StockCode", "Date"])["DailyDemand"]
        .reindex(full_index, fill_value=0)
    )

    daily_stats = (
        daily_series.groupby(level="StockCode")
        .agg(
            AverageDailyDemand="mean",
            DemandStdDev="std"
        )
        .reset_index()
    )

    daily_stats["DemandStdDev"] = (
        daily_stats["DemandStdDev"].fillna(0)
    )

    daily_stats["AverageDailyDemand"] = (
        daily_stats["AverageDailyDemand"].clip(lower=0)
    )

    # Coefficient of variation is preserved as a raw diagnostic metric.
    # When average demand is zero, CV is undefined and stored as zero.
    daily_stats["DemandCV"] = np.where(
        daily_stats["AverageDailyDemand"] > 0,
        daily_stats["DemandStdDev"]
        / daily_stats["AverageDailyDemand"],
        0.0
    )

    # Classify variability for descriptive reporting.
    # These labels are descriptive, not the final inventory risk score.
    daily_stats["DemandVariability"] = np.select(
        [
            daily_stats["DemandCV"] <= 0.5,
            daily_stats["DemandCV"] <= 1.0,
            daily_stats["DemandCV"] <= 2.0,
        ],
        [
            "Low",
            "Moderate",
            "High",
        ],
        default="Very High"
    )

    # ========================================================
    # 6. ABC ANALYSIS
    # ========================================================

    print("[3/6] Performing ABC product classification...")

    product_summary = product_summary.sort_values(
        "TotalRevenue",
        ascending=False
    ).reset_index(drop=True)

    total_revenue = product_summary["TotalRevenue"].sum()

    if total_revenue <= 0:
        raise ValueError("Total product revenue must be positive.")

    product_summary["RevenuePercentage"] = (
        product_summary["TotalRevenue"] / total_revenue * 100
    )

    product_summary["CumulativeRevenuePercentage"] = (
        product_summary["RevenuePercentage"].cumsum()
    )

    product_summary["ABC_Class"] = classify_abc(
        product_summary["TotalRevenue"]
    )

    # ABC output contains the expected columns.
    abc_analysis = product_summary[
        [
            "StockCode",
            "TotalRevenue",
            "RevenuePercentage",
            "CumulativeRevenuePercentage",
            "ABC_Class"
        ]
    ].copy()

    abc_analysis[
        [
            "TotalRevenue",
            "RevenuePercentage",
            "CumulativeRevenuePercentage"
        ]
    ] = abc_analysis[
        [
            "TotalRevenue",
            "RevenuePercentage",
            "CumulativeRevenuePercentage"
        ]
    ].round(2)

    save_csv(abc_analysis, ABC_OUTPUT)

    # ========================================================
    # 7. INVENTORY PLANNING CALCULATIONS
    # ========================================================

    print("[4/6] Calculating safety stock and reorder points...")

    inventory = product_summary.merge(
        daily_stats,
        on="StockCode",
        how="left",
        validate="one_to_one"
    )

    inventory["AverageDailyDemand"] = (
        inventory["AverageDailyDemand"].fillna(0)
    )

    inventory["DemandStdDev"] = (
        inventory["DemandStdDev"].fillna(0)
    )

    inventory["DemandCV"] = (
        inventory["DemandCV"].replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)
    )

    # Expected demand during supplier lead time.
    inventory["LeadTimeDemand"] = (
        inventory["AverageDailyDemand"] * LEAD_TIME_DAYS
    )

    # Safety stock based on daily demand variation and lead time.
    inventory["SafetyStock"] = (
        SERVICE_LEVEL_Z
        * inventory["DemandStdDev"]
        * math.sqrt(LEAD_TIME_DAYS)
    )

    # Reorder point = expected lead-time demand + safety stock.
    inventory["ReorderPoint"] = (
        inventory["LeadTimeDemand"]
        + inventory["SafetyStock"]
    )

    # Target stock level covers lead time plus the review period,
    # with additional safety stock.
    inventory["RecommendedStockLevel"] = (
        inventory["AverageDailyDemand"]
        * (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS)
        + inventory["SafetyStock"]
    )

    # Without actual on-hand stock, this is a target quantity assuming
    # zero available stock, not an actual purchase order.
    inventory["RecommendedOrderQuantity"] = np.ceil(
        inventory["RecommendedStockLevel"].clip(lower=0)
    ).astype("int64")

    # ========================================================
    # 8. IMPROVED INVENTORY RISK SCORING
    # ========================================================

    print("[5/6] Calculating inventory risk scores...")

    inventory["ABC_RiskWeight"] = (
        inventory["ABC_Class"]
        .map({
            "A": 1.0,
            "B": 0.6,
            "C": 0.3
        })
        .fillna(0.3)
    )

    # Stabilized variability is used only for the risk score.
    # Raw DemandCV remains available for analysis.
    stable_variability = (
        inventory["DemandStdDev"]
        / (
            inventory["AverageDailyDemand"]
            + VARIABILITY_STABILIZER
        )
    ).clip(lower=0)

    variability_component = (
        stable_variability / VARIABILITY_SCORE_CAP
    ).clip(0, 1)

    demand_reference = inventory[
        "AverageDailyDemand"
    ].quantile(0.95)

    if pd.isna(demand_reference) or demand_reference <= 0:
        demand_reference = 1.0

    demand_component = (
        inventory["AverageDailyDemand"] / demand_reference
    ).clip(0, 1)

    inventory["InventoryRiskScore"] = (
        ABC_RISK_WEIGHT * inventory["ABC_RiskWeight"]
        + VARIABILITY_RISK_WEIGHT * variability_component
        + DEMAND_RISK_WEIGHT * demand_component
    ).clip(0, 100).round(2)

    # Transparent thresholds:
    # Low:      score <= 30
    # Moderate: 30 < score <= 50
    # High:     50 < score <= 70
    # Critical: score > 70
    inventory["InventoryRiskCategory"] = pd.cut(
        inventory["InventoryRiskScore"],
        bins=[-0.01, 30, 50, 70, 100.01],
        labels=["Low", "Moderate", "High", "Critical"],
        right=True
    ).astype("object")

    inventory["InventoryStatus"] = (
        inventory["InventoryRiskCategory"]
        .map({
            "Low": "Routine Monitoring",
            "Moderate": "Monitor",
            "High": "Review Soon",
            "Critical": "Urgent Review"
        })
    )

    # ========================================================
    # 9. ROUNDING AND OUTPUT COLUMNS
    # ========================================================

    numeric_columns = [
        "TotalDemand",
        "AverageDailyDemand",
        "DemandStdDev",
        "DemandCV",
        "TotalRevenue",
        "RevenuePercentage",
        "CumulativeRevenuePercentage",
        "LeadTimeDemand",
        "SafetyStock",
        "ReorderPoint",
        "RecommendedStockLevel"
    ]

    for col in numeric_columns:
        if col in inventory.columns:
            inventory[col] = inventory[col].round(2)

    # Select and order output columns to maintain compatibility.
    output_columns = [
        "StockCode",
        "TotalDemand",
        "AverageDailyDemand",
        "DemandStdDev",
        "DemandCV",
        "DemandVariability",
        "TotalRevenue",
        "RevenuePercentage",
        "CumulativeRevenuePercentage",
        "ABC_Class",
        "LeadTimeDemand",
        "SafetyStock",
        "ReorderPoint",
        "RecommendedStockLevel",
        "RecommendedOrderQuantity",
        "InventoryRiskScore",
        "InventoryRiskCategory",
        "InventoryStatus"
    ]

    inventory_output = inventory[output_columns].copy()

    inventory_output = inventory_output.sort_values(
        ["InventoryRiskScore", "TotalRevenue"],
        ascending=[False, False]
    ).reset_index(drop=True)

    save_csv(inventory_output, INVENTORY_OUTPUT)

    # Priority products: all High and Critical products.
    priority_products = inventory_output[
        inventory_output["InventoryRiskCategory"].isin(
            ["High", "Critical"]
        )
    ].copy()

    save_csv(priority_products, PRIORITY_OUTPUT)

    # ========================================================
    # 10. RISK SUMMARY
    # ========================================================

    risk_summary = (
        inventory_output
        .groupby(
            "InventoryRiskCategory",
            observed=False,
            dropna=False
        )
        .agg(
            Products=("StockCode", "nunique"),
            AverageRiskScore=("InventoryRiskScore", "mean"),
            AverageDailyDemand=("AverageDailyDemand", "mean"),
            TotalRevenue=("TotalRevenue", "sum")
        )
        .reset_index()
    )

    risk_summary["AverageRiskScore"] = (
        risk_summary["AverageRiskScore"].round(2)
    )

    risk_summary["AverageDailyDemand"] = (
        risk_summary["AverageDailyDemand"].round(2)
    )

    risk_summary["TotalRevenue"] = (
        risk_summary["TotalRevenue"].round(2)
    )

    save_csv(risk_summary, RISK_SUMMARY_OUTPUT)

    # ========================================================
    # 11. FINAL VALIDATION AND SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("INVENTORY OPTIMIZATION COMPLETED")
    print("=" * 65)

    print(f"\nProducts analyzed: {len(inventory_output):,}")
    print(f"Calendar days analyzed: {calendar_days:,}")
    print(f"Total revenue: {total_revenue:,.2f}")

    print("\nABC class counts:")
    print(inventory_output["ABC_Class"].value_counts().to_string())

    print("\nInventory risk category counts:")
    print(
        inventory_output["InventoryRiskCategory"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High", "Critical"],
            fill_value=0
        )
        .to_string()
    )

    print("\nInventory status counts:")
    print(inventory_output["InventoryStatus"].value_counts().to_string())

    print("\nRisk score summary:")
    print(
        inventory_output["InventoryRiskScore"]
        .describe()
        .round(2)
        .to_string()
    )

    # Check that the reorder-point formula holds before rounding.
    formula_difference = (
        inventory["LeadTimeDemand"]
        + inventory["SafetyStock"]
        - inventory["ReorderPoint"]
    ).abs()

    print(
        "\nMaximum reorder-point formula difference "
        f"(before rounding): {formula_difference.max():.10f}"
    )

    # Check the saved output columns and key values.
    if inventory_output["StockCode"].duplicated().any():
        raise ValueError(
            "Validation failed: duplicate product codes in output."
        )

    if inventory_output["InventoryRiskScore"].isna().any():
        raise ValueError(
            "Validation failed: missing inventory risk scores."
        )

    if not inventory_output["InventoryRiskScore"].between(0, 100).all():
        raise ValueError(
            "Validation failed: risk scores outside 0–100."
        )

    if (inventory_output["RecommendedOrderQuantity"] < 0).any():
        raise ValueError(
            "Validation failed: negative recommended order quantities."
        )

    print("\nAll four output files have been generated:")
    print(f"1. {ABC_OUTPUT}")
    print(f"2. {INVENTORY_OUTPUT}")
    print(f"3. {PRIORITY_OUTPUT}")
    print(f"4. {RISK_SUMMARY_OUTPUT}")

    print("\nPlanning assumptions:")
    print(f"  Supplier lead time: {LEAD_TIME_DAYS} days")
    print(f"  Review period:      {REVIEW_PERIOD_DAYS} days")
    print(f"  Service-level Z:    {SERVICE_LEVEL_Z}")
    print(
        "\nReminder: order quantities assume zero available stock. "
        "Use actual on-hand stock and supplier lead times before "
        "placing real orders."
    )


if __name__ == "__main__":
    main()
