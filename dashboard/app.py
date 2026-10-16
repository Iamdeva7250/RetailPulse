
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RetailPulse | Retail Analytics",
    page_icon="ðŸ“Š",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load PostgreSQL settings without displaying credentials.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

load_dotenv(PROJECT_DIR / ".env")


@st.cache_resource
def get_postgres_engine():
    """Create a reusable PostgreSQL connection engine."""
    required_keys = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]

    missing = [
        key for key in required_keys
        if not os.getenv(key)
    ]

    if missing:
        raise ValueError(
            "Missing PostgreSQL settings in the project .env file: "
            + ", ".join(missing)
        )

    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        database=os.getenv("POSTGRES_DB"),
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=0,
        connect_args={"connect_timeout": 5},
    )


def check_postgres_connection():
    """Test the database and verify the RetailPulse sales table."""
    engine = get_postgres_engine()

    with engine.connect() as connection:
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        table_rows = connection.execute(
            text(
                "SELECT COUNT(*) "
                "FROM public.feature_engineered_sales"
            )
        ).scalar_one()

    return database_name, table_rows



st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            padding: 14px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. DATASET FILES
# ============================================================

FILES = {
    "clean_sales": "retail_clean.csv",
    "retail_sales": "retail_sales.csv",
    "engineered_sales": "feature_engineered_sales.csv",
    "customer_features": "customer_features.csv",
    "rfm_segments": "rfm_customer_segments.csv",
    "rfm_summary": "rfm_segment_summary.csv",
    "customer_clusters": "customer_clusters.csv",
    "cluster_summary": "cluster_summary.csv",
    "churn_predictions": "customer_churn_predictions.csv",
    "churn_comparison": "churn_model_comparison.csv",
    "churn_importance": "churn_feature_importance.csv",
    "daily_features": "daily_sales_features.csv",
    "monthly_features": "monthly_sales_features.csv",
    "product_demand": "product_daily_demand.csv",
    "forecast_evaluation": "forecast_model_evaluation.csv",
    "monthly_forecast": "monthly_revenue_forecast.csv",
    "forecast_90_days": "revenue_forecast_90_days.csv",
    "lstm_evaluation": "lstm_forecast_evaluation.csv",
    "lstm_test": "lstm_test_predictions.csv",
    "lstm_forecast": "lstm_revenue_forecast_90_days.csv",
    "abc_products": "abc_product_analysis.csv",
    "inventory": "inventory_optimization.csv",
    "inventory_priority": "inventory_priority_products.csv",
    "inventory_risk": "inventory_risk_summary.csv",
    "product_features": "product_features.csv",
}


# ============================================================
# 3. DATA LOADING AND HELPER FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(path_string, modified_time):
    """Load and cache one CSV file."""
    del modified_time  # Included in the cache key to refresh changed files.
    return pd.read_csv(path_string, low_memory=False)


def get_dataset(filename):
    """Return a dataset or None when it is unavailable."""
    path = DATA_DIR / filename

    if not path.exists():
        return None

    try:
        return load_csv(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None


def get_file(key):
    """Load a dataset by its FILES dictionary key."""
    filename = FILES.get(key)
    return get_dataset(filename) if filename else None


def normalize_column(name):
    """Normalize a column name for flexible matching."""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_column(df, candidates):
    """Find the first matching column name."""
    if df is None or df.empty:
        return None

    normalized = {
        normalize_column(column): column
        for column in df.columns
    }

    for candidate in candidates:
        match = normalized.get(normalize_column(candidate))
        if match is not None:
            return match

    # Allow partial matches for differently named columns.
    for candidate in candidates:
        target = normalize_column(candidate)
        for normalized_name, original_name in normalized.items():
            if target and (
                target in normalized_name
                or normalized_name in target
            ):
                return original_name

    return None


def numeric_column(df, candidates):
    """Return a numeric Series for the first matching column."""
    column = find_column(df, candidates)

    if column is None:
        return None

    return pd.to_numeric(df[column], errors="coerce")


def metric_value(df, candidates, operation="sum"):
    """Calculate a numeric metric safely."""
    values = numeric_column(df, candidates)

    if values is None:
        return None

    values = values.dropna()

    if values.empty:
        return None

    if operation == "sum":
        return float(values.sum())
    if operation == "mean":
        return float(values.mean())
    if operation == "median":
        return float(values.median())
    if operation == "max":
        return float(values.max())

    return None


def format_number(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def format_integer(value):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{int(value):,}"


def show_metric_row(metrics):
    """Display metric cards: [(label, value, help_text), ...]."""
    columns = st.columns(len(metrics))

    for column, item in zip(columns, metrics):
        label, value, *help_text = item
        column.metric(
            label,
            value,
            help=help_text[0] if help_text else None,
        )


def show_table(df, title=None, max_rows=500):
    if title:
        st.subheader(title)

    if df is None:
        st.info("This dataset is not available yet.")
        return

    st.caption(
        f"{len(df):,} rows Ã— {len(df.columns):,} columns"
    )
    st.dataframe(
        df.head(max_rows),
        use_container_width=True,
        hide_index=True,
    )

    if len(df) > max_rows:
        st.caption(
            f"Showing the first {max_rows:,} rows. "
            "Download the file to access all rows."
        )


def show_download(df, filename, key):
    if df is None:
        return

    st.download_button(
        label=f"Download {filename}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def show_chart(df, x_candidates, y_candidates, chart_type="bar",
               title=None):
    """Draw a chart when suitable columns exist."""
    if df is None or df.empty:
        st.info("No data is available for this chart.")
        return

    x_column = find_column(df, x_candidates)
    y_column = find_column(df, y_candidates)

    if y_column is None:
        st.info(
            "The expected numeric column was not found. "
            "Use the dataset explorer to inspect the available columns."
        )
        return

    chart_df = df.copy()
    chart_df[y_column] = pd.to_numeric(
        chart_df[y_column], errors="coerce"
    )
    chart_df = chart_df.dropna(subset=[y_column])

    if x_column is not None:
        chart_df = chart_df[[x_column, y_column]].copy()
        chart_df = chart_df.groupby(
            x_column, dropna=False, as_index=False
        )[y_column].sum()
        chart_df = chart_df.sort_values(x_column)

        if len(chart_df) > 100:
            chart_df = chart_df.tail(100)

        chart_df = chart_df.set_index(x_column)
    else:
        chart_df = chart_df[[y_column]].head(100)

    if chart_df.empty:
        st.info("There are no usable numeric values to chart.")
        return

    if title:
        st.subheader(title)

    if chart_type == "line":
        st.line_chart(chart_df)
    elif chart_type == "area":
        st.area_chart(chart_df)
    else:
        st.bar_chart(chart_df)


def show_missing_notice(df, filename):
    if df is None:
        st.warning(
            f"`{filename}` was not found or could not be read "
            "from `data/processed/`."
        )


def find_date_column(df):
    return find_column(
        df,
        ["InvoiceDate", "Invoice Date", "Date", "OrderDate",
         "Order Date", "ds", "date"],
    )


def find_revenue_column(df):
    return find_column(
        df,
        ["Revenue", "TotalRevenue", "Total Revenue",
         "Sales", "SalesAmount", "Amount", "PredictedRevenue",
         "ForecastRevenue", "yhat"],
    )


def find_customer_column(df):
    return find_column(
        df,
        ["CustomerID", "Customer ID", "Customer",
         "CustomerCode", "customer_id"],
    )


def list_csv_files():
    if not DATA_DIR.exists():
        return []

    return sorted(DATA_DIR.glob("*.csv"))


def inspect_csv_files():
    """List every CSV without loading complete datasets into memory."""
    rows = []

    for path in list_csv_files():
        try:
            # Read only the header to determine column count.
            header = pd.read_csv(path, nrows=0)

            # Count physical lines without constructing a large DataFrame.
            # This is suitable for ordinary CSVs without embedded newlines.
            with path.open("rb") as file:
                row_count = max(sum(1 for _ in file) - 1, 0)

            rows.append({
                "Dataset": path.stem.replace("_", " ").title(),
                "File": path.name,
                "Status": "Readable",
                "Rows": row_count,
                "Columns": len(header.columns),
            })

        except Exception as exc:
            rows.append({
                "Dataset": path.stem.replace("_", " ").title(),
                "File": path.name,
                "Status": f"Read error: {exc}",
                "Rows": None,
                "Columns": None,
            })

    return pd.DataFrame(rows)


# ============================================================
# 4. HEADER AND SIDEBAR
# ============================================================

st.title("📊 RetailPulse")

st.markdown(
    "AI-powered customer analytics, sales intelligence, "
    "demand forecasting, and inventory optimization."
)

with st.sidebar:
    st.header("RetailPulse")
    st.caption("Retail analytics dashboard")

    st.markdown("---")
    st.write("**Project data location**")
    st.code(str(DATA_DIR), language=None)

    if DATA_DIR.exists():
        csv_count = len(list_csv_files())
        st.success(f"{csv_count} CSV files detected")
    else:
        st.error("The processed data folder was not found.")

    if st.button("Refresh cached data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

   
    st.markdown("---")
    st.subheader("PostgreSQL Status")

    try:
        database_name, table_rows = check_postgres_connection()

        st.success("PostgreSQL connected")
        st.caption(f"Database: {database_name}")
        st.metric("Database sales rows", f"{table_rows:,}")

    except Exception as exc:
        st.warning(
            "PostgreSQL connection unavailable. "
            "CSV dashboard remains available."
        )
        st.caption(f"Diagnostic: {type(exc).__name__}: {exc}")

    st.markdown("---")
    st.caption(
        "Dashboard charts currently read CSV files "
        "from data/processed/."
    )
    st.caption("Power BI files are not modified.")



# ============================================================
# 5. MAIN DASHBOARD TABS
# ============================================================

tab_overview, tab_sales, tab_customers, tab_churn, \
tab_forecasting, tab_inventory, tab_health = st.tabs(
[
    "📊 Overview",
    "💰 Sales Analytics",
    "👥 Customers",
    "🔍 Churn",
    "📈 Forecasting",
    "📦 Inventory",
    "🗂️ Data Health & CSV Explorer",
]
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab_overview:
    st.header("Business Overview")
    st.write(
        "A high-level view of revenue, transactions, customers, "
        "products, churn predictions, and inventory risk."
    )

    sales = get_file("engineered_sales")
    if sales is None:
        sales = get_file("retail_sales")
    if sales is None:
        sales = get_file("clean_sales")

    customers = get_file("customer_features")
    churn = get_file("churn_predictions")
    inventory = get_file("inventory")
    forecast = get_file("forecast_90_days")

    revenue = metric_value(
        sales,
        ["Revenue", "TotalRevenue", "Sales", "Amount"],
        "sum",
    )
    quantity = metric_value(
        sales, ["Quantity", "TotalQuantity"], "sum"
    )
    customer_col = find_customer_column(customers)
    customer_count = (
        customers[customer_col].nunique()
        if customers is not None and customer_col is not None
        else None
    )

    product_col = find_column(
        inventory, ["StockCode", "ProductID", "Product", "SKU"]
    )
    product_count = (
        inventory[product_col].nunique()
        if inventory is not None and product_col is not None
        else None
    )

    risk_col = find_column(
        inventory, ["RiskCategory", "InventoryRisk", "RiskLevel",
                    "Risk Category"]
    )
    high_risk_count = None

    if inventory is not None and risk_col is not None:
        risk_text = inventory[risk_col].astype(str).str.lower()
        high_risk_count = int(
            risk_text.str.contains("high|critical", regex=True).sum()
        )

    churn_col = find_column(
        churn, ["ChurnPrediction", "PredictedChurn", "Churn",
                "ChurnLabel", "Prediction"]
    )
    predicted_churn_count = None

    if churn is not None and churn_col is not None:
        values = churn[churn_col]
        if pd.api.types.is_numeric_dtype(values):
            predicted_churn_count = int(
                pd.to_numeric(values, errors="coerce").fillna(0).sum()
            )
        else:
            text_values = values.astype(str).str.lower()
            predicted_churn_count = int(
                text_values.isin(
                    ["1", "yes", "true", "churn", "churned"]
                ).sum()
            )

    forecast_total = metric_value(
        forecast,
        ["ForecastRevenue", "PredictedRevenue", "Revenue",
         "yhat", "Forecast"],
        "sum",
    )

    show_metric_row([
        ("Total Revenue", format_number(revenue)),
        ("Quantity Sold", format_integer(quantity)),
        ("Customers", format_integer(customer_count)),
        ("Products", format_integer(product_count)),
    ])

    show_metric_row([
        ("Predicted Churners", format_integer(predicted_churn_count)),
        ("High/Critical Risk Products", format_integer(high_risk_count)),
        ("90-Day Forecast Total", format_number(forecast_total)),
        ("Processed CSV Files", format_integer(len(list_csv_files()))),
    ])

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        show_chart(
            sales,
            ["InvoiceDate", "Date", "OrderDate"],
            ["Revenue", "TotalRevenue", "Sales", "Amount"],
            "line",
            "Sales / Revenue Trend",
        )

    with right:
        show_chart(
            get_file("rfm_summary"),
            ["Segment", "RFM_Segment", "CustomerSegment"],
            ["CustomerCount", "Count", "Customers", "NumCustomers"],
            "bar",
            "RFM Customer Segments",
        )

    st.markdown("---")
    st.subheader("Recently Available Forecast")
    if forecast is not None:
        show_table(forecast, max_rows=10)
        show_download(
            forecast,
            FILES["forecast_90_days"],
            "overview_forecast_download",
        )
    else:
        st.info("The 90-day forecast file is not available.")


# ============================================================
# TAB 2: SALES ANALYTICS
# ============================================================

with tab_sales:
    st.header("Sales Analytics")

    sales_source = st.selectbox(
        "Choose sales dataset",
        [
            FILES["engineered_sales"],
            FILES["retail_sales"],
            FILES["clean_sales"],
            FILES["daily_features"],
            FILES["monthly_features"],
        ],
        key="sales_source_select",
    )
    sales_df = get_dataset(sales_source)

    if sales_df is not None:
        revenue_col = find_revenue_column(sales_df)
        quantity_col = find_column(
            sales_df, ["Quantity", "TotalQuantity"]
        )
        invoice_col = find_column(
            sales_df, ["InvoiceNo", "Invoice", "OrderID", "TransactionID"]
        )
        date_col = find_date_column(sales_df)
        country_col = find_column(sales_df, ["Country"])
        product_col = find_column(
            sales_df, ["StockCode", "ProductID", "Product", "SKU"]
        )

        total_revenue = (
            pd.to_numeric(sales_df[revenue_col], errors="coerce").sum()
            if revenue_col else None
        )
        total_quantity = (
            pd.to_numeric(sales_df[quantity_col], errors="coerce").sum()
            if quantity_col else None
        )
        invoices = (
            sales_df[invoice_col].nunique()
            if invoice_col else None
        )
        countries = (
            sales_df[country_col].nunique()
            if country_col else None
        )

        show_metric_row([
            ("Revenue", format_number(total_revenue)),
            ("Quantity", format_integer(total_quantity)),
            ("Transactions", format_integer(invoices)),
            ("Countries", format_integer(countries)),
        ])

        st.markdown("---")
        left, right = st.columns(2)

        with left:
            show_chart(
                sales_df,
                [date_col] if date_col else ["InvoiceDate", "Date"],
                [revenue_col] if revenue_col else ["Revenue", "Sales"],
                "line",
                "Revenue Over Time",
            )

        with right:
            if country_col and revenue_col:
                country_sales = sales_df.copy()
                country_sales[revenue_col] = pd.to_numeric(
                    country_sales[revenue_col], errors="coerce"
                )
                country_sales = (
                    country_sales.groupby(country_col, as_index=False)
                    [revenue_col].sum()
                    .sort_values(revenue_col, ascending=False)
                    .head(15)
                )
                st.subheader("Top Countries by Revenue")
                st.bar_chart(
                    country_sales.set_index(country_col)[revenue_col]
                )
            elif product_col and revenue_col:
                product_sales = sales_df.copy()
                product_sales[revenue_col] = pd.to_numeric(
                    product_sales[revenue_col], errors="coerce"
                )
                product_sales = (
                    product_sales.groupby(product_col, as_index=False)
                    [revenue_col].sum()
                    .sort_values(revenue_col, ascending=False)
                    .head(15)
                )
                st.subheader("Top Products by Revenue")
                st.bar_chart(
                    product_sales.set_index(product_col)[revenue_col]
                )
            else:
                st.info(
                    "Country or product revenue chart is unavailable "
                    "for the selected dataset."
                )

        st.markdown("---")
        show_table(sales_df, "Sales Dataset Preview", max_rows=100)
        show_download(
            sales_df, sales_source, "sales_dataset_download"
        )
    else:
        show_missing_notice(None, sales_source)


# ============================================================
# TAB 3: CUSTOMER ANALYTICS
# ============================================================

with tab_customers:
    st.header("Customer Analytics")

    customer_choice = st.selectbox(
        "Choose customer dataset",
        [
            FILES["customer_features"],
            FILES["rfm_segments"],
            FILES["rfm_summary"],
            FILES["customer_clusters"],
            FILES["cluster_summary"],
            FILES["product_features"],
        ],
        key="customer_dataset_select",
    )

    customer_df = get_dataset(customer_choice)

    if customer_df is not None:
        customer_id_col = find_customer_column(customer_df)
        segment_col = find_column(
            customer_df,
            ["Segment", "RFM_Segment", "CustomerSegment", "Cluster",
             "ClusterLabel", "ClusterName"],
        )
        recency_col = find_column(customer_df, ["Recency"])
        frequency_col = find_column(customer_df, ["Frequency"])
        monetary_col = find_column(
            customer_df, ["Monetary", "TotalSpend", "TotalRevenue",
                          "CustomerRevenue"]
        )

        unique_customers = (
            customer_df[customer_id_col].nunique()
            if customer_id_col else len(customer_df)
        )

        show_metric_row([
            ("Rows", format_integer(len(customer_df))),
            ("Unique Customers", format_integer(unique_customers)),
            ("Segments / Clusters",
             format_integer(customer_df[segment_col].nunique())
             if segment_col else "N/A"),
            ("Average Monetary Value",
             format_number(
                 pd.to_numeric(
                     customer_df[monetary_col], errors="coerce"
                 ).mean()
             ) if monetary_col else "N/A"),
        ])

        left, right = st.columns(2)

        with left:
            if segment_col:
                segment_counts = (
                    customer_df[segment_col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                )
                st.subheader("Customer Distribution")
                st.bar_chart(segment_counts)
            else:
                st.info(
                    "No segment or cluster column was detected "
                    "in this dataset."
                )

        with right:
            if recency_col and frequency_col:
                scatter_df = customer_df[
                    [recency_col, frequency_col]
                ].copy()
                scatter_df[recency_col] = pd.to_numeric(
                    scatter_df[recency_col], errors="coerce"
                )
                scatter_df[frequency_col] = pd.to_numeric(
                    scatter_df[frequency_col], errors="coerce"
                )
                scatter_df = scatter_df.dropna()

                st.subheader("Recency vs Frequency")
                if not scatter_df.empty:
                    st.scatter_chart(
                        scatter_df.head(3000),
                        x=recency_col,
                        y=frequency_col,
                    )
                else:
                    st.info("No valid numeric RFM data to plot.")
            else:
                st.info(
                    "Recency and Frequency columns are not both present."
                )

        st.markdown("---")
        show_table(customer_df, "Customer Dataset Preview", 100)
        show_download(
            customer_df,
            customer_choice,
            "customer_dataset_download",
        )
    else:
        show_missing_notice(None, customer_choice)


# ============================================================
# TAB 4: CHURN ANALYTICS
# ============================================================

with tab_churn:
    st.header("Customer Churn Analysis")

    churn_predictions = get_file("churn_predictions")
    churn_comparison = get_file("churn_comparison")
    churn_importance = get_file("churn_importance")

    if churn_predictions is not None:
        prediction_col = find_column(
            churn_predictions,
            ["ChurnPrediction", "PredictedChurn", "Churn",
             "ChurnLabel", "Prediction"],
        )
        probability_col = find_column(
            churn_predictions,
            ["ChurnProbability", "Probability", "ChurnProb",
             "PredictedProbability", "Churn_Probability"],
        )
        customer_col = find_customer_column(churn_predictions)

        total_rows = len(churn_predictions)
        predicted_churners = None
        churn_rate = None

        if prediction_col:
            values = churn_predictions[prediction_col]
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                predicted_churners = int(
                    numeric.fillna(0).eq(1).sum()
                )
            else:
                normalized = values.astype(str).str.lower().str.strip()
                predicted_churners = int(
                    normalized.isin(
                        ["1", "yes", "true", "churn", "churned"]
                    ).sum()
                )

            churn_rate = (
                predicted_churners / total_rows * 100
                if total_rows else None
            )

        show_metric_row([
            ("Prediction Rows", format_integer(total_rows)),
            ("Unique Customers",
             format_integer(churn_predictions[customer_col].nunique())
             if customer_col else "N/A"),
            ("Predicted Churners",
             format_integer(predicted_churners)),
            ("Predicted Churn Rate",
             f"{churn_rate:.2f}%" if churn_rate is not None else "N/A"),
        ])

        if probability_col:
            churn_predictions[probability_col] = pd.to_numeric(
                churn_predictions[probability_col], errors="coerce"
            )
            probability_values = churn_predictions[
                probability_col
            ].dropna()

            st.subheader("Churn Probability Distribution")
            if not probability_values.empty:
                st.bar_chart(
                    probability_values.clip(0, 1).value_counts(
                        bins=10, sort=False
                    ).sort_index()
                )
            else:
                st.info("No valid churn probabilities are available.")
        else:
            st.info(
                "No churn probability column was detected. "
                "The predictions table is still available below."
            )

        show_table(
            churn_predictions, "Customer Churn Predictions", 100
        )
        show_download(
            churn_predictions,
            FILES["churn_predictions"],
            "churn_predictions_download",
        )
    else:
        show_missing_notice(
            None, FILES["churn_predictions"]
        )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        if churn_comparison is not None:
            show_table(
                churn_comparison, "Churn Model Comparison", 100
            )
            show_download(
                churn_comparison,
                FILES["churn_comparison"],
                "churn_comparison_download",
            )
        else:
            st.info("Churn model comparison file is unavailable.")

    with right:
        if churn_importance is not None:
            importance_col = find_column(
                churn_importance,
                ["Feature", "FeatureName", "Variable", "Column"],
            )
            value_col = find_column(
                churn_importance,
                ["Importance", "FeatureImportance", "Score",
                 "ImportanceScore"],
            )

            if importance_col and value_col:
                importance_plot = churn_importance.copy()
                importance_plot[value_col] = pd.to_numeric(
                    importance_plot[value_col], errors="coerce"
                )
                importance_plot = (
                    importance_plot.dropna(subset=[value_col])
                    .sort_values(value_col, ascending=False)
                    .head(15)
                )
                st.subheader("Important Churn Features")
                if not importance_plot.empty:
                    st.bar_chart(
                        importance_plot.set_index(importance_col)[value_col]
                    )
            else:
                st.info(
                    "Feature importance columns could not be identified."
                )

            show_table(churn_importance, "Feature Importance Data", 100)
            show_download(
                churn_importance,
                FILES["churn_importance"],
                "churn_importance_download",
            )
        else:
            st.info("Churn feature importance file is unavailable.")


# ============================================================
# TAB 5: DEMAND FORECASTING
# ============================================================

with tab_forecasting:
    st.header("Demand and Revenue Forecasting")

    forecast_choice = st.selectbox(
        "Choose forecast dataset",
        [
            FILES["monthly_forecast"],
            FILES["forecast_90_days"],
            FILES["lstm_forecast"],
            FILES["forecast_evaluation"],
            FILES["lstm_evaluation"],
            FILES["lstm_test"],
            FILES["daily_features"],
            FILES["monthly_features"],
            FILES["product_demand"],
        ],
        key="forecast_dataset_select",
    )

    forecast_df = get_dataset(forecast_choice)

    if forecast_df is not None:
        date_col = find_date_column(forecast_df)
        revenue_col = find_revenue_column(forecast_df)
        demand_col = find_column(
            forecast_df,
            ["Quantity", "Demand", "PredictedDemand",
             "ForecastDemand", "yhat", "Forecast"],
        )

        show_metric_row([
            ("Rows", format_integer(len(forecast_df))),
            ("Columns", format_integer(len(forecast_df.columns))),
            ("Forecast / Revenue Total",
             format_number(
                 pd.to_numeric(
                     forecast_df[revenue_col], errors="coerce"
                 ).sum()
             ) if revenue_col else "N/A"),
            ("Forecast / Demand Average",
             format_number(
                 pd.to_numeric(
                     forecast_df[demand_col], errors="coerce"
                 ).mean()
             ) if demand_col else "N/A"),
        ])

        st.markdown("---")

        if revenue_col:
            show_chart(
                forecast_df,
                [date_col] if date_col else ["Date", "Month", "ds"],
                [revenue_col],
                "line",
                "Revenue Forecast / Trend",
            )
        elif demand_col:
            show_chart(
                forecast_df,
                [date_col] if date_col else ["Date", "Month", "ds"],
                [demand_col],
                "line",
                "Demand Forecast / Trend",
            )
        else:
            st.info(
                "No recognized forecast value column was found. "
                "Inspect the dataset below to see its columns."
            )

        st.markdown("---")
        show_table(forecast_df, "Forecast Dataset Preview", 100)
        show_download(
            forecast_df,
            forecast_choice,
            "forecast_dataset_download",
        )
    else:
        show_missing_notice(None, forecast_choice)

    st.markdown("---")
    st.subheader("Forecast Model Evaluation")

    eval_left, eval_right = st.columns(2)

    with eval_left:
        evaluation = get_file("forecast_evaluation")
        if evaluation is not None:
            show_table(evaluation, "Standard Forecast Evaluation", 100)
            show_download(
                evaluation,
                FILES["forecast_evaluation"],
                "forecast_eval_download",
            )
        else:
            st.info("Standard forecast evaluation is unavailable.")

    with eval_right:
        lstm_evaluation = get_file("lstm_evaluation")
        if lstm_evaluation is not None:
            show_table(lstm_evaluation, "LSTM Evaluation", 100)
            show_download(
                lstm_evaluation,
                FILES["lstm_evaluation"],
                "lstm_eval_download",
            )
        else:
            st.info("LSTM evaluation is unavailable.")


# ============================================================
# TAB 6: INVENTORY OPTIMIZATION
# ============================================================

with tab_inventory:
    st.header("Inventory Optimization")

    inventory_choice = st.selectbox(
        "Choose inventory dataset",
        [
            FILES["inventory"],
            FILES["inventory_priority"],
            FILES["inventory_risk"],
            FILES["abc_products"],
            FILES["product_features"],
        ],
        key="inventory_dataset_select",
    )

    inventory_df = get_dataset(inventory_choice)

    if inventory_df is not None:
        product_col = find_column(
            inventory_df, ["StockCode", "ProductID", "Product", "SKU"]
        )
        abc_col = find_column(
            inventory_df, ["ABC_Class", "ABCClass", "ABC Category"]
        )
        risk_col = find_column(
            inventory_df,
            ["RiskCategory", "InventoryRisk", "RiskLevel",
             "Risk Category"],
        )
        reorder_col = find_column(
            inventory_df,
            ["ReorderPoint", "Reorder_Point", "Reorder Level"],
        )
        order_qty_col = find_column(
            inventory_df,
            ["RecommendedOrderQty", "RecommendedOrderQuantity",
             "OrderQuantity", "OrderQty", "SuggestedOrderQty"],
        )

        show_metric_row([
            ("Rows", format_integer(len(inventory_df))),
            ("Unique Products",
             format_integer(inventory_df[product_col].nunique())
             if product_col else "N/A"),
            ("ABC Classes",
             format_integer(inventory_df[abc_col].nunique())
             if abc_col else "N/A"),
            ("Average Reorder Point",
             format_number(
                 pd.to_numeric(
                     inventory_df[reorder_col], errors="coerce"
                 ).mean()
             ) if reorder_col else "N/A"),
        ])

        left, right = st.columns(2)

        with left:
            if abc_col:
                st.subheader("ABC Product Distribution")
                st.bar_chart(
                    inventory_df[abc_col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                )
            else:
                st.info(
                    "No ABC class column was detected in this dataset."
                )

        with right:
            if risk_col:
                st.subheader("Inventory Risk Distribution")
                st.bar_chart(
                    inventory_df[risk_col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                )
            else:
                st.info(
                    "No inventory risk category column was detected."
                )

        if order_qty_col:
            st.markdown("---")
            st.subheader("Recommended Order Quantity")
            order_df = inventory_df.copy()
            order_df[order_qty_col] = pd.to_numeric(
                order_df[order_qty_col], errors="coerce"
            )
            order_df = (
                order_df.dropna(subset=[order_qty_col])
                .sort_values(order_qty_col, ascending=False)
                .head(20)
            )

            if product_col and not order_df.empty:
                st.bar_chart(
                    order_df.set_index(product_col)[order_qty_col]
                )

        st.markdown("---")
        show_table(inventory_df, "Inventory Dataset Preview", 100)
        show_download(
            inventory_df,
            inventory_choice,
            "inventory_dataset_download",
        )
    else:
        show_missing_notice(None, inventory_choice)

    st.markdown("---")
    st.subheader("Inventory Priority Products")

    priority_df = get_file("inventory_priority")
    if priority_df is not None:
        show_table(priority_df, max_rows=100)
        show_download(
            priority_df,
            FILES["inventory_priority"],
            "inventory_priority_download",
        )
    else:
        st.info("Inventory priority dataset is unavailable.")


# ============================================================
# TAB 7: DATA HEALTH AND ALL CSV EXPLORER
# ============================================================

with tab_health:
    st.header("Data Health & CSV Explorer")
    st.write(
        "This section automatically discovers every CSV file in "
        "`data/processed/`. It is not limited to the files configured "
        "for the individual analytics tabs."
    )

    if not DATA_DIR.exists():
        st.error(f"Folder not found: {DATA_DIR}")
    else:
        csv_paths = list_csv_files()

        show_metric_row([
            ("CSV Files", format_integer(len(csv_paths))),
            ("Data Folder Exists", "Yes"),
            ("Project Folder", PROJECT_DIR.name),
        ])

        st.markdown("---")
        st.subheader("All Processed CSV Files")

        if csv_paths:
            file_inventory = inspect_csv_files()
            st.dataframe(
                file_inventory,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download CSV inventory report",
                data=file_inventory.to_csv(index=False).encode("utf-8"),
                file_name="retailpulse_csv_inventory.csv",
                mime="text/csv",
                key="csv_inventory_report_download",
            )
        else:
            st.warning("No CSV files were found in data/processed/.")

        st.markdown("---")
        st.subheader("Explore Any CSV")

        if csv_paths:
            selected_path = st.selectbox(
                "Select any processed CSV file",
                csv_paths,
                format_func=lambda path: path.name,
                key="all_csv_explorer_select",
            )

            try:
                selected_df = get_dataset(selected_path.name)

                if selected_df is not None:
                    show_metric_row([
                        ("Rows", format_integer(len(selected_df))),
                        ("Columns",
                         format_integer(len(selected_df.columns))),
                        ("Missing Cells",
                         format_integer(
                             int(selected_df.isna().sum().sum())
                         )),
                        ("Duplicate Rows",
                         format_integer(
                             int(selected_df.duplicated().sum())
                         )),
                    ])

                    st.subheader("Column Names")
                    st.code(
                        ", ".join(
                            str(column)
                            for column in selected_df.columns
                        ),
                        language=None,
                    )

                    st.subheader("Column Data Types")
                    dtype_df = pd.DataFrame({
                        "Column": selected_df.columns.astype(str),
                        "Data Type": [
                            str(dtype)
                            for dtype in selected_df.dtypes
                        ],
                        "Missing Values": [
                            int(selected_df[column].isna().sum())
                            for column in selected_df.columns
                        ],
                        "Missing Percentage": [
                            round(
                                selected_df[column].isna().mean() * 100,
                                2,
                            )
                            for column in selected_df.columns
                        ],
                    })

                    st.dataframe(
                        dtype_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    preview_rows = st.slider(
                        "Preview row count",
                        min_value=10,
                        max_value=500,
                        value=100,
                        step=10,
                        key="csv_preview_rows",
                    )

                    st.dataframe(
                        selected_df.head(preview_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download selected CSV",
                        data=selected_path.read_bytes(),
                        file_name=selected_path.name,
                        mime="text/csv",
                        key="all_csv_selected_download",
                    )

            except Exception as exc:
                st.error(
                    f"Could not inspect {selected_path.name}: {exc}"
                )
        else:
            st.info("Add CSV files to data/processed/ to explore them.")

        st.markdown("---")
        st.subheader("Configured Dashboard Datasets")

        configured_rows = []

        for dataset_key, filename in FILES.items():
            path = DATA_DIR / filename
            configured_rows.append({
                "Dashboard Dataset": dataset_key.replace("_", " ").title(),
                "File": filename,
                "Status": "Available" if path.exists() else "Missing",
            })

        st.dataframe(
            pd.DataFrame(configured_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The CSV Explorer discovers every CSV file automatically. "
            "The dedicated analytics charts use the configured datasets "
            "listed above."
        )


# ============================================================
# 6. FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "RetailPulse | Customer Analytics â€¢ Churn Prediction â€¢ "
    "Demand Forecasting â€¢ Inventory Optimization"
)
