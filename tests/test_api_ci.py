
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Run the API against isolated, temporary CSV fixtures."""
    from api import main

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    sales = pd.DataFrame({
        "InvoiceNo": ["1001", "1001", "1002"],
        "StockCode": ["A1", "A2", "A1"],
        "CustomerID": [101, 101, 102],
        "Quantity": [2, 3, 1],
        "Revenue": [20.0, 30.0, 10.0],
    })
    sales.to_csv(
        processed_dir / "feature_engineered_sales.csv",
        index=False,
    )

    forecast = pd.DataFrame({
        "Date": ["2026-10-17", "2026-10-18"],
        "ForecastRevenue": [100.0, 110.0],
    })
    forecast.to_csv(
        processed_dir / "monthly_revenue_forecast.csv",
        index=False,
    )

    inventory = pd.DataFrame({
        "StockCode": ["A1", "A2"],
        "Priority": ["High", "Medium"],
    })
    inventory.to_csv(
        processed_dir / "inventory_priority_products.csv",
        index=False,
    )

    churn = pd.DataFrame({
        "CustomerID": [101],
        "ChurnPrediction": [0],
    })
    churn.to_csv(
        processed_dir / "customer_churn_predictions.csv",
        index=False,
    )

    rfm = pd.DataFrame({
        "CustomerID": [101],
        "Segment": ["Loyal"],
    })
    rfm.to_csv(
        processed_dir / "rfm_customer_segments.csv",
        index=False,
    )

    abc = pd.DataFrame({
        "StockCode": ["A1"],
        "ABC_Class": ["A"],
    })
    abc.to_csv(
        processed_dir / "abc_product_analysis.csv",
        index=False,
    )

    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(main, "DATA_DIR", processed_dir)

    with TestClient(main.app) as test_client:
        yield test_client


def test_ci_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "RetailPulse"


def test_ci_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["available_csv_count"] == 6


def test_ci_datasets_endpoint(client):
    response = client.get("/datasets")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 6

    names = {item["name"] for item in data["datasets"]}
    assert "feature_engineered_sales" in names


def test_ci_dataset_preview(client):
    response = client.get("/datasets/feature_engineered_sales?limit=2")

    assert response.status_code == 200
    data = response.json()

    assert data["rows_in_file"] == 3
    assert data["preview_limit"] == 2
    assert len(data["data"]) == 2


def test_ci_dataset_not_found(client):
    response = client.get("/datasets/does_not_exist")

    assert response.status_code == 404


def test_ci_dataset_preview_limit_validation(client):
    assert client.get("/datasets/feature_engineered_sales?limit=0").status_code == 422
    assert client.get("/datasets/feature_engineered_sales?limit=501").status_code == 422


def test_ci_analytics_summary(client):
    response = client.get("/analytics/summary")

    assert response.status_code == 200
    data = response.json()

    assert data["dataset_used"] == "feature_engineered_sales"
    assert data["row_count"] == 3
    assert data["column_count"] == 5
    assert data["total_revenue"] == pytest.approx(60.0)
    assert data["total_quantity"] == 6
    assert data["unique_customers"] == 2
    assert data["unique_invoices"] == 2
    assert data["unique_products"] == 2



@pytest.mark.parametrize(
    ("endpoint", "expected_key", "expected_rows"),
    [
        ("/forecast/revenue?limit=1", "forecast", 2),
        ("/inventory/priority?limit=1", "products", 2),
        ("/analytics/churn?limit=1", "predictions", 1),
        ("/analytics/rfm?limit=1", "segments", 1),
        ("/analytics/abc-products?limit=1", "products", 1),
    ],
)
def test_ci_analysis_endpoints(
    client, endpoint, expected_key, expected_rows
):
    response = client.get(endpoint)

    assert response.status_code == 200
    data = response.json()

    # Total rows in the underlying CSV
    assert data["rows_available"] == expected_rows

    # The limit=1 parameter returns only one record
    assert len(data[expected_key]) == 1



def test_ci_analysis_endpoint_limit_validation(client):
    assert client.get("/forecast/revenue?limit=0").status_code == 422
    assert client.get("/inventory/priority?limit=501").status_code == 422
    assert client.get("/analytics/churn?limit=0").status_code == 422
    assert client.get("/analytics/rfm?limit=501").status_code == 422
    assert client.get("/analytics/abc-products?limit=0").status_code == 422
