
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    """Create a test client without starting a separate server."""
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


# Existing project-integrity tests

def test_project_directories_exist():
    assert (PROJECT_DIR / "api").is_dir()
    assert (PROJECT_DIR / "dashboard").is_dir()
    assert (PROJECT_DIR / "data").is_dir()
    assert (PROJECT_DIR / "tests").is_dir()


def test_dashboard_file_exists():
    assert (PROJECT_DIR / "dashboard" / "app.py").is_file()


def test_processed_data_directory_exists():
    assert (PROJECT_DIR / "data" / "processed").is_dir()


def test_feature_engineered_dataset_exists():
    dataset = (
        PROJECT_DIR
        / "data"
        / "processed"
        / "feature_engineered_sales.csv"
    )
    assert dataset.is_file()
    assert dataset.stat().st_size > 0


# API endpoint tests

def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert data["project"] == "RetailPulse"
    assert data["message"] == "RetailPulse API is running."
    assert data["version"] == "1.0.0"
    assert data["documentation"] == "/docs"
    assert data["health_check"] == "/health"
    assert data["available_datasets"] == "/datasets"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["project_directory_exists"] is True
    assert data["processed_data_directory_exists"] is True
    assert data["available_csv_count"] == 25


def test_datasets_endpoint(client):
    response = client.get("/datasets")

    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 25
    assert len(data["datasets"]) == 25

    names = {item["name"] for item in data["datasets"]}

    assert "feature_engineered_sales" in names
    assert "customer_churn_predictions" in names
    assert "monthly_revenue_forecast" in names

    for item in data["datasets"]:
        assert item["filename"].endswith(".csv")
        assert item["size_bytes"] > 0
        assert item["preview_endpoint"].startswith("/datasets/")


def test_analytics_summary_endpoint(client):
    response = client.get("/analytics/summary")

    assert response.status_code == 200
    data = response.json()

    assert data["dataset_used"] == "feature_engineered_sales"
    assert data["row_count"] == 779425
    assert data["column_count"] == 48
    assert data["total_revenue"] == pytest.approx(
        17374804.268, abs=0.01
    )
    assert data["total_quantity"] == 10513952
    assert data["unique_customers"] == 5878
    assert data["unique_invoices"] == 36969
    assert data["unique_products"] == 4631
