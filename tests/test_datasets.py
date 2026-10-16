
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def dataset_names(client):
    response = client.get("/datasets")
    assert response.status_code == 200

    data = response.json()
    return [item["name"] for item in data["datasets"]]


def test_dataset_list_has_25_entries(client):
    response = client.get("/datasets")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 25
    assert len(data["datasets"]) == 25


@pytest.mark.parametrize(
    "dataset_name",
    [
        "abc_product_analysis",
        "churn_feature_importance",
        "churn_model_comparison",
        "cluster_summary",
        "customer_churn_predictions",
        "customer_clusters",
        "customer_features",
        "daily_sales_features",
        "feature_engineered_sales",
        "forecast_model_evaluation",
        "inventory_optimization",
        "inventory_priority_products",
        "inventory_risk_summary",
        "lstm_forecast_evaluation",
        "lstm_revenue_forecast_90_days",
        "lstm_test_predictions",
        "monthly_revenue_forecast",
        "monthly_sales_features",
        "product_daily_demand",
        "product_features",
        "retail_clean",
        "retail_sales",
        "revenue_forecast_90_days",
        "rfm_customer_segments",
        "rfm_segment_summary",
    ],
)
def test_dataset_preview_endpoint(client, dataset_name):
    response = client.get(f"/datasets/{dataset_name}")

    assert response.status_code == 200, (
        f"Preview failed for {dataset_name}: {response.text}"
    )

    data = response.json()
    assert isinstance(data, dict)

    # Support common API response formats.
    assert data, f"Empty response for {dataset_name}"


def test_unknown_dataset_returns_client_error(client):
    response = client.get("/datasets/this_dataset_does_not_exist")

    assert response.status_code in (400, 404, 422)


def test_dataset_names_are_safe(client, dataset_names):
    for name in dataset_names:
        assert "/" not in name
        assert "\\" not in name
        assert ".." not in name
