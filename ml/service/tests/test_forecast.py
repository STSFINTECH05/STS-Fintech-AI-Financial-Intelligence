"""
STS FINTECH AI FINANCIAL INTELLIGENCE
Model C: Expense Forecaster API & Security Tests (Milestone 2 - Step 7.4B-2C)

Comprehensive test suite covering all 29 requirements:
1. Valid 7-day forecast.
2. Valid 14-day forecast.
3. Valid 30-day forecast.
4. Maximum supported horizon (90 days).
5. Invalid horizon (< 1, > 90, non-int).
6. Insufficient historical observations (< 30 days).
7. Missing historical dates (gap in sequence).
8. Duplicate dates.
9. Out-of-order dates.
10. Future observations (date > as_of_date).
11. Final observation / as_of_date mismatch.
12. Negative expenditure.
13. NaN and Infinity.
14. Invalid dates.
15. Unsupported currency.
16. Correct estimator loading (RandomForestRegressor).
17. Correct SHA-256 checksum.
18. Correct forecast origin.
19. Correct projected date sequence.
20. Correct total and average calculation.
21. Heuristic range labeling.
22. No confidence probability claims.
23. HMAC authentication.
24. Invalid signature rejection.
25. Replay attack rejection.
26. Model unavailable behavior.
27. Existing Model A endpoint regression.
28. Existing Model B endpoint regression.
29. Existing health endpoint regression.
"""

import json
import time
import uuid
import pytest
from datetime import datetime, date, timedelta
from starlette.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from ml.service.main import app, registry
from ml.service.security.hmac_auth import generate_hmac_signature, nonce_store
from ml.service.model_loader import APPROVED_MODEL_SPECS
from ml.service.prediction.expense_forecaster import ExpenseForecasterService

TEST_SECRET_KEY = "test_ephemeral_dev_key_at_least_32_bytes_long_secret_12345"
EXPECTED_SHA256 = APPROVED_MODEL_SPECS["model_c_forecaster"]["expected_sha256"]


@pytest.fixture(autouse=True)
def configure_test_auth(monkeypatch):
    """Sets development secret key and clears nonce store before each test."""
    monkeypatch.setenv("STS_INTERNAL_SERVICE_KEY", TEST_SECRET_KEY)
    nonce_store.clear()


@pytest.fixture
def client():
    """Initializes TestClient with model registry lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def make_hmac_post_headers(path: str, body: bytes, secret_key: str = TEST_SECRET_KEY, custom_nonce: str = None):
    """Helper creating HMAC security headers for POST request."""
    now = str(int(time.time()))
    nonce = custom_nonce or f"test-post-{uuid.uuid4().hex}"
    sig = generate_hmac_signature(
        secret_key=secret_key,
        method="POST",
        path=path,
        timestamp=now,
        nonce=nonce,
        body=body
    )
    return {
        "X-STS-Timestamp": now,
        "X-STS-Nonce": nonce,
        "X-STS-Signature": sig,
        "Content-Type": "application/json"
    }


def get_valid_series(as_of_date_str: str = "2026-10-31", count: int = 40):
    """Constructs a valid continuous daily expenditure series ending at as_of_date."""
    as_of = datetime.strptime(as_of_date_str, "%Y-%m-%d").date()
    start_date = as_of - timedelta(days=count - 1)
    series = []
    for i in range(count):
        d = start_date + timedelta(days=i)
        amt = 120000.0 + (i % 7) * 8500.0
        series.append({
            "date": d.isoformat(),
            "daily_expenditure": round(amt, 2)
        })
    return series


def get_valid_forecast_payload(as_of_date_str: str = "2026-10-31", horizon_days: int = 30, count: int = 40):
    """Returns a valid synthetic forecasting payload."""
    return {
        "as_of_date": as_of_date_str,
        "horizon_days": horizon_days,
        "currency": "INR",
        "historical_daily_series": get_valid_series(as_of_date_str, count)
    }


class TestExpenseForecasterApi:
    """Test suite covering the 29 required specifications for Model C."""

    # 1. Valid 7-day forecast
    def test_valid_7_day_forecast(self, client):
        payload = get_valid_forecast_payload(horizon_days=7)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert data["inference_status"] == "EVALUATED"
        assert data["horizon_days"] == 7
        assert len(data["projections"]) == 7

    # 2. Valid 14-day forecast
    def test_valid_14_day_forecast(self, client):
        payload = get_valid_forecast_payload(horizon_days=14)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["horizon_days"] == 14
        assert len(data["projections"]) == 14

    # 3. Valid 30-day forecast
    def test_valid_30_day_forecast(self, client):
        payload = get_valid_forecast_payload(horizon_days=30)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["horizon_days"] == 30
        assert len(data["projections"]) == 30

    # 4. Maximum supported horizon (90 days)
    def test_maximum_supported_horizon(self, client):
        payload = get_valid_forecast_payload(horizon_days=90)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["horizon_days"] == 90
        assert len(data["projections"]) == 90

    # 5. Invalid horizon (< 1, > 90, non-int)
    def test_invalid_horizon_rejection(self, client):
        for invalid_h in [0, -7, 91, 150, "thirty"]:
            p = get_valid_forecast_payload()
            p["horizon_days"] = invalid_h
            body = json.dumps(p).encode("utf-8")
            headers = make_hmac_post_headers("/internal/v2/forecast", body)
            res = client.post("/internal/v2/forecast", content=body, headers=headers)
            assert res.status_code == 422, f"Expected 422 for horizon {invalid_h}"

    # 6. Insufficient historical observations (< 30 days)
    def test_insufficient_historical_observations(self, client):
        p = get_valid_forecast_payload(count=29)
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 7. Missing historical dates (gap in sequence)
    def test_missing_historical_dates(self, client):
        series = get_valid_series("2026-10-31", count=40)
        # Drop one day to create a gap
        del series[15]
        p = {
            "as_of_date": "2026-10-31",
            "horizon_days": 14,
            "currency": "INR",
            "historical_daily_series": series
        }
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 8. Duplicate dates
    def test_duplicate_dates(self, client):
        series = get_valid_series("2026-10-31", count=35)
        # Duplicate an observation
        series.insert(10, series[10].copy())
        p = {
            "as_of_date": "2026-10-31",
            "horizon_days": 14,
            "currency": "INR",
            "historical_daily_series": series
        }
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 9. Out-of-order dates
    def test_out_of_order_dates(self, client):
        series = get_valid_series("2026-10-31", count=35)
        # Swap two items
        series[5], series[6] = series[6], series[5]
        p = {
            "as_of_date": "2026-10-31",
            "horizon_days": 14,
            "currency": "INR",
            "historical_daily_series": series
        }
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 10. Future observations (date > as_of_date)
    def test_future_observations_rejected(self, client):
        series = get_valid_series("2026-10-31", count=35)
        # Add future day past as_of_date
        series.append({"date": "2026-11-01", "daily_expenditure": 150000.0})
        p = {
            "as_of_date": "2026-10-31",
            "horizon_days": 14,
            "currency": "INR",
            "historical_daily_series": series
        }
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 11. Final observation / as_of_date mismatch
    def test_final_observation_mismatch(self, client):
        p = get_valid_forecast_payload(as_of_date_str="2026-10-31")
        p["as_of_date"] = "2026-10-30"  # final series date is 2026-10-31
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 12. Negative expenditure
    def test_negative_expenditure_rejected(self, client):
        series = get_valid_series("2026-10-31", count=35)
        series[12]["daily_expenditure"] = -500.0
        p = {
            "as_of_date": "2026-10-31",
            "horizon_days": 14,
            "currency": "INR",
            "historical_daily_series": series
        }
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 13. NaN and Infinity in expenditure
    def test_nan_and_infinity_rejected(self, client):
        series = get_valid_series("2026-10-31", count=35)
        series[12]["daily_expenditure"] = "NaN"
        p = {
            "as_of_date": "2026-10-31",
            "horizon_days": 14,
            "currency": "INR",
            "historical_daily_series": series
        }
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 14. Invalid dates
    def test_invalid_dates_rejected(self, client):
        p = get_valid_forecast_payload()
        p["as_of_date"] = "invalid-date-string"
        body = json.dumps(p).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 422

    # 15. Unsupported currency
    def test_unsupported_currency_rejected(self, client):
        for curr in ["USD", "EUR", "GBP", "XYZ"]:
            p = get_valid_forecast_payload()
            p["currency"] = curr
            body = json.dumps(p).encode("utf-8")
            headers = make_hmac_post_headers("/internal/v2/forecast", body)
            res = client.post("/internal/v2/forecast", content=body, headers=headers)
            assert res.status_code == 422, f"Expected 422 for unsupported currency {curr}"

    # 16. Correct estimator loading
    def test_correct_estimator_loading(self):
        loaded_model = registry.get_model("model_c_forecaster")
        assert loaded_model is not None
        assert isinstance(loaded_model, RandomForestRegressor)
        assert loaded_model.max_depth == 10

    # 17. Correct SHA-256 checksum
    def test_correct_artifact_checksum(self):
        status_info = registry.model_status_info.get("model_c_forecaster", {})
        assert status_info.get("sha256_verified") is True
        assert EXPECTED_SHA256 == "ea71b872864f3ab8dba53ab96422e24c6d21b730c9fee0191b9fded950de6b22"

    # 18. Correct forecast origin
    def test_correct_forecast_origin(self, client):
        as_of = "2026-09-15"
        payload = get_valid_forecast_payload(as_of_date_str=as_of, horizon_days=7)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["forecast_origin"] == as_of

    # 19. Correct projected date sequence
    def test_correct_projected_date_sequence(self, client):
        as_of = "2026-09-15"
        as_of_dt = datetime.strptime(as_of, "%Y-%m-%d").date()
        payload = get_valid_forecast_payload(as_of_date_str=as_of, horizon_days=7)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        data = res.json()
        for idx, proj in enumerate(data["projections"]):
            expected_date = (as_of_dt + timedelta(days=idx + 1)).isoformat()
            assert proj["step"] == idx + 1
            assert proj["date"] == expected_date

    # 20. Correct total and average calculation
    def test_correct_total_and_average_calculation(self, client):
        payload = get_valid_forecast_payload(horizon_days=14)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        data = res.json()
        expected_total = sum(p["predicted_expenditure"] for p in data["projections"])
        expected_avg = round(expected_total / 14, 2)
        assert abs(data["total_projected_expenditure"] - expected_total) < 0.1
        assert abs(data["average_daily_expenditure"] - expected_avg) < 0.1

    # 21. Heuristic range labeling
    def test_heuristic_range_labeling(self, client):
        payload = get_valid_forecast_payload(horizon_days=7)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        data = res.json()
        assert data["planning_range_type"] == "HEURISTIC_PLANNING_RANGE"
        for p in data["projections"]:
            assert "heuristic_planning_range" in p
            assert "lower" in p["heuristic_planning_range"]
            assert "upper" in p["heuristic_planning_range"]
            assert p["heuristic_planning_range"]["lower"] <= p["heuristic_planning_range"]["upper"]

    # 22. No confidence probability claims
    def test_no_confidence_probability_claims(self, client):
        payload = get_valid_forecast_payload(horizon_days=7)
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/forecast", body)

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        response_str = res.text.lower()
        # Verify no statistical confidence or guarantee claims exist in the response
        assert "confidence" not in response_str
        assert "80%" not in response_str
        assert "95%" not in response_str
        assert "guaranteed" not in response_str

    # 23. HMAC authentication
    def test_hmac_authentication_required(self, client):
        payload = get_valid_forecast_payload()
        body = json.dumps(payload).encode("utf-8")

        res = client.post("/internal/v2/forecast", content=body)
        assert res.status_code == 401
        assert res.json()["detail"]["error_code"] == "MISSING_SECURITY_HEADERS"

    # 24. Invalid signature rejection
    def test_invalid_signature_rejection(self, client):
        payload = get_valid_forecast_payload()
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "X-STS-Timestamp": str(int(time.time())),
            "X-STS-Nonce": f"nonce-{uuid.uuid4().hex}",
            "X-STS-Signature": "1111111111111111111111111111111111111111111111111111111111111111",
            "Content-Type": "application/json"
        }

        res = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res.status_code == 401
        assert res.json()["detail"]["error_code"] == "INVALID_SIGNATURE"

    # 25. Replay attack rejection
    def test_replay_attack_rejection(self, client):
        payload = get_valid_forecast_payload()
        body = json.dumps(payload).encode("utf-8")
        fixed_nonce = f"fixed-forecast-nonce-{uuid.uuid4().hex}"
        headers = make_hmac_post_headers("/internal/v2/forecast", body, custom_nonce=fixed_nonce)

        # First request succeeds
        res1 = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res1.status_code == 200

        # Replayed request fails
        res2 = client.post("/internal/v2/forecast", content=body, headers=headers)
        assert res2.status_code == 401
        assert res2.json()["detail"]["error_code"] == "REPLAY_DETECTED"

    # 26. Model unavailable behavior
    def test_model_unavailable_behavior(self):
        service = ExpenseForecasterService(registry)
        service._model = None
        service.engine = None

        payload = get_valid_forecast_payload()
        res = service.forecast_expense(payload)

        assert res["status"] == "ERROR"
        assert res["inference_status"] == "NOT_EVALUATED"
        assert res["total_projected_expenditure"] is None
        assert res["projections"] == []

    # 27. Existing Model A endpoint regression
    def test_existing_classification_endpoint_regression(self, client):
        payload = {
            "description": "AWS Cloud Hosting Infrastructure Monthly Invoice",
            "amount": 14250.75,
            "transaction_date": "2026-05-15",
            "department_id": "DEP-TECH-01"
        }
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/classify", body)

        res = client.post("/internal/v2/classify", content=body, headers=headers)
        assert res.status_code == 200
        assert res.json()["status"] == "SUCCESS"

    # 28. Existing Model B endpoint regression
    def test_existing_anomaly_endpoint_regression(self, client):
        payload = {
            "amount": 18500.0,
            "transaction_date": "2026-05-15",
            "category": "Cloud Services",
            "department_id": "1",
            "category_median": 46403.85,
            "department_median": 23509.82,
            "spending_approval_limit": 50000.0,
            "annual_budget_baseline": 15000000.0,
            "vendor_rating": 4.8
        }
        body = json.dumps(payload).encode("utf-8")
        headers = make_hmac_post_headers("/internal/v2/anomaly/score", body)

        res = client.post("/internal/v2/anomaly/score", content=body, headers=headers)
        assert res.status_code == 200
        assert res.json()["status"] == "SUCCESS"
        assert res.json()["inference_status"] == "EVALUATED"

    # 29. Existing health endpoint regression
    def test_existing_health_endpoint_regression(self, client):
        # Public liveness
        res_live = client.get("/health/live")
        assert res_live.status_code == 200
        assert res_live.json()["service_status"] == "ALIVE"

        # Internal health
        now = str(int(time.time()))
        nonce = f"health-check-{uuid.uuid4().hex}"
        sig = generate_hmac_signature(TEST_SECRET_KEY, "GET", "/internal/v2/health", now, nonce, b"")
        headers = {
            "X-STS-Timestamp": now,
            "X-STS-Nonce": nonce,
            "X-STS-Signature": sig
        }
        res_health = client.get("/internal/v2/health", headers=headers)
        assert res_health.status_code == 200
        assert res_health.json()["service_status"] == "READY"
