import os

import httpx
import pytest


def _endpoint() -> str:
    endpoint = os.getenv("MYENDPOINT")
    if not endpoint:
        pytest.skip("MYENDPOINT not set; skipping API integration tests.")
    return endpoint.rstrip("/")


def test_health() -> None:
    resp = httpx.get(f"{_endpoint()}/health", timeout=10)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_single_text() -> None:
    resp = httpx.post(f"{_endpoint()}/predict", json={"text": "STARBUCKS"}, timeout=30)
    assert resp.status_code == 200

    payload = resp.json()
    assert isinstance(payload["model"], str)
    assert isinstance(payload["predictions"], list)
    assert len(payload["predictions"]) == 1

    pred = payload["predictions"][0]
    assert isinstance(pred["label_id"], int)
    assert 0.0 <= float(pred["confidence"]) <= 1.0


def test_predict_batch() -> None:
    resp = httpx.post(f"{_endpoint()}/predict", json={"texts": ["STARBUCKS", "UBER"]}, timeout=30)
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_predict_requires_input() -> None:
    resp = httpx.post(f"{_endpoint()}/predict", json={}, timeout=10)
    assert resp.status_code == 422
