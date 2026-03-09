"""
Tests for the Distributed AI Inference Engine backend.
Run: pytest backend/tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../api"))

import pytest
from app import app
from model import load_model, predict
from mpi_dispatcher import dispatch_inference


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Model tests ─────────────────────────────────────────────────────────────

def test_model_loads():
    model, _ = load_model()
    assert model is not None


def test_predict_positive():
    model, vectorizer = load_model()
    result = predict(model, vectorizer, "This is excellent and works great")
    assert result["label"] in ("positive", "negative")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_negative():
    model, vectorizer = load_model()
    result = predict(model, vectorizer, "This is terrible and completely broken")
    assert result["label"] == "negative"


def test_predict_returns_scores():
    model, vectorizer = load_model()
    result = predict(model, vectorizer, "test input")
    assert "scores" in result
    assert "positive" in result["scores"]
    assert "negative" in result["scores"]


# ── API tests ────────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"


def test_infer_endpoint(client):
    res = client.post("/api/infer",
                      json={"text": "This product is amazing"},
                      content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert "label" in data
    assert "confidence" in data
    assert "latency_ms" in data


def test_infer_missing_text(client):
    res = client.post("/api/infer", json={}, content_type="application/json")
    assert res.status_code == 400


def test_infer_text_too_long(client):
    res = client.post("/api/infer",
                      json={"text": "x" * 2001},
                      content_type="application/json")
    assert res.status_code == 400


def test_batch_endpoint(client):
    texts = ["Great product", "Terrible experience", "Works well"]
    res = client.post("/api/infer/batch",
                      json={"texts": texts},
                      content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["results"]) == 3
    assert data["workers_used"] >= 1


def test_batch_empty(client):
    res = client.post("/api/infer/batch",
                      json={"texts": []},
                      content_type="application/json")
    assert res.status_code == 400


# ── MPI dispatcher tests ─────────────────────────────────────────────────────

def test_dispatch_single():
    results = dispatch_inference(["Hello world"])
    assert len(results) == 1
    assert "label" in results[0]
    assert "worker" in results[0]


def test_dispatch_multi():
    texts = [f"Sample text number {i}" for i in range(8)]
    results = dispatch_inference(texts)
    assert len(results) == 8
    workers = set(r["worker"] for r in results)
    assert len(workers) > 1  # confirmed work was distributed
