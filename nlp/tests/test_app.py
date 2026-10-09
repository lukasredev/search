"""The sidecar's HTTP contract, with the two transformer models replaced by fakes.

searchd's NlpClient depends on these response shapes field for field, so they are
pinned here; the models themselves are exercised by the live end-to-end check.
"""
import pathlib
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import app  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeEmbedder:
    dim = 4

    def encode(self, texts, normalize_embeddings=True, **_):
        assert normalize_embeddings, "COSINE in Lucene relies on unit vectors"
        v = np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeReranker:
    tokenizer = FakeTokenizer()

    def predict(self, pairs, **_):
        return np.array([float(len(set(q.split()) & set(d.split()))) for q, d in pairs])


@pytest.fixture()
def client(monkeypatch):
    # embedder()/reranker() only load a model when none is cached, so priming the cache
    # keeps the real weights out of the test. No `with`: that would run the lifespan,
    # which loads the real models in the background.
    monkeypatch.setattr(app, "_models", {"embed": FakeEmbedder(), "rerank": FakeReranker()})
    return TestClient(app.app)


def test_embed_returns_one_unit_vector_per_text(client):
    r = client.post("/embed", json={"texts": ["a", "abc"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == app.EMBED_MODEL
    assert body["dim"] == 4
    assert len(body["vectors"]) == 2
    for v in body["vectors"]:
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)
    assert body["ms"] >= 0


def test_rerank_scores_in_input_order(client):
    r = client.post("/rerank", json={"query": "vitamin d", "docs": ["nothing here", "vitamin d bones"]})
    assert r.status_code == 200
    body = r.json()
    assert [i["score"] for i in body["items"]] == [0.0, 2.0]
    assert [i["doc_tokens"] for i in body["items"]] == [2, 3]
    assert body["max_length"] == app.MAX_LEN
    assert body["model"] == app.RERANK_MODEL


def test_rerank_flags_documents_that_overflow_the_window(client):
    query = "one two three"
    budget = app.MAX_LEN - 3 - 3  # [CLS] q [SEP] d [SEP]
    fits, overflows = "w " * budget, "w " * (budget + 1)
    items = client.post("/rerank", json={"query": query, "docs": [fits, overflows]}).json()["items"]
    assert [i["truncated"] for i in items] == [False, True]
    assert items[1]["doc_tokens"] == budget + 1


@pytest.mark.parametrize("path,body", [
    ("/embed", {"texts": []}),
    ("/embed", {}),
    ("/embed", {"texts": "not a list"}),
    ("/embed", {"texts": ["x"] * (app.MAX_EMBED_TEXTS + 1)}),
    ("/rerank", {"query": "q", "docs": []}),
    ("/rerank", {"docs": ["d"]}),
    ("/rerank", {"query": "q", "docs": ["d"] * (app.MAX_RERANK_DOCS + 1)}),
])
def test_bad_requests_are_rejected_before_reaching_the_model(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_readiness_waits_for_the_models(client, monkeypatch):
    monkeypatch.setattr(app, "_ready", app.threading.Event())
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["ok"] is False
    app._ready.set()
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json() == {"ok": True, "loaded": ["embed", "rerank"]}


def test_liveness_and_health(client):
    assert client.get("/livez").json() == {"ok": True}
    health = client.get("/health").json()
    assert health["ok"] is True
    assert health["embed_model"] == app.EMBED_MODEL
    assert health["loaded"] == ["embed", "rerank"]


def test_warmup_runs_both_models(client):
    r = client.post("/warmup")
    assert r.status_code == 200 and r.json()["ok"] is True
