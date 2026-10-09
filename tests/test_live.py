"""End-to-end checks against a running stack: real index, real models, real HTTP.

The unit suites replace the sidecar with fakes; this is what proves the two halves
actually agree. Skipped unless SEARCHD_URL is set:

    SEARCHD_URL=http://localhost:8080 nlp/.venv/bin/python -m pytest tests

Set AUTH_USER and AUTH_PASS too if the service requires them. Stdlib only.
"""
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pytest

BASE = os.environ.get("SEARCHD_URL", "").rstrip("/")
USER, PASS = os.environ.get("AUTH_USER"), os.environ.get("AUTH_PASS")
AUTH = ({"Authorization": "Basic " + base64.b64encode(f"{USER}:{PASS}".encode()).decode()}
        if USER and PASS else {})

pytestmark = pytest.mark.skipif(not BASE, reason="SEARCHD_URL not set")


def call(path, body=None, auth=True, method=None):
    headers = dict(AUTH) if auth else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def get_json(path):
    status, _, body = call(path)
    assert status == 200, (path, status, body[:300])
    return json.loads(body)


def search(q, **params):
    return get_json("/api/search?" + urllib.parse.urlencode({"q": q, **params}))


@pytest.fixture(scope="module")
def judged():
    """A judged query from the served dataset, so eval lights up."""
    return get_json("/api/queries?limit=1")["queries"][0]


def test_both_halves_are_up():
    h = get_json("/api/health")
    assert h["ok"] and h["docs"] > 0
    assert h["nlp"]["ok"], "the nlp sidecar is down; the rest of this suite would run degraded"
    assert call("/readyz", auth=False)[0] == 200
    assert call("/livez", auth=False)[0] == 200


def test_full_pipeline_on_a_judged_query(judged):
    d = search(judged["text"])
    assert "warnings" not in d, d.get("warnings")
    assert [s["name"] for s in d["stages"]] == ["analyze", "bm25", "dense", "rrf", "rerank"]
    assert d["qid"] == judged["qid"]
    assert d["eval"]["judged"]
    assert set(d["eval"]["perStage"]) == {"bm25", "dense", "rrf", "rerank"}
    stages = {s["name"]: s for s in d["stages"]}
    assert stages["dense"]["params"]["dim"] == 384
    assert stages["rerank"]["params"]["model"].startswith("cross-encoder/")
    # Every ranked doc is described once in the docs map.
    for s in ("bm25", "dense", "rrf", "rerank"):
        assert all(r["docId"] in d["docs"] for r in stages[s]["results"])


def test_bm25_explanation_is_complete_and_adds_up():
    d = search("grep recursive files", mode="bm25_only", rerank="false", bm25_n=3)
    for r in d["stages"][1]["results"]:
        ex = r["explain"]
        for t in ex["terms"]:
            for key in ("df", "numDocs", "tf", "dl", "avgdl", "idf", "contrib"):
                assert key in t, f"{key} missing from {t}"
        assert sum(t["contrib"] for t in ex["terms"]) == pytest.approx(r["score"], abs=1e-3)


def test_the_readme_example_loses_its_symbols_in_the_analyzer():
    d = search("What's the difference between $@ and $*", rerank="false")
    analyzed = d["stages"][0]["detail"]["chain"][-1]["tokens"]
    assert "$@" not in analyzed and "$*" not in analyzed
    assert d["stages"][2]["name"] == "dense" and d["stages"][2]["results"]


def test_dense_vectors_match_what_the_index_stored():
    d = search("how do I list open ports", mode="dense_only", rerank="false", dense_n=1)
    top = d["stages"][1]["results"][0]
    doc = get_json(f"/api/doc/{urllib.parse.quote(top['docId'])}")
    assert len(doc["vector"]) == 384
    norm = sum(v * v for v in doc["vector"]) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_index_preview_embeds_with_the_real_model():
    status, _, body = call("/api/index-preview", {"text": "Grep searches files for patterns."})
    assert status == 200
    p = json.loads(body)
    assert p["vector"]["dim"] == 384
    assert p["summary"]["uniqueTerms"] == 4


def test_inspection_endpoints():
    assert get_json("/api/term?t=directories")["analyzedTo"] == "directori"
    assert get_json("/api/analyze?text=Running")["tokens"][0]["term"] == "run"
    assert get_json("/api/stats")["docs"] > 0
    assert call("/api/doc/does-not-exist")[0] == 404


def test_bad_input_is_a_client_error():
    assert call("/api/search?q=x&mode=nope")[0] == 400
    assert call("/api/index-preview", method="POST")[0] == 400


def test_security_headers():
    _, headers, _ = call("/api/health")
    assert headers.get("X-Content-Type-Options") == "nosniff"


@pytest.mark.skipif(not AUTH, reason="service runs without auth")
def test_auth_is_enforced_everywhere_but_the_probes():
    for path in ("/", "/api/health", "/api/search?q=x", "/api/doc/1"):
        assert call(path, auth=False)[0] == 401, path
    assert call("/readyz", auth=False)[0] == 200
