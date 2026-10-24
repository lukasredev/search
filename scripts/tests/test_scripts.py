"""The two data scripts: dataset normalization and demo-query selection."""
import io
import json
import math
import pathlib
import sys
import zipfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import fetch_dataset  # noqa: E402
import find_demo_queries  # noqa: E402


def beir_files():
    """A miniature BEIR dataset, as the files appear inside the zip."""
    corpus = [
        {"_id": "d1", "title": " Vitamin D ", "text": "Low levels.\n", "metadata": {}},
        {"_id": "d2", "title": None, "text": "Calcium."},
        {"_id": "d3", "text": "No title key at all."},
    ]
    queries = [
        {"_id": "q1", "text": "vitamin   d\tdeficiency"},
        {"_id": "q2", "text": "never judged"},
        {"_id": "q3", "text": "calcium"},
    ]
    qrels = "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\nq3\td2\t1\n"
    return {
        "corpus.jsonl": "".join(json.dumps(d) + "\n" for d in corpus).encode(),
        "queries.jsonl": "".join(json.dumps(q) + "\n" for q in queries).encode(),
        "qrels/test.tsv": qrels.encode(),
    }


def test_normalize_writes_the_three_files_the_indexer_reads(tmp_path):
    files = beir_files()
    fetch_dataset.normalize("test", lambda f: io.BytesIO(files[f]), tmp_path)

    corpus = [json.loads(l) for l in (tmp_path / "corpus.jsonl").read_text().splitlines()]
    assert corpus == [
        {"id": "d1", "title": "Vitamin D", "text": "Low levels."},
        {"id": "d2", "title": "", "text": "Calcium."},
        {"id": "d3", "title": "", "text": "No title key at all."},
    ]
    # Only judged queries survive, with their whitespace (tabs included) collapsed so
    # the TSV stays two columns.
    assert (tmp_path / "queries.tsv").read_text() == "q1\tvitamin d deficiency\nq3\tcalcium\n"
    # Grade-0 rows are kept here; the Java loader is what ignores them.
    assert (tmp_path / "qrels.tsv").read_text() == "q1\td1\t2\nq1\td2\t0\nq3\td2\t1\n"


def test_download_reads_a_beir_zip_with_its_top_level_folder(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "scifact.zip", "w") as z:
        for name, data in beir_files().items():
            z.writestr(f"scifact/{name}", data)
    # Already present, so fetch() must not touch the network.
    with fetch_dataset.download("scifact", "test", raw) as open_file:
        fetch_dataset.normalize("test", open_file, tmp_path / "out")
    assert (tmp_path / "out" / "queries.tsv").read_text().count("\n") == 2


def test_unknown_dataset_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fetch_dataset.py", "../../etc"])
    assert fetch_dataset.main() == 2
    assert "unknown dataset" in capsys.readouterr().err


def test_ndcg_matches_the_java_evaluator():
    # Same case as EvaluatorTest.ndcgUsesExponentialGainAndLogDiscount.
    rel = {"a": 2, "b": 1}
    dcg = 1 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert find_demo_queries.ndcg(["x", "b", "a"], rel) == pytest.approx(dcg / idcg)
    assert find_demo_queries.ndcg(["a", "b"], rel) == pytest.approx(1.0)
    assert find_demo_queries.ndcg(["x"], {}) == 0.0


def test_demo_query_search_end_to_end(tmp_path, monkeypatch):
    """Ranks queries by which stage wins, and keeps hand-pinned examples on regeneration."""
    root = tmp_path
    data = root / "data" / "tiny"
    data.mkdir(parents=True)
    (data / "queries.tsv").write_text("q1\tlexical\nq2\tsemantic\nq3\tboth\n")
    (data / "qrels.tsv").write_text("q1\tA\t1\nq2\tB\t1\nq3\tC\t1\nq3\tD\t1\n")

    rankings = {  # query -> stage -> ranked doc ids
        "lexical": {"bm25": ["A"], "dense": ["x"], "rrf": ["A"]},
        "semantic": {"bm25": ["x"], "dense": ["B"], "rrf": ["B"]},
        "both": {"bm25": ["C", "x"], "dense": ["D", "y"], "rrf": ["C", "D"]},
    }
    seen = []

    def fake_fetch(url):
        if url.endswith("/api/health"):
            return {"dataset": "tiny"}
        seen.append(url)
        q = url.rsplit("q=", 1)[1]
        stages = [{"name": n, "results": [{"docId": d} for d in ids]} for n, ids in rankings[q].items()]
        stages[2]["detail"] = {"overlap": 0}
        return {"stages": stages}

    dest = root / "data" / "index" / "tiny" / "demo_queries.json"
    dest.parent.mkdir(parents=True)
    pinned = {"text": "hand picked", "ladder": "1", "ndcg": {"bm25": 1, "dense": 0, "rrf": 1}, "qid": "p"}
    dest.write_text(json.dumps({"note": "keep me", "lexical-win": [pinned], "rerank-win": [{"text": "r"}]}))

    monkeypatch.setattr(find_demo_queries, "ROOT", root)
    monkeypatch.setattr(find_demo_queries, "fetch", fake_fetch)
    monkeypatch.setattr(sys, "argv", ["find_demo_queries.py", "--dataset", "tiny"])
    find_demo_queries.main()

    out = json.loads(dest.read_text())
    assert [r["qid"] for r in out["semantic-win"]] == ["q2"]
    assert out["lexical-win"][0] == pinned
    assert [r["qid"] for r in out["lexical-win"][1:]] == ["q1"]
    assert [r["qid"] for r in out["fusion-win"]] == ["q3"]
    assert out["rerank-win"] == [{"text": "r"}]
    assert out["note"] == "keep me"
    assert all("rerank=false" in u for u in seen)


def test_demo_query_search_refuses_a_server_on_another_dataset(tmp_path, monkeypatch):
    """Scoring one dataset's rankings against another's qrels would pick nonsense."""
    monkeypatch.setattr(find_demo_queries, "ROOT", tmp_path)
    monkeypatch.setattr(find_demo_queries, "fetch", lambda url: {"dataset": "quora"})
    monkeypatch.setattr(sys, "argv", ["find_demo_queries.py", "--dataset", "scifact"])
    with pytest.raises(SystemExit, match="serving 'quora'"):
        find_demo_queries.main()
