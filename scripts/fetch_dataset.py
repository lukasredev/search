#!/usr/bin/env python3
"""Fetch a BEIR dataset and normalize it to the three files the Java indexer reads.

Deliberately stdlib-only: the indexer must never depend on a Python environment,
and students should be able to read this top to bottom.

    data/<name>/corpus.jsonl   {"id": ..., "title": ..., "text": ...}
    data/<name>/queries.tsv    qid \t query
    data/<name>/qrels.tsv      qid \t docid \t relevance
"""
import json
import pathlib
import sys
import urllib.request
import zipfile

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATASETS = {"scifact": "test", "nfcorpus": "test", "quora": "test"}
ROOT = pathlib.Path(__file__).resolve().parent.parent


def download(name: str, raw: pathlib.Path) -> pathlib.Path:
    zpath = raw / f"{name}.zip"
    if zpath.exists():
        print(f"  zip already present: {zpath}")
        return zpath
    url = f"{BASE}/{name}.zip"
    print(f"  downloading {url}")
    raw.mkdir(parents=True, exist_ok=True)
    tmp = zpath.with_suffix(".zip.part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            print(f"\r  {f.tell() / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(zpath)
    return zpath


def normalize(name: str, split: str, zpath: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = {p.split("/", 1)[-1]: p for p in z.namelist() if not p.endswith("/")}

        n = 0
        with z.open(names["corpus.jsonl"]) as src, open(out / "corpus.jsonl", "w") as dst:
            for line in src:
                d = json.loads(line)
                # BEIR uses "_id"; keep only the three fields the indexer cares about.
                dst.write(json.dumps({
                    "id": d["_id"],
                    "title": (d.get("title") or "").strip(),
                    "text": (d.get("text") or "").strip(),
                }) + "\n")
                n += 1
        print(f"  corpus.jsonl   {n:>7} docs")

        # qrels first, so we can emit only the queries that actually have judgments.
        judged, pairs = set(), 0
        with z.open(names[f"qrels/{split}.tsv"]) as src, open(out / "qrels.tsv", "w") as dst:
            next(src)  # header: query-id\tcorpus-id\tscore
            for line in src:
                qid, docid, rel = line.decode().rstrip("\n").split("\t")[:3]
                dst.write(f"{qid}\t{docid}\t{rel}\n")
                judged.add(qid)
                pairs += 1
        print(f"  qrels.tsv      {pairs:>7} judgments over {len(judged)} queries")

        q = 0
        with z.open(names["queries.jsonl"]) as src, open(out / "queries.tsv", "w") as dst:
            for line in src:
                d = json.loads(line)
                if d["_id"] in judged:
                    dst.write(f"{d['_id']}\t{' '.join(d['text'].split())}\n")
                    q += 1
        print(f"  queries.tsv    {q:>7} judged queries")


def main() -> int:
    wanted = sys.argv[1:] or list(DATASETS)
    for name in wanted:
        if name not in DATASETS:
            print(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}", file=sys.stderr)
            return 2
        print(f"{name}:")
        zpath = download(name, ROOT / "data" / "raw")
        normalize(name, DATASETS[name], zpath, ROOT / "data" / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
