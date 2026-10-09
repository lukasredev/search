#!/usr/bin/env python3
"""Fetch a BEIR dataset and normalize it to the three files the Java indexer reads.

Deliberately stdlib-only: the indexer must never depend on a Python environment,
and students should be able to read this top to bottom.

    data/<name>/corpus.jsonl   {"id": ..., "title": ..., "text": ...}
    data/<name>/queries.tsv    qid \t query
    data/<name>/qrels.tsv      qid \t docid \t relevance
"""
import contextlib
import json
import pathlib
import sys
import urllib.request
import zipfile

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATASETS = {"scifact": "test", "nfcorpus": "test", "quora": "test", "cqadupstack-unix": "test"}
# BEIR only ships CQADupStack as one 5 GB zip of twelve subforums; MTEB mirrors each
# subforum on its own, as the same BEIR files without the zip.
HF = {"cqadupstack-unix": "https://huggingface.co/datasets/mteb/cqadupstack-unix/resolve/main"}
ROOT = pathlib.Path(__file__).resolve().parent.parent


def fetch(url: str, dest: pathlib.Path) -> pathlib.Path:
    if dest.exists():
        print(f"  already present: {dest}")
        return dest
    print(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            print(f"\r  {f.tell() / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


@contextlib.contextmanager
def download(name: str, split: str, raw: pathlib.Path):
    """Yield open(relative_path) over the dataset's BEIR files, wherever they came from."""
    if name in HF:
        files = ["corpus.jsonl", "queries.jsonl", f"qrels/{split}.tsv"]
        for f in files:
            fetch(f"{HF[name]}/{f}", raw / name / f)
        yield lambda f: open(raw / name / f, "rb")
        return
    zpath = fetch(f"{BASE}/{name}.zip", raw / f"{name}.zip")
    with zipfile.ZipFile(zpath) as z:
        names = {p.split("/", 1)[-1]: p for p in z.namelist() if not p.endswith("/")}
        yield lambda f: z.open(names[f])


def normalize(split: str, open_file, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open_file("corpus.jsonl") as src, open(out / "corpus.jsonl", "w", encoding="utf-8") as dst:
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
    with open_file(f"qrels/{split}.tsv") as src, open(out / "qrels.tsv", "w", encoding="utf-8") as dst:
        next(src)  # header: query-id\tcorpus-id\tscore
        for line in src:
            qid, docid, rel = line.decode().rstrip("\n").split("\t")[:3]
            dst.write(f"{qid}\t{docid}\t{rel}\n")
            judged.add(qid)
            pairs += 1
    print(f"  qrels.tsv      {pairs:>7} judgments over {len(judged)} queries")

    q = 0
    with open_file("queries.jsonl") as src, open(out / "queries.tsv", "w", encoding="utf-8") as dst:
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
        with download(name, DATASETS[name], ROOT / "data" / "raw") as open_file:
            normalize(DATASETS[name], open_file, ROOT / "data" / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
