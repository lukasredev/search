#!/usr/bin/env python3
"""Find the benchmark queries that best illustrate each teaching point.

A demo is only as good as its examples. This scans the whole query set against a
running searchd and ranks queries by how sharply they show one specific thing:

  lexical-win   BM25 finds it, the embedding model does not  (rare exact term)
  semantic-win  the embedding model finds it, BM25 does not  (vocabulary mismatch)
  fusion-win    neither retriever is good alone, fusion is   (the case for RRF)

Writes ui/demo_queries.json. Run it again after changing the analyzer, the
embedding model or the fusion constant — the examples can go stale.

    python3 scripts/find_demo_queries.py [--host http://localhost:8080] [--top N]
"""
import argparse
import json
import math
import pathlib
import re
import urllib.parse
import urllib.request

# Topic presets, so the curated examples match the audience in the room. A talk to CS
# undergraduates wants "how should I start learning programming?", not whichever query
# happens to have the largest metric delta.
TOPICS = {
    "cs": r"\b(program(?:ming|mer)?|cod(?:e|ing)|python|java|javascript|c\+\+|php|ruby|rust|"
          r"golang|algorithm|software|computer|machine learning|deep learning|neural net|"
          r"database|sql|linux|unix|data structure|compiler|api|git(?:hub)?|developer|"
          r"web development|android|ios|app|encryption|cryptograph|network|operating system|"
          r"recursion|hash|artificial intelligence|data scien|cyber|security|hack|server|"
          r"framework|html|css|debug|stack overflow|engineering)\b",
}

ROOT = pathlib.Path(__file__).resolve().parent.parent


def fetch(url, attempts=3):
    """The embedding sidecar is shared, so a busy box can stall a request. Retry."""
    for attempt in range(attempts):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=300).read())
        except (TimeoutError, OSError):
            if attempt == attempts - 1:
                raise
            print("  request timed out, retrying", flush=True)


def ndcg(ranked, rel, k=10):
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--top", type=int, default=5, help="examples to keep per category")
    ap.add_argument("--topic", help=f"restrict to a preset ({', '.join(TOPICS)}) or a regex")
    ap.add_argument("--limit", type=int, help="scan at most this many queries")
    args = ap.parse_args()

    data = ROOT / "data" / args.dataset
    qrels = {}
    for line in open(data / "qrels.tsv"):
        qid, did, rel = line.rstrip("\n").split("\t")
        if float(rel) > 0:
            qrels.setdefault(qid, {})[did] = int(float(rel))
    queries = dict(l.rstrip("\n").split("\t", 1) for l in open(data / "queries.tsv"))

    if args.topic:
        pattern = re.compile(TOPICS.get(args.topic, args.topic), re.I)
        queries = {q: t for q, t in queries.items() if pattern.search(t)}
        print(f"topic filter {args.topic!r}: {len(queries)} matching queries")
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])

    rows = []
    for i, (qid, text) in enumerate(queries.items(), 1):
        url = (f"{args.host}/api/search?rerank=false&explain=false&fuse_n=50"
               f"&q={urllib.parse.quote(text)}")
        d = fetch(url)
        stages = {s["name"]: s for s in d["stages"]}
        if "bm25" not in stages or "dense" not in stages:
            continue

        rel = qrels.get(qid, {})
        got = {name: ndcg([r["docId"] for r in stages[name]["results"]], rel)
               for name in ("bm25", "dense", "rrf")}
        rows.append({
            "qid": qid, "text": text, "relevantDocs": len(rel),
            "ndcg": {k: round(v, 4) for k, v in got.items()},
            "overlap": stages["rrf"]["detail"]["overlap"],
        })
        if i % 50 == 0:
            print(f"  {i}/{len(queries)}", flush=True)

    def pick(key, label, note):
        best = sorted(rows, key=key, reverse=True)[:args.top]
        return [{**r, "category": label, "teaches": note} for r in best if key(r) > 0]

    out = {
        "dataset": args.dataset,
        "lexical-win": pick(
            lambda r: r["ndcg"]["bm25"] - r["ndcg"]["dense"], "lexical-win",
            "A rare, exact term. BM25 nails it; the embedding model blurs it into "
            "topically-similar neighbours."),
        "semantic-win": pick(
            lambda r: r["ndcg"]["dense"] - r["ndcg"]["bm25"], "semantic-win",
            "Vocabulary mismatch: the relevant document says the same thing in different "
            "words, so there is little term overlap for BM25 to work with."),
        "fusion-win": pick(
            lambda r: r["ndcg"]["rrf"] - max(r["ndcg"]["bm25"], r["ndcg"]["dense"]), "fusion-win",
            "Neither retriever is good alone. Each surfaces part of the answer and RRF "
            "rewards the documents both of them half-agreed on."),
    }

    # Written next to the index so the service can serve it; the UI asks the API for it.
    dest = ROOT / "data" / "index" / args.dataset / "demo_queries.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out["topic"] = args.topic
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    for cat in ("lexical-win", "semantic-win", "fusion-win"):
        print(f"\n{cat}:")
        for r in out[cat]:
            n = r["ndcg"]
            print(f"  [{r['qid']:>4}] bm25={n['bm25']:.3f} dense={n['dense']:.3f} "
                  f"rrf={n['rrf']:.3f}  {r['text'][:72]}")


if __name__ == "__main__":
    main()
