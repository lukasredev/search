# Lucene Search-Engine Teaching Demo

A small, complete search engine built on Apache Lucene that **shows its own work**. Type a
query and you see every stage of a modern retrieval pipeline — what the analyzer did to your
words, which documents BM25 found and exactly why, which ones the vector index found instead,
how the two rankings were fused, and what the cross-encoder changed — with per-stage quality
metrics computed against real relevance judgments.

Built for a talk on search engine architecture.

```
analyze ─┬─ BM25 top-100  (inverted index) ─┐
         │                                   ├─ RRF fuse ─ cross-encoder rerank ─ top-10
         └─ dense top-100 (HNSW graph)      ─┘
```

## The pitch, in three numbers

Measured on BEIR SciFact, 5,183 documents, 300 judged queries (`make bench`):

| after stage | nDCG@10 | recall@100 | avg latency |
|---|---|---|---|
| BM25 | 0.683 | 0.928 | 0.4 ms |
| dense | 0.645 | 0.925 | 15 ms |
| RRF fusion | **0.720** | **0.941** | 0.2 ms |
| + cross-encoder rerank | 0.693 | 0.941 | 3113 ms |

Neither retriever is better than the other — they are wrong about *different* documents, and
fusing them beats both. That is the whole argument for a hybrid pipeline, in one table.

And then the fourth row refuses to cooperate: the cross-encoder costs 3 seconds a query and
makes the ranking *worse*. That is not a bug — it reproduces across four reranker models and
three pool depths, and it matches the published BEIR figures. See
[docs/reranker-findings.md](docs/reranker-findings.md); it is the most interesting slide in the
deck, not the broken one.

## Running it

Prerequisites: JDK 21, Maven, Python 3.11 (via `uv`). No Docker needed for local dev.

```bash
make data            # fetch + normalize SciFact and NFCorpus (~5 MB, no auth)
make nlp             # terminal 1: embedding + reranking sidecar on :8081
make index           # terminal 2: build the Lucene index (~2 min, embeds the corpus)
make serve           # terminal 2: search API + UI on http://localhost:8080
```

Then open <http://localhost:8080>. `make index` caches its embeddings in
`data/<dataset>/vectors.f32`, so re-indexing after an analyzer or HNSW change takes seconds,
not minutes.

Switch corpora with `DATASET=nfcorpus make index serve`.

## What to show during the talk

**Indexing.** `Indexer.java` is deliberately linear and fits on a slide. The point to land:
one `IndexWriter` writes both the inverted index (a `TextField`) and the HNSW graph (a
`KnnFloatVectorField`), into the same segments. A modern engine is one index with two
retrieval paths, not two systems bolted together.

`GET /api/stats` then shows what each path costs on disk:

| component | size |
|---|---|
| HNSW vectors | 8.2 MB |
| stored fields | 4.9 MB |
| term vectors | 4.7 MB |
| inverted index | **2.1 MB** |

The semantic index is four times the size of the lexical one. Worth saying out loud.

**Analysis.** The UI replays the analyzer chain one filter at a time —
`StandardTokenizer → EnglishPossessiveFilter → LowerCaseFilter → StopFilter → PorterStemFilter`
— so students watch `deficiency` become `defici` and `is`/`to` disappear. `GET /api/analyze?text=`
does the same for arbitrary text.

**BM25.** Click any BM25 result to see its score decomposed per term: tf, df, idf, document
length vs. average, and each term's contribution. This comes from Lucene's own
`IndexSearcher.explain()`, and the verbatim explanation is in the payload too. The `k1` and `b`
knobs re-run the query live.

**Dense retrieval.** The stage reports the embedding time separately from the graph-walk time,
plus `visitedCount` — how many of the 5,183 vectors the HNSW search actually touched (typically
about 20%). Turn `efSearch` down and watch visited count and recall fall together.

**Fusion.** The RRF panel prints the arithmetic for every document:
`1/(60+1) + 1/(60+7) = 0.016393 + 0.014925 = 0.031318`. The header line reports how many
candidates both retrievers found — usually under a third, which is exactly why fusion works.

**Reranking.** Rank deltas are drawn per document, and the latency bar makes the argument for
you: the cross-encoder costs ~60 ms per document, so scoring all 5,183 would take five minutes.
That is why there is a candidate-retrieval phase at all. Documents past the model's 512-token
window are flagged `truncated` — a real gotcha, judged on their opening paragraph alone.

**Measuring whether a stage earns its place.** Pick any of the 300 benchmark queries and the eval table fills in with nDCG@10,
recall and MRR *after each stage*, with judged-relevant documents gold-bordered in the rankings
so you can watch relevance migrate to the top.

## API

Everything the UI draws comes from one call. The frontend computes nothing of its own, so
"open the network tab" is itself a demo step.

```
GET /api/search?q=...&mode=hybrid&k1=1.2&b=0.75&rrf_k=60&ef_search=100&rerank_n=50
```

returns `{query, qid, params, totalMs, stages[], eval, docs}` where each stage carries its
`params`, `ms`, `detail` and full `results`. `trace=off` returns just the top 10 — show it
next to the full trace to make the cost of instrumentation concrete.

| endpoint | what it gives you |
|---|---|
| `GET /api/analyze?text=` | the analyzer chain, step by step |
| `GET /api/term?t=` | df, idf, total term frequency, and the posting list |
| `GET /api/doc/{id}` | stored document plus its highest-frequency terms |
| `GET /api/stats` | index statistics and the on-disk byte breakdown |
| `GET /api/benchmark` | cached full-query-set results |
| `GET /api/queries` | the judged benchmark queries |
| `GET /api/health` | index and sidecar status |

`mode=bm25_only` / `mode=dense_only` isolate a single retriever — useful for showing that
fusion beats both parents.

## Layout

```
searchd/     Java 21 + Lucene 10.4 + Javalin. Index, pipeline, API, static UI host.
  Indexer.java        builds the index (slide-readable on purpose)
  SearchPipeline.java the five stages, one method each
  Bm25Stage.java      BM25 + explain() decomposition
  DenseStage.java     HNSW search, instrumented for visited count
  RrfFusion.java      rank fusion with the arithmetic spelled out
  RerankStage.java    cross-encoder call + rank deltas
  Evaluator.java      nDCG, recall, MRR
  Benchmark.java      full query set -> benchmark.json
nlp/         FastAPI sidecar: all-MiniLM-L6-v2 + ms-marco-MiniLM-L-6-v2, CPU
ui/          single static page, no build step
scripts/     dataset fetch + normalization (stdlib only)
data/        datasets and index (gitignored)
```

## Data

BEIR **SciFact** (5,183 docs / 300 queries) and **NFCorpus** (3,633 docs / 323 queries), both
freely downloadable with relevance judgments. Normalized on fetch to three files —
`corpus.jsonl`, `queries.tsv`, `qrels.tsv` — so any corpus in that shape drops straight in; the
Java side knows nothing about BEIR.

Robust04 is deliberately **not** supported: its corpus is TREC Disks 4&5, which require a signed
NIST agreement and cannot be redistributed or deployed. SciFact's claim-style queries also make
cleaner lexical-vs-semantic contrasts than Robust04's verbose topics.

## Status

Phases P0–P4 of `PLAN.md` are complete: indexing, BM25, dense retrieval, RRF, reranking, and
per-stage evaluation, with a plain UI. Still to do — the insight visualizations (rank-movement
connectors, PCA scatter, posting inspector), the curated demo-query set, and Docker packaging.
