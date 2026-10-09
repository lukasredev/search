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

## The pitch, in one table

nDCG@10 after each stage, measured with `make bench`, against published BEIR baselines:

| | BM25 | dense | RRF fusion | + reranker |
|---|---|---|---|---|
| **CQADupStack unix** — 47k Stack Exchange questions | 0.294 | **0.412** | 0.375 | 0.374 |
| **SciFact** — 5k biomedical claims | **0.683** | 0.645 | **0.720** | 0.693 |
| **NFCorpus** — 3.6k nutrition abstracts | **0.339** | 0.312 | **0.350** | — |
| **Quora** — 523k duplicate questions | 0.778 | **0.839** | **0.839** | 0.796 |

Read it across, not down. **Which retriever wins depends entirely on the data.** SciFact is
full of exact technical terms, so matching words beats matching meaning. Quora is people asking
the same question in different words, so matching meaning wins by six points and fusion adds
nothing on top of it. On CQADupStack unix the embedding model wins by twelve points, and fusing
in the weaker BM25 ranking costs four of them back. Anyone who tells you one retrieval method is
simply better has not run both on their own data.

And the last column refuses to cooperate anywhere: the cross-encoder makes the ranking *worse*
on SciFact and Quora and leaves unix flat (0.375 → 0.374). That is not a bug — on SciFact and
Quora it reproduces across four reranker models and three pool depths and matches the published
figures. See
[docs/reranker-findings.md](docs/reranker-findings.md); it is the most interesting slide in the
deck, not the broken one.

## Running it

Prerequisites: JDK 21, Maven, Python 3.11 (via `uv`). No Docker needed for local dev.

```bash
python3 scripts/fetch_dataset.py cqadupstack-unix   # ~50 MB, no auth; `make data` fetches all
make nlp             # terminal 1: embedding + reranking sidecar on :8081
make index           # terminal 2: build the Lucene index (~15 min, embeds the corpus)
make serve           # terminal 2: search API + UI on http://localhost:8080
```

Then open <http://localhost:8080>. `make index` caches its embeddings in
`data/<dataset>/vectors.f32`, so re-indexing after an analyzer or HNSW change takes seconds,
not minutes.

`make serve` listens on loopback only, so the auth-less API is not offered to everyone on the
venue Wi-Fi. To drive the demo from another device, set `LISTEN_HOST=0.0.0.0` and also
`AUTH_USER`/`AUTH_PASS`.

The default corpus is **CQADupStack unix**: 47k questions from Unix & Linux Stack Exchange,
with their bodies, and 1,072 queries whose relevant documents are the questions moderators
closed as duplicates. It is the one to use for a computer science audience, because every
stage of the pipeline has queries where it visibly changes what is on screen, not just the
metric:

> *"What's the difference between $@ and $\*"* — the analyzer throws away `$@` and `$*`,
> BM25 is left with "what differ between" and returns curl vs wget. The embedding model ranks
> three correct answers first.

Switch corpora with `DATASET=quora make index serve`. Quora has one-line documents and many
unlabelled near-duplicates, so later stages mostly reshuffle results that already look right;
it is 523k documents, so budget ~25 minutes for the one-time embedding pass (cached afterwards).

To run it on a server, see [docs/deploy.md](docs/deploy.md).

## What to show during the talk

**Indexing.** `Indexer.java` is deliberately linear and fits on a slide. The point to land:
one `IndexWriter` writes both the inverted index (a `TextField`) and the HNSW graph (a
`KnnFloatVectorField`), into the same segments. A modern engine is one index with two
retrieval paths, not two systems bolted together.

`GET /api/stats` then shows what each path costs on disk (CQADupStack unix, 47k posts):

| component | size |
|---|---|
| HNSW vectors | 75.4 MB |
| term vectors | 28.8 MB |
| stored fields | 28.0 MB |
| inverted index | **13.8 MB** |

The semantic index is five times the size of the lexical one. Worth saying out loud.

**Analysis.** The UI replays the analyzer chain one filter at a time —
`StandardTokenizer → EnglishPossessiveFilter → LowerCaseFilter → StopFilter → PorterStemFilter`
— so students watch `deficiency` become `defici` and `is`/`to` disappear. `GET /api/analyze?text=`
does the same for arbitrary text.

**BM25.** Click any BM25 result to see its score decomposed per term: tf, df, idf, document
length vs. average, and each term's contribution. This comes from Lucene's own
`IndexSearcher.explain()`, and the verbatim explanation is in the payload too. The `k1` and `b`
knobs re-run the query live.

**Dense retrieval.** The stage reports the embedding time separately from the graph-walk time,
plus `visitedCount` — how many of the 47,382 vectors the HNSW search actually touched (typically
about 3%). Turn `efSearch` down and watch visited count and recall fall together.

**Fusion.** The RRF panel prints the arithmetic for every document:
`1/(60+1) + 1/(60+7) = 0.016393 + 0.014925 = 0.031318`. The header line reports how many
candidates both retrievers found — usually under a third, which is exactly why fusion works.

**Reranking.** Rank deltas are drawn per document, and the latency bar makes the argument for
you: the cross-encoder costs ~70 ms per document on CPU, so scoring all 47,382 would take about
an hour.
That is why there is a candidate-retrieval phase at all. Documents past the model's 512-token
window are flagged `truncated` — a real gotcha, judged on their opening paragraph alone.

**Measuring whether a stage earns its place.** Pick any of the 1,072 judged queries and the eval
table fills in with nDCG@10, recall and MRR *after each stage*, with judged-relevant documents
ticked in the rankings so you can watch relevance migrate to the top. The last chapter puts the
same table over the whole query set beside it, from `make bench`.

## API

Everything the UI draws comes from one call. The frontend computes nothing of its own, so
"open the network tab" is itself a demo step.

```
GET /api/search?q=...&mode=hybrid&k1=1.2&b=0.75&rrf_k=60&ef_search=100&rerank_n=50
```

returns `{query, qid, params, totalMs, stages[], eval, docs}` where each stage carries its
`params`, `ms`, `detail` and full `results`. `trace=off` returns just the top 10 — show it
next to the full trace to make the cost of instrumentation concrete. Each of those results
names the stage its score came from in `scoredBy`: with `rerank_n` below 10, the ones past
the reranked head keep their fusion score.

| endpoint | what it gives you |
|---|---|
| `GET /api/analyze?text=` | the analyzer chain, step by step |
| `POST /api/index-preview` | `{"text": ...}` → what indexing that text would store: tokens, postings, vector. Writes nothing |
| `GET /api/term?t=&limit=` | df, idf, total term frequency, and up to 1,000 postings |
| `GET /api/doc/{id}` | stored document, its highest-frequency terms and its stored vector |
| `GET /api/stats` | index statistics and the on-disk byte breakdown |
| `GET /api/benchmark` | cached full-query-set results |
| `GET /api/demo-queries` | the curated example queries |
| `GET /api/queries` | the judged benchmark queries |
| `GET /api/health` | index and sidecar status |
| `GET /livez`, `/readyz` | Kubernetes probes; exempt from basic auth |

`mode=bm25_only` / `mode=dense_only` isolate a single retriever — useful for showing that
fusion beats both parents.

## Layout

```
searchd/     Java 21 + Lucene 10.4 + Javalin. Index, pipeline, API, static UI host.
  Main.java           HTTP API, basic auth, static UI host
  Indexer.java        builds the index (slide-readable on purpose)
  IndexStory.java     dry-run indexing of arbitrary text, for the first chapter
  SearchPipeline.java the five stages, one method each
  Bm25Stage.java      BM25 + explain() decomposition
  DenseStage.java     HNSW search, instrumented for visited count
  RrfFusion.java      rank fusion with the arithmetic spelled out
  RerankStage.java    cross-encoder call + rank deltas
  Evaluator.java      nDCG, recall, MRR
  Benchmark.java      full query set -> benchmark.json
  Probes.java         /livez and /readyz
  src/test/           JUnit: every stage, the HTTP API and auth, against a fake sidecar
nlp/         FastAPI sidecar: all-MiniLM-L6-v2 + ms-marco-MiniLM-L-6-v2, CPU, pinned revisions
tests/       end-to-end checks against a running stack (make test-live)
ui/          single static page, no build step; six chapters, one per pipeline stage
scripts/     dataset fetch + normalization (stdlib only), demo-query finder
docs/        deployment guide, reranker findings
data/        datasets and index (gitignored, except each index's example queries and benchmark)
```

## Tests

```bash
make test        # ~40 s: Java and Python, no models or index needed
make test-live   # against a running stack: URL=http://localhost:8080 by default
```

`make test` builds a nine-document index with the real `Indexer` and serves it through the
real HTTP app, with a fake sidecar that speaks the same JSON (hashed bag-of-words embeddings,
word-overlap rerank scores). That covers every stage, the API, basic auth, the probes and the
embedding cache. The sidecar's own tests swap its two models for fakes and pin the response
shapes `NlpClient` reads. `make test-live` is the end-to-end half: real models, real index, and
the one assumption no fake can check, which is that the two services agree.

## Data

BEIR **CQADupStack unix** (47,382 docs / 1,072 queries, the default), **SciFact** (5,183 docs /
300 queries), **NFCorpus** (3,633 docs / 323 queries) and **Quora** (522,931 docs), all freely
downloadable with relevance judgments. CQADupStack comes from the MTEB mirror on Hugging Face,
because BEIR only ships it as one 5 GB zip of twelve subforums. Normalized on fetch to three files —
`corpus.jsonl`, `queries.tsv`, `qrels.tsv` — so any corpus in that shape drops straight in; the
Java side knows nothing about BEIR.

Robust04 is deliberately **not** supported: its corpus is TREC Disks 4&5, which require a signed
NIST agreement and cannot be redistributed or deployed. SciFact's claim-style queries also make
cleaner lexical-vs-semantic contrasts than Robust04's verbose topics.

## Status

Complete and used for the talk: the full pipeline, per-stage evaluation and benchmarks on four
corpora, a six-chapter UI with curated example queries, and Docker images that also run on
Kubernetes (see [docs/deploy.md](docs/deploy.md)). Ideas from `PLAN.md` that were not built: the
rank-movement connectors between columns, a PCA scatter of the embedding space, and a posting
inspector in the UI (the data is there: `GET /api/term`).
