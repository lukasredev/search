# Lucene Search-Engine Teaching Demo — Build Plan

Goal: a locally-runnable search engine over a real IR benchmark that **shows its own work** —
indexing, BM25 candidate retrieval, dense (semantic) retrieval, RRF fusion, cross-encoder
reranking — with per-stage traces, score math, rank-movement visuals and per-stage eval metrics.
Audience: undergraduates. Primary artifact: a live demo driven from a browser during a talk.

---

## 1. Architecture

```
                 ┌──────────────────────────── browser ────────────────────────────┐
                 │  React UI: query box · pipeline waterfall · 4-stage rank board  │
                 │  BM25 explain tree · term inspector · RRF math · rerank deltas  │
                 └───────────────────────────────┬─────────────────────────────────┘
                                                 │ GET /api/search?q=...&trace=full
┌────────────────────────────────────────────────▼──────────────────────────────────┐
│ searchd  (Java 21 · Lucene 10.4 · Javalin)        — single JVM, serves UI too     │
│                                                                                    │
│  stage 0  analyze(q)          Analyzer chain trace (token-by-token)                │
│  stage 1  BM25 top-100        IndexSearcher + BM25Similarity(k1,b) + explain()     │
│  stage 2  dense top-100       KnnFloatVectorQuery over HNSW  ◄── embed(q) ──┐      │
│  stage 3  RRF fuse  -> 50     1/(k+rank) per list, k configurable           │      │
│  stage 4  rerank    -> 10     cross-encoder scores     ◄── rerank(q,docs) ──┤      │
│                                                                              │      │
│  every stage appends to one StageTrace -> returned verbatim in the response  │      │
└──────────────────────────────────────────────────────────────────────────────┼──────┘
                      ┌───────────────────────────────────────────────────────┘
                      ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ nlp  (Python 3.11 · FastAPI · sentence-transformers · CPU)                        │
│   POST /embed   all-MiniLM-L6-v2            384-d, normalized                     │
│   POST /rerank  ms-marco-MiniLM-L-6-v2      cross-encoder logits                  │
└───────────────────────────────────────────────────────────────────────────────────┘

one-shot:  indexer (same JVM image, different entrypoint) -> ./data/index  (docker volume)
```

Two containers, one `docker compose up`. The Java service owns Lucene, the pipeline and the
UI bundle; the Python service owns the two transformer models. Clean split for the talk: "the
search engine" vs "the ML models it calls".

### Why this split
- **Lucene 10 holds both indexes.** Inverted index (`TextField`) *and* HNSW graph
  (`KnnFloatVectorField`) live in the same `IndexWriter`/segments. Strong teaching point: a
  modern engine is one index with two retrieval paths, not two systems.
- **Python sidecar for the models.** `sentence-transformers` is what students will Google;
  swapping models is a one-line config change. (Alternative: all-Java via ONNX Runtime / DJL —
  removes a container but makes model swapping painful. Keep as a documented variant.)
- **No separate frontend container.** Vite builds to static assets served by Javalin, so prod
  is one JVM. `npm run dev` proxies to the JVM during development.

---

## 2. Data

### DECIDED: SciFact (default) + NFCorpus. Robust04 dropped.

| dataset | docs | queries | qrels | license | why |
|---|---|---|---|---|---|
| SciFact | 5.2k | 300 | yes | free (CC) | tiny; claim-style queries → excellent lexical-vs-semantic contrasts |
| NFCorpus | 3.6k | 323 | yes | free | consumer-health wording vs medical jargon = textbook vocabulary mismatch |
| TREC-COVID | 171k | 50 | yes | free | shows index size / HNSW build cost at a realistic scale |
| MS MARCO dev slice | configurable | 6.9k | yes (sparse) | free | web-like short queries if you want "looks like Google" |

**Robust04: dropped (decided 2026-10-07).** Its corpus is TREC Disks 4&5, which require a signed
NIST/LDC agreement and are not downloadable, so it could never ship in an image or go online. Its
only advantage for a talk was name recognition; its long verbose topics are less legible on a
slide than SciFact claims. No `robust04` loader will be written.

**TREC-COVID / MS MARCO:** not in scope, but the normalized format below means either can be added
later by writing only a fetch script.

### Normalized format (the dataset abstraction)
Everything is converted to three files, so any corpus drops in:
```
data/<name>/corpus.jsonl    {"id": "...", "title": "...", "text": "..."}
data/<name>/queries.tsv     qid \t query text
data/<name>/qrels.tsv       qid \t docid \t relevance
```
A `scripts/fetch_dataset.py` (uses `ir_datasets`) writes those. The Java indexer only ever reads
this format and knows nothing about BEIR.

---

## 3. Indexing (stage the talk opens on)

`indexer` is a single readable Java class, deliberately linear so it can be shown on a slide:

1. **Analyzer** — a custom `Analyzer` built from `StandardTokenizer` → `LowerCaseFilter` →
   `EnglishPossessiveFilter` → `StopFilter` → `PorterStemFilter`. Exposed to the UI so the
   analysis chain can be replayed token-by-token on arbitrary text.
2. **Fields per doc** — `id` (`StringField`, stored) · `title`, `text` (`TextField`, indexed, with
   term vectors + positions so we can do posting/term inspection and highlighting) ·
   `body_stored` (`StoredField`, for the reranker and snippets) ·
   `vector` (`KnnFloatVectorField`, 384-d, `COSINE`).
3. **Embedding pass** — batch of 64 doc texts → `POST /embed` → float[384]. Progress logged.
   SciFact on CPU: ~1–2 min. TREC-COVID: ~25–40 min (so cache the index in a volume / bake it).
4. **Commit + force-merge to 1 segment** for a stable, inspectable index.
5. **Write `index_stats.json`** — doc count, term count, per-file byte sizes grouped into
   *inverted index* vs *HNSW graph* vs *stored fields*. Directly becomes the "what does semantic
   search cost you?" slide.

Run it as `docker compose run --rm indexer scifact`, output to a named volume.

---

## 4. The API contract (one call returns the whole pipeline)

`GET /api/search?q=...&trace=full&k1=1.2&b=0.75&rrf_k=60&bm25_n=100&dense_n=100&fuse_n=50&rerank_n=10`

```jsonc
{
  "query": "...",
  "stages": [
    { "name": "analyze", "ms": 0.4,
      "detail": { "chain": [ {"filter":"StandardTokenizer","tokens":["Vitamin","D","deficiency"]},
                             {"filter":"LowerCaseFilter","tokens":["vitamin","d","deficiency"]},
                             {"filter":"StopFilter","tokens":["vitamin","d","deficiency"]},
                             {"filter":"PorterStemFilter","tokens":["vitamin","d","defici"]} ],
                  "terms": [ {"term":"defici","df":312,"idf":2.81} ] } },
    { "name": "bm25",   "ms": 6.2,  "params": {"k1":1.2,"b":0.75,"n":100},
      "results": [ {"docId":"4983","rank":1,"score":14.22,
                    "explain":{"terms":[{"term":"defici","tf":4,"df":312,"idf":2.81,
                                         "fieldLen":182,"avgFieldLen":214,"contrib":5.1}]}} ] },
    { "name": "dense",  "ms": 11.8, "params": {"model":"all-MiniLM-L6-v2","efSearch":100},
      "embedMs": 8.1, "visitedCount": 1431,
      "results": [ {"docId":"7721","rank":1,"score":0.812} ] },
    { "name": "rrf",    "ms": 0.3,  "params": {"k":60},
      "results": [ {"docId":"4983","rank":1,"score":0.0320,
                    "contrib":{"bm25":{"rank":1,"term":0.01639},
                               "dense":{"rank":7,"term":0.01493}}} ] },
    { "name": "rerank", "ms": 940,  "params": {"model":"ms-marco-MiniLM-L-6-v2","maxLen":512},
      "results": [ {"docId":"7721","rank":1,"score":7.41,"prevRank":4,"delta":3,
                    "truncatedTokens": 118} ] }
  ],
  "eval": { "qid":"42",
            "perStage": { "bm25":{"ndcg@10":0.41,"recall@100":0.78},
                          "dense":{"ndcg@10":0.49,"recall@100":0.85},
                          "rrf":{"ndcg@10":0.56},
                          "rerank":{"ndcg@10":0.71} } },
  "docs": { "4983": {"title":"...","snippet":"...","relevance":2} }
}
```

Design rule: **the UI invents nothing.** Every number shown on screen is in this payload, so
"open the Network tab" is itself a demo step. `trace=off` returns just the final 10 — useful for
showing the production-vs-debug cost difference.

Other endpoints: `/api/analyze?text=`, `/api/term?t=` (df, total tf, idf, containing docs),
`/api/doc/{id}`, `/api/stats` (index_stats.json), `/api/benchmark` (batch eval, cached),
`/api/queries` (benchmark queries + curated demo queries).

---

## 5. The insight UI — the actual deliverable

Single page, query box on top, stage panels below. Ordered by talk value:

1. **Rank board** (the money shot) — four vertical columns BM25 · Dense · RRF · Reranked, each a
   ranked list of doc cards. Bezier connector lines between columns, colored by movement:
   promoted (green, thickness = rank gain), demoted (red), dropped (fades out), newly introduced
   by the other retriever (blue, enters from the side). Hovering a doc highlights its path across
   all four columns. Relevant docs (from qrels) get a gold border — so the audience *sees*
   relevance migrating to the top as stages apply.
2. **Timing waterfall** — horizontal bars per stage, same payload `ms` values. Makes the
   latency/quality tradeoff concrete, and the ~1s cross-encoder bar lands the "that's why you only
   rerank 50, not 100,000" point by itself.
3. **BM25 breakdown** — click any BM25 doc → table of its query terms with tf, df, idf, field
   length vs average, and each term's contribution as a stacked bar summing to the score. Live
   `k1` / `b` sliders re-run stage 1 and animate the ranking. Rendered from Lucene's own
   `explain()`, with the raw explain tree available in a collapsible "verbatim Lucene" block.
4. **Analyzer / term inspector** — the chain trace from stage 0 as a pipeline of token chips;
   type arbitrary text and watch it tokenize/stem. Click a term → df/idf and the posting list
   (which candidate docs contain it, with positions). This is the inverted index made visible.
5. **RRF panel** — one row per doc: `rank_bm25`, `rank_dense`, the two `1/(k+rank)` terms written
   out as arithmetic, the sum, the new rank. `k` slider. Toggles for BM25-only / dense-only /
   fused so you can show fusion beating both parents.
6. **Semantic panel** — cosine scores; a 2D PCA scatter (basis precomputed at index time) of the
   query vector and candidate vectors, so "nearby in embedding space" stops being a metaphor;
   HNSW `efSearch` knob showing the recall/latency tradeoff via `visitedCount`.
7. **Reranker panel** — scatter of RRF score (x) vs cross-encoder score (y) with the diagonal
   drawn: points far off the diagonal are exactly where the reranker earns its keep. Shows the
   truncated text the cross-encoder actually saw (512-token limit is a real, teachable gotcha).
8. **Eval strip** — nDCG@10 / Recall@100 after each stage for the current benchmark query, plus a
   "run all 300 queries" mode producing a stage × metric table (precomputed and cached, with the
   per-stage deltas highlighted). The single most convincing slide in the talk: each stage's
   contribution, measured.
9. **Index stats page** — docs, terms, segments, and the inverted-index vs HNSW vs stored-fields
   byte breakdown.
10. **Curated demo queries** — a dropdown of ~8 hand-picked queries, each labelled with what it
    demonstrates: *lexical win* (exact rare term, dense misses it) · *semantic win* (paraphrase,
    zero term overlap) · *fusion win* (neither alone is good) · *reranker rescue* (relevant doc
    sitting at rank 23 after fusion, pulled to 1). Verifying these against the real index is a
    build task, not an afterthought — the talk depends on them.

Design: dark-first, high contrast, large type (projector-legible), no hover-only information
(nothing on a projector is hoverable from the back row), keyboard-driven, and a `?preset=` URL
param so each slide can deep-link a prepared state.

---

## 6. Repo layout

```
search/
├─ docker-compose.yml
├─ Makefile                      # make index DATASET=scifact / make up / make bench
├─ searchd/                      # Java 21, Maven
│  ├─ pom.xml                    # lucene-core, -analysis-common, -queryparser, -highlighter, javalin, jackson
│  └─ src/main/java/.../
│     ├─ Indexer.java            # linear, slide-readable
│     ├─ DemoAnalyzer.java
│     ├─ SearchPipeline.java     # the 5 stages, one method each
│     ├─ Bm25Stage.java  DenseStage.java  RrfFusion.java  RerankStage.java
│     ├─ StageTrace.java         # the trace object
│     ├─ Evaluator.java          # nDCG@10, Recall@k, MRR from qrels
│     ├─ NlpClient.java          # HTTP to the python sidecar
│     └─ Api.java                # Javalin routes + static UI
├─ nlp/                          # FastAPI, Dockerfile, models baked into image
├─ ui/                           # Vite + React + TS + Tailwind
├─ scripts/fetch_dataset.py      # ir_datasets -> corpus.jsonl/queries.tsv/qrels.tsv
└─ data/                         # gitignored; datasets + index (volume)
```

---

## 7. Build phases

| phase | deliverable | verification |
|---|---|---|
| **P0** ✅ toolchain | JDK 21.0.12.1 + Maven 3.9.16 in `~/.local/opt`, uv venv (py3.11) for nlp, repo skeleton, git init | `mvn package` passes |
| **P1** ✅ index + BM25 | SciFact + NFCorpus fetched, `Indexer`, `/api/search` + `explain()` decomposition | **nDCG@10 = 0.683** vs published BEIR BM25 baseline 0.665 ✓ |
| **P2** ✅ dense | nlp sidecar, cached embedding pass, `KnnFloatVectorField` + instrumented leaf-level HNSW search | **nDCG@10 = 0.645**, exactly the published all-MiniLM-L6-v2 figure ✓ |
| **P3** ✅ RRF | fusion stage, per-list contributions and the arithmetic in the trace | **nDCG@10 = 0.720 > both parents** ✓ |
| **P4** ✅ rerank | cross-encoder stage, rank deltas, truncation flags; plain UI | ~55 ms/doc; **rerank makes SciFact worse (0.720 → 0.693)** — reproduced across 4 models and 3 pool depths, see `docs/reranker-findings.md` |
| **P5** insight UI | panels 1–7 above | walk every panel against 3 real queries |
| **P6** eval | `/api/benchmark` batch run + cached stage×metric table, panel 8 | full-300-query table reproducible |
| **P7** ship | Dockerfiles, compose, prebaked index, README with a talk runbook; curated demo queries verified | `docker compose up` on a clean machine → working demo |

Order is deliberately "working end-to-end at every phase": after P1 you already have a demoable
search engine. If time runs short, the talk still works at P4 with a plainer UI.

---

## 8. Running it

**Locally (your machine, Docker):**
```
make index DATASET=scifact     # one-shot indexer container, writes to the volume
make up                        # searchd :8080 (UI + API) + nlp :8081
```
First run pulls ~1.5 GB of model weights into the `nlp` image layer; after that it is offline-capable.

**Online:** same compose on one small VM (Hetzner CX22 / Fly.io), ~2 vCPU / 4 GB is enough for
SciFact with CPU models. Bake the prebuilt index into the `searchd` image so the instance needs no
index build. Put HTTP basic auth or a Cloudflare Access rule in front, and cap `rerank_n` so a
stray crawler can't burn CPU. Build images for the right arch — note this dev box is **aarch64**,
so use `docker buildx --platform` if the target VM is x86.

**Resource notes (measured constraints of this box):** 5 cores, 11 GB RAM, 36 GB free — fine for
SciFact/NFCorpus; TREC-COVID's HNSW build will be the slow part.

---

## 9. Decisions & risks

**Decided 2026-10-07:**
- **Corpus:** SciFact + NFCorpus only; Robust04 dropped (§2).
- **Verification:** build and verify natively on this box (JDK 21 + Maven into `~/.local`, uv venv
  for Python). Dockerfiles and compose are written but **not verified here** — validating
  `docker compose up` is a task for your own machine.
- **Scope:** build through **P4** (full retrieval pipeline, measured, plain UI), then review before
  investing in the insight panels (P5–P7).

**Open decision (P5):** cross-encoder reranking measurably *hurts* on SciFact and no model or
pool depth fixes it (`docs/reranker-findings.md`). Either keep it and teach the negative result —
recommended, it is a better lesson than a stage that silently works — or add TREC-COVID, where
reranking is known to pay for itself, at the cost of ~40 min one-time embedding for 171k docs.

**Risks:**
1. **Docker build unverified** on this box (no Docker, no passwordless sudo). P7 needs your machine.
2. **Cross-encoder latency on CPU** ≈ 0.7–1.5 s for 50 docs × 512 tokens. Acceptable, and honestly
   a feature for the talk — but it means no snappy type-as-you-search on the rerank stage. Mitigate
   with a progressive response: stream stages 1–3 immediately, patch in stage 4 when it lands.
3. **Torch on aarch64** — CPU wheels exist and work; just slower than x86 AVX. Pin versions.
4. **Curated demo queries can rot** when parameters change. `scripts/find_demo_queries.py`
   regenerates them against the live index; re-run it after touching the analyzer, the embedding
   model or the RRF constant.
5. **Abandoned requests are still computed in full** server-side and only fail at write time, so
   a client that times out and retries costs the server twice. Measured: 180 aborted requests =
   5 s of worker CPU. Knobs are now clamped (`rerank_n` ≤ 200, candidates ≤ 1000); don't let the
   UI be hammered mid-talk.
6. **One box cannot index, benchmark and demo at once.** The embedding sidecar is the shared
   bottleneck; overlapping runs caused request timeouts that looked like a server hang.
