# Does the cross-encoder actually help?

Short answer on SciFact: **no**, and that is worth a slide.

## What was measured

The authoritative run is the full query set (`make bench`, 300 queries, RRF top-50 reranked):

| after stage | nDCG@10 | recall@100 | avg latency |
|---|---|---|---|
| BM25 | 0.6826 | 0.9282 | 0.5 ms |
| dense | 0.6451 | 0.9250 | 18 ms |
| RRF fusion | **0.7197** | **0.9410** | 0.2 ms |
| + cross-encoder rerank | 0.6932 | 0.9410 | 3113 ms |

Reranking costs 3 seconds per query and gives back 2.7 nDCG points.

## Was it the model?

Four rerankers of increasing size and quality, scoring the *same* RRF candidate lists.
(First two on 100 queries at pool depth 25; last two on 50 queries, same pool — compare the
deltas, not the absolute numbers, since the baselines differ by subset.)

| reranker | params | Δ nDCG@10 vs RRF | ms/doc |
|---|---|---|---|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22 M | −0.043 | 51 |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | 33 M | −0.044 | 99 |
| mixedbread-ai/mxbai-rerank-xsmall-v1 | 71 M | −0.008 | 131 |
| BAAI/bge-reranker-base | 278 M | −0.013 | 287 |

Every one of them is negative. A 13× larger model does not fix it, it just costs 6× more.

## Was it the pool depth?

Rerankers are supposed to earn their keep on deep pools — retrieve 1,000 cheaply, let the
expensive model find what was buried. Tested on 50 queries with MiniLM-L6:

| reranked pool | Δ nDCG@10 vs same-pool RRF |
|---|---|
| RRF top-25 | −0.043 |
| RRF top-100 | −0.056 |
| BM25-only top-100 | −0.022 |

Deeper is *worse*. More candidates means more chances for a weak model to promote a false
positive into the top 10.

## Why

Two things compound:

1. **Domain shift.** `ms-marco-MiniLM` is trained on MS MARCO — short web queries against web
   passages. SciFact is scientific claim verification: the "query" is a declarative claim and the
   documents are paper abstracts. Nothing about that distribution matches. This is visible in the
   published BEIR tables too, where CE reranking of BM25 on SciFact lands around 0.688 — close to
   what we measure, and below our fused 0.720.
2. **The first stage is already strong.** RRF recall@10 is 0.854 and recall@100 is 0.941, so
   there are only ~9 points of headroom between rank 10 and rank 100. To exploit that the
   reranker would have to be nearly perfect; being merely decent means it shuffles a good
   ranking and loses.

## What to do with this for the talk

**Keep it, and teach it.** The demo still shows the cross-encoder mechanism perfectly — the rank
deltas, the 512-token truncation, and a latency bar that is three orders of magnitude taller than
BM25's. Adding "...and then we measured whether it was worth it, and it wasn't" is a lesson most
undergraduates never get: pipeline stages are hypotheses, not furniture, and the measurement is
the point. The eval table makes the argument on its own.

**If you want a stage-4 win on screen**, the dataset is the lever, not the model. TREC-COVID is
the standard case where CE reranking pays for itself (published BM25 ≈ 0.656 → CE ≈ 0.757),
because the queries are short and natural-language — much closer to the reranker's training
distribution. Cost: 171k documents, roughly 40 minutes of one-time embedding on this box and a
~250 MB HNSW graph. The loader needs no code change, only a fetch script entry.

## Reproducing

```bash
make bench                       # full query set, writes benchmark.json
make bench-fast                  # same without reranking (seconds, not minutes)
```

The model comparison used a throwaway harness; the candidate lists come straight from
`/api/search?rerank=false`, so any reranker can be scored against them without touching the
Java service.
