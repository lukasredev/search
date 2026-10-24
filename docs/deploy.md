# Running it on a server

Two containers. `searchd` holds the Lucene index, the retrieval pipeline and the UI;
`nlp` holds the two transformer models. Datasets and the index live in one named volume,
so rebuilding images never costs you a re-index.

## First run

```bash
# 1. download and normalise a dataset into the volume (~50 MB, no auth)
docker compose --profile setup run --rm fetch cqadupstack-unix

# 2. build the index. Embeds the corpus, so this is the slow step:
#    cqadupstack-unix ~15 min, scifact ~2 min, nfcorpus ~1.5 min, quora ~25 min.
docker compose --profile setup run --rm indexer cqadupstack-unix

# 3. start
docker compose up -d
```

Then open <http://localhost:8080>.

Swap datasets with `DATASET`, which every service reads:

```bash
docker compose --profile setup run --rm fetch quora
DATASET=quora docker compose --profile setup run --rm indexer quora
DATASET=quora docker compose up -d
```

Optional, and worth doing before a talk — both write into the volume and are picked up by
the running service without a restart:

```bash
# per-stage metrics over the whole query set, shown in the last chapter
docker compose --profile setup run --rm bench cqadupstack-unix

# the curated example queries in the UI (needs the service up)
docker compose --profile setup run --rm demos --dataset cqadupstack-unix
```

## Exposing it

By default the port binds to `127.0.0.1` only, so nothing is reachable from outside the
host until you say so. To put it on the internet:

```bash
# .env
BIND=0.0.0.0
AUTH_USER=talk
AUTH_PASS=<something long>
```

`AUTH_USER` + `AUTH_PASS` turn on HTTP basic auth over the whole service, compared in
constant time. Keep the password in `.env`, which is git-ignored; the `demos` job reads the
same two variables. Basic auth is enough to stop a crawler discovering `/api/search` and
burning your CPU on cross-encoder inference; it is **not** enough on its own, because it
sends the password in cleartext. Put TLS in front of the service — Caddy needs two lines:

```caddyfile
search.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

The request knobs are clamped server-side regardless (`rerank_n` ≤ 200, candidates ≤ 1000,
postings from `/api/term` ≤ 1000, query terms ≤ 1024), so no query can ask for unbounded work.

## Sizing

| | memory | notes |
|---|---|---|
| `nlp` | 1.5–3 GB | both models on CPU; `mem_limit` defaults to 3g |
| `searchd` | 1–2 GB | scales with index size, not corpus size |
| `indexer` | 6 GB | **the peak**, and only while indexing |

The indexer holds every vector on the heap at once before Lucene writes the graph — 522k
Quora vectors are about 800 MB of `float[]`. On a small server, index on a bigger machine
(or locally) and ship the volume, rather than renting 8 GB you need for 25 minutes a year.

A 2 vCPU / 4 GB instance runs SciFact comfortably. Reranking is the only slow part: about
55 ms per document on CPU, so a 50-document rerank is ~3 s. `TORCH_THREADS` defaults to 4;
set it to your core count.

Disk: images are roughly 300 MB (`searchd`) and 1.5–2 GB (`nlp`, mostly torch and the baked
model weights). The volume holds the dataset, the cached embeddings and the index — 85 MB
for SciFact, about 2.5 GB for Quora.

## Architecture

Build on the same architecture you deploy to, or build explicitly:

```bash
docker buildx build --platform linux/amd64 -f searchd/Dockerfile -t search-demo/searchd .
```

The `nlp` image picks its torch wheel from `TARGETARCH`: on `amd64` it takes the CPU-only
build from PyTorch's own index, because the PyPI wheel drags in ~2.5 GB of CUDA packages
that will never be used. On `arm64` the PyPI wheel is already CPU-only.

Prebuilt multi-arch (`amd64` + `arm64`) images are published by
[`.github/workflows/images.yml`](../.github/workflows/images.yml) to
`ghcr.io/<owner>/<repo>/searchd` and `ghcr.io/<owner>/<repo>/nlp`: `:latest` tracks the
default branch, `:sha-<short>` pins a commit, and a `v1.2.3` tag publishes `:1.2.3`.
The same workflow runs `make test` first and builds nothing if it fails.

## Checking it works

```bash
docker compose config                       # compose file is valid
docker compose build                        # the real test
docker compose --profile setup run --rm fetch cqadupstack-unix
docker compose --profile setup run --rm indexer cqadupstack-unix
docker compose up -d
curl -fsS localhost:8080/api/health | jq    # {"ok":true,"docs":47382,"nlp":{"ok":true}}
curl -fsS "localhost:8080/api/search?q=vitamin+d&rerank=false" | jq '.stages[].name'
```

`/api/health` reports the sidecar's status too, so one call tells you whether both halves
are up. If `searchd` exits immediately it will say so: an empty volume produces a message
telling you to run the indexer, not a stack trace.

## Kubernetes

The two images run unchanged on Kubernetes. Compose's `HEALTHCHECK`s are ignored there;
use these probes instead, on both containers (`searchd` on 8080, `nlp` on 8081):

| path | means | fails when |
|---|---|---|
| `/livez` | restart me | `searchd`: a request has run longer than `STUCK_REQUEST_SECONDS` (600). `nlp`: the process is not answering |
| `/readyz` | send me traffic | `searchd`: the index cannot be searched. `nlp`: models still loading (a few seconds) |

```yaml
livenessProbe:  { httpGet: { path: /livez,  port: 8080 }, periodSeconds: 20, failureThreshold: 3 }
readinessProbe: { httpGet: { path: /readyz, port: 8080 }, periodSeconds: 5 }
```

Neither `searchd` probe depends on `nlp`: without it, search degrades to BM25 with a warning,
so an `nlp` outage should not take `searchd` out of service or restart it. Both probe paths
skip basic auth, so they keep working with `AUTH_USER`/`AUTH_PASS` set.

The stuck-request check exists because of one known failure: if a request handler and its
exception handler both throw, Javalin 6.4 retries forever and the thread spins at 100% CPU,
while every endpoint, `/readyz` included, keeps answering. Liveness restarts the pod instead.

What else the images expect:

- **Data** — one `ReadWriteOnce` volume at `/data`, mounted read-only in `searchd`. Populate
  it with two Jobs running the same commands as the compose `fetch` and `indexer` services
  (the indexer is `entrypoint index <dataset>` in the `searchd` image). The indexer also
  copies the hand-picked example queries and benchmark results into the index directory.
- **Security** — both images run as uid 10001, so `runAsNonRoot: true` works. Set
  `fsGroup: 10001` so the volume is writable by the indexer; the compose `fetch` job does a
  `chown` instead.
- **Shutdown** — on SIGTERM, `searchd` stops accepting connections and waits up to
  `SHUTDOWN_GRACE_SECONDS` (20) for in-flight searches; keep `terminationGracePeriodSeconds`
  above that. A short `preStop` sleep (5 s) lets the endpoint be removed before it stops.
- **Resources** — `searchd` sizes its heap from the container limit (`MaxRAMPercentage=70`),
  so set a memory limit. `nlp` is CPU-bound: a rerank takes 3–4 s on 4 cores, so give it
  `TORCH_THREADS` worth of CPU and scale it out rather than up.
- **Network** — `nlp` has no authentication, so nothing but `searchd` should reach it.
  Point `searchd` at it with `NLP_URL` and lock it down with a NetworkPolicy (assuming
  pods labelled `app: nlp` and `app: searchd`; the indexer and bench Jobs need access too):

  ```yaml
  apiVersion: networking.k8s.io/v1
  kind: NetworkPolicy
  metadata: { name: nlp-from-searchd-only }
  spec:
    podSelector: { matchLabels: { app: nlp } }
    policyTypes: [Ingress]
    ingress:
      - from: [{ podSelector: { matchLabels: { app: searchd } } }]
        ports: [{ port: 8081 }]
  ```

## Changing models

Model weights are baked into the `nlp` image at build time so containers start instantly
and need no network. Each model is pinned to a Hugging Face commit (`EMBED_REVISION`,
`RERANK_REVISION`), so an upstream update cannot change results between two builds.
Changing the model therefore means a rebuild with a matching revision — the default
revisions are commits of the default models, and an empty revision means "latest":

```bash
docker compose build --build-arg RERANK_MODEL=BAAI/bge-reranker-base \
                     --build-arg RERANK_REVISION= nlp
```

Setting `RERANK_MODEL` at runtime alone still works, but the model is then downloaded on
first use, and the first request pays for it. Note that the embedding model is baked into
the *index* as well — changing `EMBED_MODEL` means re-running the indexer, since the
stored vectors came from the old model and comparing across models is meaningless. The
indexer notices: `vectors.f32.json` records which model filled the embedding cache, and a
different model re-embeds the corpus instead of reusing it.
