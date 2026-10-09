"""The two transformer models the search engine calls out to.

Kept deliberately thin: it loads a bi-encoder and a cross-encoder and exposes one
endpoint each. All the search logic lives in the Java service — this is just "the
ML models", which is exactly the boundary we want to draw during the talk.
"""
import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Hub commits, so a model update upstream cannot change results under us. Empty means
# the latest revision, for trying a model before pinning it.
EMBED_REVISION = os.environ.get("EMBED_REVISION") or None
RERANK_REVISION = os.environ.get("RERANK_REVISION") or None
MAX_LEN = int(os.environ.get("RERANK_MAX_LEN", "512"))

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

_models: dict[str, object] = {}
_load_lock = threading.Lock()
_ready = threading.Event()


def embedder() -> SentenceTransformer:
    with _load_lock:
        if "embed" not in _models:
            print(f"loading {EMBED_MODEL}", flush=True)
            _models["embed"] = SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION, device="cpu")
    return _models["embed"]


def reranker() -> CrossEncoder:
    with _load_lock:
        if "rerank" not in _models:
            print(f"loading {RERANK_MODEL}", flush=True)
            _models["rerank"] = CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, max_length=MAX_LEN,
                                          device="cpu")
    return _models["rerank"]


def _load_and_warm() -> None:
    # In the background, so /livez answers while the weights load and only /readyz waits.
    embedder().encode(["warmup"], normalize_embeddings=True)
    reranker().predict([("warmup", "warmup")])
    _ready.set()
    print("models loaded, ready", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_load_and_warm, name="load-models", daemon=True).start()
    yield


app = FastAPI(title="search-demo nlp", lifespan=lifespan)


# Bounds on one request. The sidecar has no auth, so it should not take unbounded work
# from whoever can reach it; searchd sends one query or an EMBED_BATCH (256) of docs, and
# reranks at most 200. An empty list is a client error, not a crash in the model.
MAX_EMBED_TEXTS = 4096
MAX_RERANK_DOCS = 1000


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_EMBED_TEXTS)


class EmbedResponse(BaseModel):
    model: str
    dim: int
    ms: float
    vectors: list[list[float]]


class RerankRequest(BaseModel):
    query: str
    docs: list[str] = Field(min_length=1, max_length=MAX_RERANK_DOCS)


class RerankItem(BaseModel):
    score: float
    # How many of the doc's tokens actually reached the model. The 512-token
    # truncation is a real gotcha and worth showing in the UI.
    doc_tokens: int
    truncated: bool


class RerankResponse(BaseModel):
    model: str
    max_length: int
    ms: float
    items: list[RerankItem]


# Kubernetes probes. Async on purpose: they run on the event loop, so a pod busy embedding
# in the thread pool still answers them instead of looking dead and being restarted.
@app.get("/livez")
async def livez() -> dict:
    return {"ok": True}


@app.get("/readyz")
async def readyz(response: Response) -> dict:
    if not _ready.is_set():
        response.status_code = 503
    return {"ok": _ready.is_set(), "loaded": sorted(_models)}


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "loaded": sorted(_models),
    }


@app.post("/warmup")
def warmup() -> dict:
    t0 = time.perf_counter()
    embedder().encode(["warmup"], normalize_embeddings=True)
    reranker().predict([("warmup", "warmup")])
    return {"ok": True, "ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    m = embedder()
    t0 = time.perf_counter()
    # Normalized vectors mean Lucene's COSINE similarity is a plain dot product.
    vecs = m.encode(req.texts, normalize_embeddings=True, batch_size=32,
                    show_progress_bar=False, convert_to_numpy=True)
    ms = (time.perf_counter() - t0) * 1000
    return EmbedResponse(model=EMBED_MODEL, dim=int(vecs.shape[1]), ms=round(ms, 2),
                         vectors=vecs.tolist())


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    m = reranker()
    pairs = [(req.query, d) for d in req.docs]
    t0 = time.perf_counter()
    scores = m.predict(pairs, batch_size=16, show_progress_bar=False)
    ms = (time.perf_counter() - t0) * 1000

    # Count the tokens of the doc half only, so the UI can report what got cut.
    tok = m.tokenizer
    q_len = len(tok(req.query, add_special_tokens=False)["input_ids"])
    budget = MAX_LEN - q_len - 3  # [CLS] q [SEP] d [SEP]
    # One batched call: the fast tokenizers do the whole list in Rust at once.
    doc_ids = tok(req.docs, add_special_tokens=False)["input_ids"]
    items = []
    for ids, score in zip(doc_ids, scores):
        n = len(ids)
        items.append(RerankItem(score=float(score), doc_tokens=n, truncated=n > budget))

    return RerankResponse(model=RERANK_MODEL, max_length=MAX_LEN, ms=round(ms, 2), items=items)
