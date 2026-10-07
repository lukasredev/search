"""The two transformer models the search engine calls out to.

Kept deliberately thin: it loads a bi-encoder and a cross-encoder and exposes one
endpoint each. All the search logic lives in the Java service — this is just "the
ML models", which is exactly the boundary we want to draw during the talk.
"""
import os
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_LEN = int(os.environ.get("RERANK_MAX_LEN", "512"))

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

app = FastAPI(title="search-demo nlp")
_models: dict[str, object] = {}


def embedder() -> SentenceTransformer:
    if "embed" not in _models:
        print(f"loading {EMBED_MODEL}", flush=True)
        _models["embed"] = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _models["embed"]


def reranker() -> CrossEncoder:
    if "rerank" not in _models:
        print(f"loading {RERANK_MODEL}", flush=True)
        _models["rerank"] = CrossEncoder(RERANK_MODEL, max_length=MAX_LEN, device="cpu")
    return _models["rerank"]


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    model: str
    dim: int
    ms: float
    vectors: list[list[float]]


class RerankRequest(BaseModel):
    query: str
    docs: list[str]


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
    items = []
    for doc, score in zip(req.docs, scores):
        n = len(tok(doc, add_special_tokens=False)["input_ids"])
        items.append(RerankItem(score=float(score), doc_tokens=n, truncated=n > budget))

    return RerankResponse(model=RERANK_MODEL, max_length=MAX_LEN, ms=round(ms, 2), items=items)
