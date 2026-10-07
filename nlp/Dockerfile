FROM python:3.11-slim
LABEL org.opencontainers.image.title="search-demo nlp" \
      org.opencontainers.image.description="CPU embedding and reranking sidecar"

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Install torch before anything that depends on it. On amd64 the PyPI wheel drags in
# ~2.5 GB of CUDA packages we will never use, so take it from PyTorch's CPU index
# instead; the aarch64 wheels on PyPI are already CPU-only.
ARG TARGETARCH
ARG TORCH_VERSION=2.5.1
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" ; \
    else \
      pip install --no-cache-dir "torch==${TORCH_VERSION}" ; \
    fi

COPY nlp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights into the image so a container starts instantly and needs no network.
# If you override EMBED_MODEL or RERANK_MODEL at runtime, that model is downloaded on
# first use instead — rebuild with new build args to keep startup offline.
ENV HF_HOME=/opt/models
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}', device='cpu'); \
CrossEncoder('${RERANK_MODEL}', device='cpu', max_length=512)" \
 && chown -R app:app /opt/models

COPY nlp/app.py .

ENV EMBED_MODEL=${EMBED_MODEL} \
    RERANK_MODEL=${RERANK_MODEL} \
    RERANK_MAX_LEN=512 \
    TORCH_THREADS=4 \
    NLP_PORT=8081

EXPOSE 8081
USER app
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=6 \
  CMD curl -fsS "http://127.0.0.1:${NLP_PORT}/health" || exit 1

CMD ["sh","-c","exec uvicorn app:app --host 0.0.0.0 --port ${NLP_PORT}"]
