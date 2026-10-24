# Base images come from AWS's mirror of the Docker Hub official images: CI runners share
# IPs, so anonymous Docker Hub pulls hit its rate limit and fail the build.
FROM public.ecr.aws/docker/library/python:3.14-slim
LABEL org.opencontainers.image.title="search-demo nlp" \
      org.opencontainers.image.description="CPU embedding and reranking sidecar"

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Install torch before anything that depends on it. The PyPI wheels drag in ~3 GB of
# CUDA packages we will never use, on aarch64 too by torch 2.14, so take it from
# PyTorch's CPU index, which has wheels for both architectures. The version comes from
# requirements.txt, so the next step finds it satisfied and a Dependabot bump of that
# one pin cannot leave the two disagreeing.
COPY nlp/requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch==$(sed -n 's/^torch==//p' requirements.txt)" \
 && pip install --no-cache-dir -r requirements.txt

# Bake the weights into the image so a container starts instantly and needs no network.
# If you override EMBED_MODEL or RERANK_MODEL at runtime, that model is downloaded on
# first use instead — rebuild with new build args to keep startup offline. Override the
# matching *_REVISION too: the defaults are commits of the default models.
ENV HF_HOME=/opt/models
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Pinned hub commits of the two models above. Change them together with the model.
ARG EMBED_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ARG RERANK_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}', revision='${EMBED_REVISION}' or None, device='cpu'); \
CrossEncoder('${RERANK_MODEL}', revision='${RERANK_REVISION}' or None, device='cpu', max_length=512)" \
 && chown -R app:app /opt/models

COPY nlp/app.py .

ENV EMBED_MODEL=${EMBED_MODEL} \
    RERANK_MODEL=${RERANK_MODEL} \
    EMBED_REVISION=${EMBED_REVISION} \
    RERANK_REVISION=${RERANK_REVISION} \
    RERANK_MAX_LEN=512 \
    TORCH_THREADS=4 \
    NLP_PORT=8081

EXPOSE 8081
# Numeric, so Kubernetes can enforce runAsNonRoot; uid 10001 is the "app" user above.
USER 10001
# Kubernetes ignores this and uses its own probes: /livez and /readyz, see docs/deploy.md.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=6 \
  CMD curl -fsS "http://127.0.0.1:${NLP_PORT}/readyz" || exit 1

CMD ["sh","-c","exec uvicorn app:app --host 0.0.0.0 --port ${NLP_PORT}"]
