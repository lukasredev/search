# ── build ────────────────────────────────────────────────────────────────────
# The jar is platform-independent, so compile natively even when building for another
# architecture: Maven under QEMU emulation is many times slower.
# CI pulls Docker Hub images through mirror.gcr.io, see .github/workflows/images.yml.
FROM --platform=$BUILDPLATFORM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /src

# Dependencies first, so editing a source file does not re-download the world.
COPY searchd/pom.xml ./pom.xml
RUN mvn -B -q dependency:go-offline -DskipTests || true

COPY searchd/src ./src
RUN mvn -B -q package -DskipTests

# ── runtime ──────────────────────────────────────────────────────────────────
FROM eclipse-temurin:21-jre
# The commit this image was built from; /api/health reports it and the UI footer shows it.
# Declared after the build stage so a new commit does not invalidate the Maven layers.
ARG GIT_SHA=""
LABEL org.opencontainers.image.title="search-demo searchd" \
      org.opencontainers.image.description="Lucene index, retrieval pipeline and UI" \
      org.opencontainers.image.revision="${GIT_SHA}"

# curl is only here so the container can report its own health to compose.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=build /src/target/searchd.jar /app/searchd.jar
COPY ui /app/ui
# The hand-picked example queries and benchmark results for each dataset. The data
# volume starts empty, so the indexer copies these next to the index it builds.
COPY data/index /app/defaults
COPY searchd/docker-entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint

ENV DATA_ROOT=/data \
    INDEX_ROOT=/data/index \
    UI_DIR=/app/ui \
    GIT_SHA=${GIT_SHA} \
    DATASET=cqadupstack-unix \
    NLP_URL=http://nlp:8081 \
    PORT=8080 \
    LISTEN_HOST=0.0.0.0 \
    JAVA_OPTS="-XX:MaxRAMPercentage=70"

EXPOSE 8080
# Numeric, so Kubernetes can enforce runAsNonRoot; uid 10001 is the "app" user above.
USER 10001
# Kubernetes ignores this and uses its own probes: /livez and /readyz, see docs/deploy.md.
HEALTHCHECK --interval=15s --timeout=4s --start-period=40s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/readyz" || exit 1

ENTRYPOINT ["entrypoint"]
CMD ["serve"]
