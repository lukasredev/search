# ── build ────────────────────────────────────────────────────────────────────
FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /src

# Dependencies first, so editing a source file does not re-download the world.
COPY searchd/pom.xml ./pom.xml
RUN mvn -B -q dependency:go-offline -DskipTests || true

COPY searchd/src ./src
RUN mvn -B -q package -DskipTests

# ── runtime ──────────────────────────────────────────────────────────────────
FROM eclipse-temurin:21-jre
LABEL org.opencontainers.image.title="search-demo searchd" \
      org.opencontainers.image.description="Lucene index, retrieval pipeline and UI"

# curl is only here so the container can report its own health to compose.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=build /src/target/searchd.jar /app/searchd.jar
COPY ui /app/ui
COPY searchd/docker-entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint

ENV DATA_ROOT=/data \
    INDEX_ROOT=/data/index \
    UI_DIR=/app/ui \
    DATASET=cqadupstack-unix \
    NLP_URL=http://nlp:8081 \
    PORT=8080 \
    JAVA_OPTS="-XX:MaxRAMPercentage=70"

EXPOSE 8080
USER app
HEALTHCHECK --interval=15s --timeout=4s --start-period=40s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

ENTRYPOINT ["entrypoint"]
CMD ["serve"]
