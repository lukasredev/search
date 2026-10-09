# Local (non-Docker) dev. Docker packaging is phase P7; see PLAN.md.
SHELL      := /bin/bash
JAVA_HOME  ?= $(HOME)/.local/opt/jdk21
MVN        := $(HOME)/.local/opt/maven/bin/mvn
JAVA       := $(JAVA_HOME)/bin/java
JAVA_FLAGS := --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
# 522k Quora vectors are ~800 MB of float[] before they reach Lucene.
INDEX_HEAP ?= -Xmx6g
JAR        := searchd/target/searchd.jar
# Long-running JVMs run from a snapshot of the jar, never the build output itself.
# maven-shade replaces target/searchd.jar on every `make build`, and a JVM that then
# needs a class it has not loaded yet dies with NoClassDefFoundError - which is how a
# 20-minute index build was lost once. Rebuilding while something runs is normal during
# a talk, so make it safe rather than a rule to remember.
RUNJAR     := searchd/target/searchd-running.jar
DATASET    ?= cqadupstack-unix
export JAVA_HOME

.PHONY: help data build index serve nlp bench clean reindex demos test test-live \
        docker-build docker-setup docker-up docker-down docker-logs \
        docker-bench docker-demos

help:
	@echo "make data            fetch + normalize every dataset"
	@echo "make build           compile the Java service"
	@echo "make index           build the Lucene index (needs nlp running)"
	@echo "make nlp             run the embedding/rerank sidecar on :8081"
	@echo "make serve           run the search API + UI on :8080"
	@echo "make bench           run the full query set, write benchmark.json"
	@echo "make reindex         rebuild the index reusing cached embeddings"
	@echo "make demos           regenerate curated example queries (TOPIC=--topic cs)"
	@echo "make test            unit + API tests, Java and Python (no models, no index)"
	@echo "make test-live       end-to-end checks against a running stack (URL=...)"
	@echo
	@echo "Docker:"
	@echo "  make docker-build    build both images"
	@echo "  make docker-setup    fetch the dataset and build the index in a volume"
	@echo "  make docker-up       start, make docker-down to stop"
	@echo
	@echo "DATASET=$(DATASET)   (cqadupstack-unix | scifact | nfcorpus | quora)"

data:
	python3 scripts/fetch_dataset.py

build:
	$(MVN) -q -B -f searchd/pom.xml package

$(JAR): build

nlp:
	cd nlp && TORCH_THREADS=4 ./run.sh

index: $(JAR)
	@cp -f $(JAR) $(RUNJAR)
	cd searchd && $(JAVA) $(JAVA_FLAGS) $(INDEX_HEAP) -cp target/searchd-running.jar \
	  ai.deepjudge.searchdemo.Indexer $(DATASET)

# Same as index; the embedding cache in data/$(DATASET)/vectors.f32 makes it fast.
reindex: index

serve: $(JAR)
	@cp -f $(JAR) $(RUNJAR)
	cd searchd && DATASET=$(DATASET) UI_DIR=$(CURDIR)/ui $(JAVA) $(JAVA_FLAGS) \
	  -cp target/searchd-running.jar ai.deepjudge.searchdemo.Main

bench: $(JAR)
	@cp -f $(JAR) $(RUNJAR)
	cd searchd && $(JAVA) $(JAVA_FLAGS) -cp target/searchd-running.jar \
	  ai.deepjudge.searchdemo.Benchmark $(DATASET) 99999 rerank

bench-fast: $(JAR)
	@cp -f $(JAR) $(RUNJAR)
	cd searchd && $(JAVA) $(JAVA_FLAGS) -cp target/searchd-running.jar \
	  ai.deepjudge.searchdemo.Benchmark $(DATASET) 99999 norerank

demos: ## regenerate the curated example queries (needs serve running)
	python3 scripts/find_demo_queries.py --dataset $(DATASET) $(TOPIC)

# Java against a fake sidecar and a throwaway index, Python against fake models. Fast,
# and needs neither the real models nor a built index.
PYTEST := nlp/.venv/bin/python -m pytest
test:
	$(MVN) -q -B -f searchd/pom.xml test
	@$(PYTEST) --version >/dev/null 2>&1 || uv pip install --python nlp/.venv/bin/python -q -r nlp/requirements-dev.txt
	$(PYTEST) -q nlp/tests scripts/tests

# The same checks a talk depends on, against the real models and index. Set AUTH_USER and
# AUTH_PASS as well if the service requires them.
URL ?= http://localhost:8080
test-live:
	SEARCHD_URL=$(URL) $(PYTEST) -q tests

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-setup:
	docker compose --profile setup run --rm fetch $(DATASET)
	DATASET=$(DATASET) docker compose --profile setup run --rm indexer $(DATASET)

docker-bench:
	DATASET=$(DATASET) docker compose --profile setup run --rm bench $(DATASET)

docker-demos:
	docker compose --profile setup run --rm demos --dataset $(DATASET) $(TOPIC)

docker-up:
	DATASET=$(DATASET) docker compose up -d
	@echo "http://localhost:$${PORT:-8080}"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f --tail=80

clean:
	$(MVN) -q -B -f searchd/pom.xml clean
