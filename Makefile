# Local (non-Docker) dev. Docker packaging is phase P7; see PLAN.md.
SHELL      := /bin/bash
JAVA_HOME  ?= $(HOME)/.local/opt/jdk21
MVN        := $(HOME)/.local/opt/maven/bin/mvn
JAVA       := $(JAVA_HOME)/bin/java
JAVA_FLAGS := --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
JAR        := searchd/target/searchd.jar
DATASET    ?= scifact
export JAVA_HOME

.PHONY: help data build index serve nlp bench clean reindex

help:
	@echo "make data            fetch + normalize scifact and nfcorpus"
	@echo "make build           compile the Java service"
	@echo "make index           build the Lucene index (needs nlp running)"
	@echo "make nlp             run the embedding/rerank sidecar on :8081"
	@echo "make serve           run the search API + UI on :8080"
	@echo "make bench           run the full query set, write benchmark.json"
	@echo "make reindex         rebuild the index reusing cached embeddings"
	@echo
	@echo "DATASET=$(DATASET)   (scifact | nfcorpus)"

data:
	python3 scripts/fetch_dataset.py

build:
	$(MVN) -q -B -f searchd/pom.xml package

$(JAR): build

nlp:
	cd nlp && TORCH_THREADS=4 ./run.sh

index: $(JAR)
	cd searchd && $(JAVA) $(JAVA_FLAGS) -cp target/searchd.jar \
	  ai.deepjudge.searchdemo.Indexer $(DATASET)

# Same as index; the embedding cache in data/$(DATASET)/vectors.f32 makes it fast.
reindex: index

serve: $(JAR)
	cd searchd && DATASET=$(DATASET) UI_DIR=$(CURDIR)/ui $(JAVA) $(JAVA_FLAGS) \
	  -cp target/searchd.jar ai.deepjudge.searchdemo.Main

bench: $(JAR)
	cd searchd && $(JAVA) $(JAVA_FLAGS) -cp target/searchd.jar \
	  ai.deepjudge.searchdemo.Benchmark $(DATASET) 99999 rerank

bench-fast: $(JAR)
	cd searchd && $(JAVA) $(JAVA_FLAGS) -cp target/searchd.jar \
	  ai.deepjudge.searchdemo.Benchmark $(DATASET) 99999 norerank

clean:
	$(MVN) -q -B -f searchd/pom.xml clean
