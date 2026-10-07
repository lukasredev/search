#!/bin/sh
# serve | index [dataset] | bench [dataset] — one image, three jobs.
set -eu

# The Vector API makes HNSW search several times faster; without --add-modules the JVM
# falls back to scalar code and only tells you in a warning.
VECTOR_FLAGS="--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED"

cmd="${1:-serve}"
if [ $# -gt 0 ]; then shift; fi

case "$cmd" in
  serve)
    exec java $VECTOR_FLAGS ${JAVA_OPTS:-} -cp /app/searchd.jar \
      ai.deepjudge.searchdemo.Main "$@"
    ;;
  index)
    # Building the HNSW graph holds every vector on the heap at once, so give the
    # indexer more room than the server needs.
    exec java $VECTOR_FLAGS ${INDEX_JAVA_OPTS:--XX:MaxRAMPercentage=80} \
      -cp /app/searchd.jar ai.deepjudge.searchdemo.Indexer "${1:-$DATASET}"
    ;;
  bench)
    exec java $VECTOR_FLAGS ${JAVA_OPTS:-} -cp /app/searchd.jar \
      ai.deepjudge.searchdemo.Benchmark "${1:-$DATASET}" "${2:-99999}" "${3:-rerank}"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac
