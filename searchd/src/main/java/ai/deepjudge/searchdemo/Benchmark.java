package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.FSDirectory;

/**
 * Runs the whole query set and averages the metrics after each stage.
 *
 * <p>This is the number that makes the talk land: one table, four rows, nDCG climbing as
 * stages are added. It is also how we check the implementation is not quietly broken —
 * BM25 on SciFact should land near the published BEIR baseline of about 0.665 nDCG@10.
 *
 * <pre>
 *   mvn exec:java -Dexec.mainClass=ai.deepjudge.searchdemo.Benchmark -Dexec.args="scifact"
 * </pre>
 */
public final class Benchmark {

  public static void main(String[] args) throws Exception {
    String dataset = args.length > 0 ? args[0] : Config.dataset();
    int limit = args.length > 1 ? Integer.parseInt(args[1]) : Integer.MAX_VALUE;
    boolean rerank = args.length <= 2 || !"norerank".equals(args[2]);
    int rerankN = args.length > 3 ? Integer.parseInt(args[3]) : 50;

    Path dataDir = Config.dataRoot().resolve(dataset);
    Path indexDir = Config.indexRoot().resolve(dataset);

    try (var dir = FSDirectory.open(indexDir); var reader = DirectoryReader.open(dir)) {
      Dataset ds = Dataset.load(dataDir);
      var searcher = new IndexSearcher(reader);
      var pipeline = new SearchPipeline(searcher, new DemoAnalyzer(), ds,
          new NlpClient(Config.nlpUrl(), java.time.Duration.ofMinutes(5)));

      Map<String, Object> report = run(pipeline, ds, limit, rerank, rerankN);
      System.out.println();
      print(report);
      Path out = indexDir.resolve("benchmark.json");
      if (rerank && rerankN != 50) out = indexDir.resolve("benchmark-rerank" + rerankN + ".json");
      if (!rerank) out = indexDir.resolve("benchmark-norerank.json");
      java.nio.file.Files.writeString(out, new com.fasterxml.jackson.databind.ObjectMapper()
          .writerWithDefaultPrettyPrinter().writeValueAsString(report));
      System.out.println("\nwrote " + out);
    }
  }

  /** Averages each metric over every judged query, per stage. */
  public static Map<String, Object> run(SearchPipeline pipeline, Dataset ds, int limit,
                                        boolean rerank, int rerankN) throws IOException {
    // stage -> metric -> running sum
    Map<String, Map<String, Double>> sums = new LinkedHashMap<>();
    Map<String, Integer> counts = new LinkedHashMap<>();
    Map<String, Double> stageMs = new LinkedHashMap<>();

    List<Map.Entry<String, String>> queries =
        new ArrayList<>(ds.queries().entrySet()).subList(0, Math.min(limit, ds.queries().size()));

    long t0 = System.nanoTime();
    int done = 0;
    for (Map.Entry<String, String> q : queries) {
      SearchPipeline.Request req = new SearchPipeline.Request(
          q.getValue(), SearchPipeline.Mode.HYBRID,
          true,      // trace on, so we can read per-stage timings
          false,     // explain off: it is expensive and we only want metrics here
          rerank,
          1.2f, 0.75f, 100, 100, 100, RrfFusion.DEFAULT_K, 50, rerankN);

      Map<String, Object> res = pipeline.search(req);

      @SuppressWarnings("unchecked")
      Map<String, Object> eval = (Map<String, Object>) res.get("eval");
      if (Boolean.TRUE.equals(eval.get("judged"))) {
        @SuppressWarnings("unchecked")
        Map<String, Map<String, Object>> perStage =
            (Map<String, Map<String, Object>>) eval.get("perStage");
        perStage.forEach((stage, metrics) -> {
          Map<String, Double> acc = sums.computeIfAbsent(stage, s -> new TreeMap<>());
          metrics.forEach((metric, value) -> {
            if (value instanceof Number n && !metric.equals("retrieved") && !metric.equals("relevantTotal")) {
              acc.merge(metric, n.doubleValue(), Double::sum);
            }
          });
          counts.merge(stage, 1, Integer::sum);
        });
      }

      // In-process these are still Trace.Stage records; only the HTTP layer turns
      // them into JSON objects.
      if (res.get("stages") instanceof List<?> stages) {
        for (Object s : stages) {
          Trace.Stage stage = (Trace.Stage) s;
          stageMs.merge(stage.name(), stage.ms(), Double::sum);
        }
      }

      if (++done % 25 == 0 || done == queries.size()) {
        double elapsed = (System.nanoTime() - t0) / 1e9;
        System.out.printf("\r  %d/%d queries  (%.1fs, %.2fs/query)   ",
            done, queries.size(), elapsed, elapsed / done);
      }
    }

    Map<String, Object> perStage = new LinkedHashMap<>();
    sums.forEach((stage, acc) -> {
      int n = counts.get(stage);
      Map<String, Object> avg = new TreeMap<>();
      acc.forEach((metric, sum) -> avg.put(metric, Evaluator.r4(sum / n)));
      avg.put("queries", n);
      avg.put("avgMs", Trace.round(stageMs.getOrDefault(stage, 0.0) / n));
      perStage.put(stage, avg);
    });

    return Trace.map(
        "dataset", ds.name(),
        "queries", queries.size(),
        "rerank", rerank,
        "rerankN", rerankN,
        "totalSeconds", Trace.round((System.nanoTime() - t0) / 1e9),
        "perStage", perStage);
  }

  private static void print(Map<String, Object> report) {
    System.out.printf("%s — %s queries, rerankN=%s%n",
        report.get("dataset"), report.get("queries"), report.get("rerankN"));
    System.out.printf("%-8s %9s %10s %11s %8s %9s%n",
        "stage", "nDCG@10", "recall@10", "recall@100", "MRR", "avg ms");
    System.out.println("-".repeat(60));
    @SuppressWarnings("unchecked")
    Map<String, Map<String, Object>> perStage =
        (Map<String, Map<String, Object>>) report.get("perStage");
    perStage.forEach((stage, m) -> System.out.printf("%-8s %9s %10s %11s %8s %9s%n",
        stage, m.get("ndcg@10"), m.get("recall@10"), m.get("recall@100"), m.get("mrr"), m.get("avgMs")));
  }

  private Benchmark() {}
}
