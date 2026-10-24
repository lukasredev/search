package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

@SuppressWarnings("unchecked")
class BenchmarkTest {

  private final TestIndex idx = TestIndex.get();

  @Test
  void averagesEveryStageOverTheJudgedQueries() throws Exception {
    var report = Benchmark.run(idx.pipeline(), idx.dataset, Integer.MAX_VALUE, true, 50);
    assertEquals(3, report.get("queries"));
    var perStage = (Map<String, Map<String, Object>>) report.get("perStage");
    assertEquals(List.of("bm25", "dense", "rrf", "rerank"), List.copyOf(perStage.keySet()));
    for (var m : perStage.values()) {
      assertEquals(3, m.get("queries"));
      double ndcg = (Double) m.get("ndcg@10");
      assertTrue(ndcg >= 0 && ndcg <= 1, m.toString());
      assertTrue(m.containsKey("avgMs"));
    }
  }

  @Test
  void averagesMatchThePerQueryMetrics() throws Exception {
    var report = Benchmark.run(idx.pipeline(), idx.dataset, 2, false, 50);
    assertEquals(2, report.get("queries"));
    double sum = 0;
    for (String q : List.of(TestIndex.Q_VITAMIN, TestIndex.Q_SHELL)) {
      var res = idx.pipeline().search(new SearchPipeline.Request(q, SearchPipeline.Mode.HYBRID,
          true, false, false, 1.2f, 0.75f, 100, 100, 100, 60, 50, 50));
      var eval = (Map<String, Object>) res.get("eval");
      sum += (Double) ((Map<String, Map<String, Object>>) eval.get("perStage")).get("rrf").get("ndcg@10");
    }
    var perStage = (Map<String, Map<String, Object>>) report.get("perStage");
    assertEquals(Evaluator.r4(sum / 2), perStage.get("rrf").get("ndcg@10"));
  }

  @Test
  void scoresEachQueryAgainstItsOwnJudgmentsEvenWhenTextsRepeat() throws Exception {
    // Quora has two judged queries with the same text. The second must be scored against
    // its own qrels (d6, which BM25 cannot find for this query), not the first one's.
    Path dir = java.nio.file.Files.createTempDirectory("dup").resolve(TestIndex.DATASET);
    TestIndex.writeDataset(dir);
    java.nio.file.Files.writeString(dir.resolve("queries.tsv"),
        "q1\t" + TestIndex.Q_VITAMIN + "\n" + "q4\t" + TestIndex.Q_VITAMIN + "\n");
    java.nio.file.Files.writeString(dir.resolve("qrels.tsv"), "q1\td1\t2\n" + "q4\td6\t1\n");
    Dataset dup = Dataset.load(dir);
    var pipeline = new SearchPipeline(idx.searcher, new DemoAnalyzer(), dup, idx.nlp.client());

    var single = (Map<String, Map<String, Object>>) ((Map<String, Object>) pipeline
        .search(SearchPipeline.Request.of(TestIndex.Q_VITAMIN), "q1").get("eval")).get("perStage");
    double q1 = (Double) single.get("bm25").get("ndcg@10");
    assertTrue(q1 > 0);

    var report = Benchmark.run(pipeline, dup, Integer.MAX_VALUE, false, 50);
    var perStage = (Map<String, Map<String, Object>>) report.get("perStage");
    assertEquals(Evaluator.r4(q1 / 2), perStage.get("bm25").get("ndcg@10"));
  }

  @Test
  void refusesToRecordDegradedRuns() throws Exception {
    try (FakeNlp down = new FakeNlp()) {
      down.failStatus = 500;
      var e = assertThrows(IllegalStateException.class,
          () -> Benchmark.run(idx.pipeline(down.client()), idx.dataset, 5, true, 50));
      assertTrue(e.getMessage().contains("degraded"));
    }
  }
}
