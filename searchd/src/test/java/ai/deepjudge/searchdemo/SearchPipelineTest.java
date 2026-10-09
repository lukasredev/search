package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import ai.deepjudge.searchdemo.SearchPipeline.Mode;
import ai.deepjudge.searchdemo.SearchPipeline.Request;

@SuppressWarnings("unchecked")
class SearchPipelineTest {

  private final TestIndex idx = TestIndex.get();

  private static List<String> stageNames(Map<String, Object> res) {
    return ((List<Trace.Stage>) res.get("stages")).stream().map(Trace.Stage::name).toList();
  }

  private static Trace.Stage stage(Map<String, Object> res, String name) {
    return ((List<Trace.Stage>) res.get("stages")).stream()
        .filter(s -> s.name().equals(name)).findFirst().orElseThrow();
  }

  private static Request with(String q, Mode mode, boolean rerank) {
    Request d = Request.of(q);
    return new Request(q, mode, true, true, rerank, d.k1(), d.b(), d.bm25N(), d.denseN(),
        d.efSearch(), d.rrfK(), d.fuseN(), d.rerankN());
  }

  @Test
  void hybridRunsEveryStageInOrder() throws Exception {
    var res = idx.pipeline().search(Request.of(TestIndex.Q_VITAMIN));
    assertEquals(List.of("analyze", "bm25", "dense", "rrf", "rerank"), stageNames(res));
    assertEquals("hybrid", res.get("mode"));
    assertEquals(TestIndex.DATASET, res.get("dataset"));
    assertNull(res.get("warnings"));
    assertTrue(res.get("totalMs") instanceof Double);

    // Every doc referenced by any stage is in the docs map, exactly once.
    var docs = (Map<String, Map<String, Object>>) res.get("docs");
    for (String s : List.of("bm25", "dense", "rrf", "rerank")) {
      for (var r : stage(res, s).results()) assertTrue(docs.containsKey((String) r.get("docId")));
    }
    assertEquals(2, docs.get("d1").get("relevance"));
  }

  @Test
  void judgedQueriesGetPerStageMetrics() throws Exception {
    var res = idx.pipeline().search(Request.of("Vitamin D Deficiency"));
    assertEquals("q1", res.get("qid"));
    var eval = (Map<String, Object>) res.get("eval");
    assertEquals(true, eval.get("judged"));
    assertEquals(2, eval.get("relevantTotal"));
    var perStage = (Map<String, Map<String, Object>>) eval.get("perStage");
    assertEquals(List.of("bm25", "dense", "rrf", "rerank"), List.copyOf(perStage.keySet()));
    // The word-overlap reranker puts d1 (all three query words) first.
    assertEquals(1.0, perStage.get("rerank").get("mrr"));
    assertEquals(1.0, perStage.get("bm25").get("recall@10"));
  }

  @Test
  void unjudgedQueriesSaySo() throws Exception {
    var res = idx.pipeline().search(Request.of("bones"));
    assertNull(res.get("qid"));
    assertEquals(false, ((Map<String, Object>) res.get("eval")).get("judged"));
  }

  @Test
  void singleRetrieverModesSkipFusion() throws Exception {
    var bm25 = idx.pipeline().search(with(TestIndex.Q_VITAMIN, Mode.BM25_ONLY, false));
    assertEquals(List.of("analyze", "bm25"), stageNames(bm25));
    var perStage = (Map<String, Object>) ((Map<String, Object>) bm25.get("eval")).get("perStage");
    assertEquals(List.of("bm25", "rrf", "rerank"), List.copyOf(perStage.keySet()));
    // With nothing to fuse, "rrf" is the BM25 ranking carried through unchanged.
    assertEquals(perStage.get("bm25"), perStage.get("rrf"));

    var dense = idx.pipeline().search(with(TestIndex.Q_VITAMIN, Mode.DENSE_ONLY, true));
    assertEquals(List.of("analyze", "dense", "rerank"), stageNames(dense));
  }

  @Test
  void fuseNCapsTheSingleRetrieverPathToo() throws Exception {
    Request d = Request.of("vitamin bone shell");
    var res = idx.pipeline().search(new Request(d.query(), Mode.BM25_ONLY, false, false, false,
        d.k1(), d.b(), 100, 100, 100, 60, 2, 50));
    assertEquals(2, ((List<?>) res.get("results")).size());
  }

  @Test
  void queriesThatAnalyzeToNothingStillGetDenseResults() throws Exception {
    // The README example: "$@ and $*" vanishes in the analyzer, but the embedding sees it.
    var res = idx.pipeline().search(with("$@ $*", Mode.HYBRID, false));
    assertEquals(List.of("analyze", "bm25", "dense"), stageNames(res));
    assertEquals(List.of(), stage(res, "bm25").results());
    assertFalse(stage(res, "dense").results().isEmpty());
  }

  @Test
  void traceOffReturnsOnlyTheTopTen() throws Exception {
    Request d = Request.of(TestIndex.Q_VITAMIN);
    var res = idx.pipeline().search(new Request(d.query(), Mode.HYBRID, false, false, true,
        d.k1(), d.b(), d.bm25N(), d.denseN(), d.efSearch(), d.rrfK(), d.fuseN(), d.rerankN()));
    assertNull(res.get("stages"));
    var results = (List<Map<String, Object>>) res.get("results");
    assertTrue(results.size() <= 10 && !results.isEmpty());
    assertEquals("d1", results.get(0).get("docId"));
    assertEquals(List.of("docId", "rank", "score", "scoredBy"), List.copyOf(results.get(0).keySet()));
    assertEquals("rerank", results.get(0).get("scoredBy"));
  }

  @Test
  @SuppressWarnings("unchecked")
  void traceOffSaysWhichStageScoredEachResult() throws Exception {
    Request d = Request.of(TestIndex.Q_VITAMIN);
    var res = idx.pipeline().search(new Request(d.query(), Mode.HYBRID, false, false, true,
        d.k1(), d.b(), d.bm25N(), d.denseN(), d.efSearch(), d.rrfK(), d.fuseN(), 1));
    var results = (List<Map<String, Object>>) res.get("results");
    assertTrue(results.size() > 1);
    assertEquals("rerank", results.get(0).get("scoredBy"));
    assertTrue(results.subList(1, results.size()).stream()
        .allMatch(r -> "rrf".equals(r.get("scoredBy"))));
  }

  @Test
  void aDeadSidecarDegradesToLexicalWithWarnings() throws Exception {
    try (FakeNlp down = new FakeNlp()) {
      down.failStatus = 500;
      var res = idx.pipeline(down.client()).search(Request.of(TestIndex.Q_VITAMIN));
      assertEquals(List.of("analyze", "bm25"), stageNames(res));
      var warnings = (List<String>) res.get("warnings");
      assertEquals(2, warnings.size());
      assertTrue(warnings.get(0).startsWith("dense retrieval unavailable"), warnings.get(0));
      assertTrue(warnings.get(1).startsWith("reranking unavailable"), warnings.get(1));
      assertEquals("d1", stage(res, "bm25").results().get(0).get("docId"));
    }
  }

  @Test
  void anUnreachableSidecarDegradesTheSameWay() throws Exception {
    FakeNlp gone = new FakeNlp();
    NlpClient client = gone.client();
    gone.close();
    var res = idx.pipeline(client).search(Request.of(TestIndex.Q_VITAMIN));
    assertEquals(2, ((List<?>) res.get("warnings")).size());
  }

  @Test
  void rejectsQueriesWithMoreTermsThanABooleanQueryMayHold() {
    String q = "word ".repeat(org.apache.lucene.search.IndexSearcher.getMaxClauseCount() + 1);
    assertThrows(SearchPipeline.QueryTooLongException.class,
        () -> idx.pipeline().search(Request.of(q)));
  }

  @Test
  void paramsEchoTheClampedValues() throws Exception {
    var res = idx.pipeline().search(new Request("vitamin", Mode.BM25_ONLY, true, false, true,
        50f, 2f, 5000, 1, 1, 1, 1, 9999));
    var params = (Map<String, Object>) res.get("params");
    assertEquals(10f, params.get("k1"));
    assertEquals(1f, params.get("b"));
    assertEquals(SearchPipeline.MAX_CANDIDATES, params.get("bm25N"));
    assertEquals(SearchPipeline.MAX_RERANK, params.get("rerankN"));
  }
}
