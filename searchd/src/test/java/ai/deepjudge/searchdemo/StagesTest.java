package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Each retrieval stage on its own, against the real index the {@link Indexer} built. */
class StagesTest {

  private final TestIndex idx = TestIndex.get();
  private final DemoAnalyzer analyzer = new DemoAnalyzer();

  private List<String> terms(String q) {
    var chain = analyzer.trace(q);
    return chain.get(chain.size() - 1).tokens();
  }

  private static List<String> ids(List<Candidate> cs) {
    return cs.stream().map(Candidate::docId).toList();
  }

  // ---- BM25 ------------------------------------------------------------------

  @Test
  void bm25FindsDocsSharingStemmedTerms() throws Exception {
    var bm25 = new Bm25Stage(idx.searcher, analyzer);
    Trace trace = new Trace();
    var hits = bm25.run(terms("vitamin deficiencies"), new Bm25Stage.Params(1.2f, 0.75f, 10),
        trace, new DocStore(idx.reader), true);

    // d1 has both stems, so it must lead; d8 matches only through stemming "deficiency".
    assertEquals("d1", hits.get(0).docId());
    assertTrue(ids(hits).containsAll(List.of("d1", "d2", "d8")), ids(hits).toString());
    assertFalse(ids(hits).contains("d6"));
    for (int i = 1; i < hits.size(); i++) assertTrue(hits.get(i - 1).score() >= hits.get(i).score());
    assertEquals(3L, trace.stages().get(0).detail().get("totalHits"));
  }

  @Test
  @SuppressWarnings("unchecked")
  void bm25ExplanationAddsUpToTheScore() throws Exception {
    var bm25 = new Bm25Stage(idx.searcher, analyzer);
    Trace trace = new Trace();
    var hits = bm25.run(terms("vitamin deficiency"), new Bm25Stage.Params(1.2f, 0.75f, 10),
        trace, new DocStore(idx.reader), true);

    Map<String, Object> top = trace.stages().get(0).results().get(0);
    Map<String, Object> explain = (Map<String, Object>) top.get("explain");
    var perTerm = (List<Map<String, Object>>) explain.get("terms");
    assertEquals(List.of("vitamin", "defici"), perTerm.stream().map(t -> t.get("term")).toList());

    double sum = perTerm.stream().mapToDouble(t -> (Double) t.get("contrib")).sum();
    assertEquals(hits.get(0).score(), (Double) explain.get("total"), 1e-3);
    assertEquals(hits.get(0).score(), sum, 1e-3);
    // The pieces the UI draws must all be there.
    for (String key : List.of("df", "numDocs", "tf", "dl", "avgdl", "idf", "tfComponent")) {
      assertTrue(perTerm.get(0).containsKey(key), key + " missing from " + perTerm.get(0));
    }
    assertEquals((long) TestIndex.DOCS.size(), perTerm.get(0).get("numDocs"));
    assertTrue(((String) explain.get("verbatim")).contains("BM25Similarity"));
  }

  @Test
  void bm25KnobsChangeScores() throws Exception {
    var bm25 = new Bm25Stage(idx.searcher, analyzer);
    var q = terms("running processes");
    double defaults = bm25.run(q, new Bm25Stage.Params(1.2f, 0.75f, 5), new Trace(),
        new DocStore(idx.reader), false).get(0).score();
    double noTf = bm25.run(q, new Bm25Stage.Params(0f, 0.75f, 5), new Trace(),
        new DocStore(idx.reader), false).get(0).score();
    // k1 = 0 removes term frequency entirely: the score is the sum of idfs.
    assertNotEquals(defaults, noTf);
    var qt = bm25.queryTerms(q);
    assertEquals(qt.stream().mapToDouble(Bm25Stage.QueryTerm::idf).sum(), noTf, 1e-3);
  }

  @Test
  void bm25WithNoTermsRecordsAnEmptyStage() throws Exception {
    Trace trace = new Trace();
    var hits = new Bm25Stage(idx.searcher, analyzer).run(List.of(),
        new Bm25Stage.Params(1.2f, 0.75f, 10), trace, new DocStore(idx.reader), true);
    assertEquals(List.of(), hits);
    assertEquals("query analyzed to zero terms", trace.stages().get(0).detail().get("note"));
  }

  @Test
  void queryTermsReportDfAndLuceneIdf() throws Exception {
    var qt = new Bm25Stage(idx.searcher, analyzer).queryTerms(List.of("vitamin", "zzz"));
    int n = TestIndex.DOCS.size();
    assertEquals(2, qt.get(0).df());
    assertEquals(Bm25Stage.round(Math.log(1 + (n - 2 + 0.5) / 2.5)), qt.get(0).idf());
    assertEquals(0, qt.get(1).df());
  }

  // ---- dense -----------------------------------------------------------------

  @Test
  void denseReturnsNearestNeighboursByCosine() throws Exception {
    Trace trace = new Trace();
    // Same words as d5's title and text, so its fake embedding is the closest.
    var hits = new DenseStage(idx.searcher, idx.nlp.client()).run(
        "curl vs wget What is the difference between curl and wget downloads",
        new DenseStage.Params(3, 10, Integer.MAX_VALUE), trace, new DocStore(idx.reader));

    assertEquals(3, hits.size());
    assertEquals("d5", hits.get(0).docId());
    assertEquals(1.0, hits.get(0).score(), 1e-3);
    for (int i = 1; i < hits.size(); i++) assertTrue(hits.get(i - 1).score() >= hits.get(i).score());

    var detail = trace.stages().get(0).detail();
    assertTrue(((Number) detail.get("visitedCount")).longValue() > 0);
    assertEquals(false, detail.get("earlyTerminated"));
    assertEquals(FakeNlp.EMBED_MODEL, trace.stages().get(0).params().get("model"));
  }

  @Test
  void denseVisitLimitEarlyTerminates() throws Exception {
    Trace trace = new Trace();
    new DenseStage(idx.searcher, idx.nlp.client()).run("vitamin",
        new DenseStage.Params(5, 5, 1), trace, new DocStore(idx.reader));
    assertEquals(true, trace.stages().get(0).detail().get("earlyTerminated"));
  }

  // ---- rerank ----------------------------------------------------------------

  @Test
  @SuppressWarnings("unchecked")
  void rerankReordersByCrossEncoderScoreAndRecordsMovement() throws Exception {
    DocStore docs = new DocStore(idx.reader);
    // Feed the docs in a deliberately bad order; d1 contains every query word.
    List<Candidate> input = List.of(cand(docs, "d6", 1), cand(docs, "d2", 2), cand(docs, "d1", 3));
    Trace trace = new Trace();
    var out = new RerankStage(idx.nlp.client()).run("vitamin d deficiency", input,
        new RerankStage.Params(10), trace, docs);

    assertEquals(List.of("d1", "d2", "d6"), ids(out));
    assertEquals(List.of(1, 2, 3), out.stream().map(Candidate::rank).toList());
    var top = trace.stages().get(0).results().get(0);
    assertEquals(3, top.get("prevRank"));
    assertEquals(2, top.get("delta"));
    assertEquals(FakeNlp.RERANK_MODEL, trace.stages().get(0).params().get("model"));
  }

  @Test
  void rerankOnlyScoresTheTopNAndFlagsTruncation() throws Exception {
    DocStore docs = new DocStore(idx.reader);
    List<Candidate> input = List.of(cand(docs, "d9", 1), cand(docs, "d1", 2), cand(docs, "d2", 3));
    Trace trace = new Trace();
    var out = new RerankStage(idx.nlp.client()).run("benchmark", input,
        new RerankStage.Params(2), trace, docs);
    assertEquals(2, out.size());
    assertEquals(1, trace.stages().get(0).detail().get("truncatedDocs"));
  }

  @Test
  void rerankRejectsAScoreCountThatDoesNotMatch() throws Exception {
    try (FakeNlp nlp = new FakeNlp()) {
      nlp.shortRerank = true;
      DocStore docs = new DocStore(idx.reader);
      List<Candidate> input = List.of(cand(docs, "d1", 1), cand(docs, "d2", 2));
      assertThrows(NlpClient.NlpUnavailableException.class, () -> new RerankStage(nlp.client())
          .run("q", input, new RerankStage.Params(10), new Trace(), docs));
    }
  }

  // ---- doc store -------------------------------------------------------------

  @Test
  @SuppressWarnings("unchecked")
  void docStoreSnipsLongTextAtAWordBoundaryAndMarksRelevance() throws Exception {
    DocStore docs = new DocStore(idx.reader);
    cand(docs, "d9", 1);
    cand(docs, "d8", 2);
    var json = docs.toJson(Map.of("d9", 2));
    var d9 = (Map<String, Object>) json.get("d9");
    String snippet = (String) d9.get("snippet");
    assertTrue(snippet.endsWith("…") && snippet.length() <= 421, snippet);
    assertTrue(TestIndex.LONG_TEXT.startsWith(snippet.substring(0, snippet.length() - 1)));
    assertFalse(snippet.substring(0, snippet.length() - 1).endsWith(" "));
    assertEquals(TestIndex.LONG_TEXT.length(), d9.get("chars"));
    assertEquals(2, d9.get("relevance"));
    assertEquals(0, ((Map<String, Object>) json.get("d8")).get("relevance"));
    // Untitled docs reach the reranker as their bare text.
    assertEquals("Untitled document about deficiency in sleep.", docs.fullText("d8"));
  }

  /** Looks a doc up by id and registers it with the store, as a retrieval stage would. */
  private Candidate cand(DocStore docs, String id, int rank) throws Exception {
    var top = idx.searcher.search(new org.apache.lucene.search.TermQuery(
        new org.apache.lucene.index.Term(Fields.ID, id)), 1);
    int luceneDoc = top.scoreDocs[0].doc;
    docs.remember(luceneDoc, id);
    return new Candidate(id, luceneDoc, 1.0 / rank, rank);
  }
}
