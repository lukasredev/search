package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ai.deepjudge.searchdemo.SearchPipeline.Mode;
import ai.deepjudge.searchdemo.SearchPipeline.Request;

class SearchRequestTest {

  @Test
  void defaultsAreTheTalkDefaults() {
    Request r = Request.of("q");
    assertEquals(Mode.HYBRID, r.mode());
    assertTrue(r.trace() && r.explain() && r.rerank());
    assertEquals(1.2f, r.k1());
    assertEquals(0.75f, r.b());
    assertEquals(100, r.bm25N());
    assertEquals(RrfFusion.DEFAULT_K, r.rrfK());
    assertEquals(50, r.fuseN());
    assertEquals(50, r.rerankN());
  }

  @Test
  void clampsEveryKnobSoNoRequestCanAskForUnboundedWork() {
    Request r = new Request("q", Mode.HYBRID, true, true, true,
        99f, 7f, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000);
    assertEquals(10f, r.k1());
    assertEquals(1f, r.b());
    assertEquals(SearchPipeline.MAX_CANDIDATES, r.bm25N());
    assertEquals(SearchPipeline.MAX_CANDIDATES, r.denseN());
    assertEquals(SearchPipeline.MAX_CANDIDATES, r.efSearch());
    assertEquals(SearchPipeline.MAX_CANDIDATES, r.fuseN());
    assertEquals(SearchPipeline.MAX_RERANK, r.rerankN());
    assertEquals(10_000, r.rrfK());
  }

  @Test
  void clampsNegativesAndNonFiniteFloats() {
    Request r = new Request("q", Mode.HYBRID, true, true, true,
        Float.NaN, Float.NEGATIVE_INFINITY, -5, 0, -1, -60, 0, -3);
    assertEquals(0f, r.k1());
    assertEquals(0f, r.b());
    assertEquals(1, r.bm25N());
    assertEquals(1, r.denseN());
    assertEquals(1, r.efSearch());
    assertEquals(1, r.rrfK());
    assertEquals(1, r.fuseN());
    assertEquals(0, r.rerankN());
    assertFalse(r.rerank(), "rerank_n=0 must switch reranking off");
  }

  @Test
  void denseParamsWidenTheBeamToAtLeastN() {
    assertEquals(50, new DenseStage.Params(50, 10, 100).efSearch());
    assertEquals(200, new DenseStage.Params(50, 200, 100).efSearch());
  }
}
