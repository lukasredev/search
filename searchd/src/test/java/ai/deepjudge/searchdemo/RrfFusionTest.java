package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class RrfFusionTest {

  private static List<Candidate> list(String... ids) {
    List<Candidate> out = new java.util.ArrayList<>();
    // Raw scores deliberately on wildly different scales: RRF must ignore them.
    for (int i = 0; i < ids.length; i++) out.add(new Candidate(ids[i], i, 1000.0 - i, i + 1));
    return out;
  }

  @Test
  void sumsReciprocalRanksAcrossLists() {
    Trace trace = new Trace();
    var fused = RrfFusion.fuse(List.of(
            new RrfFusion.Input("bm25", list("a", "b", "c")),
            new RrfFusion.Input("dense", list("c", "a", "d"))),
        new RrfFusion.Params(60, 10), trace);

    assertEquals(List.of("a", "c", "b", "d"), fused.stream().map(Candidate::docId).toList());
    assertEquals(1.0 / 61 + 1.0 / 62, fused.get(0).score(), 1e-12);
    assertEquals(List.of(1, 2, 3, 4), fused.stream().map(Candidate::rank).toList());
    // Lucene doc ids are carried through from whichever list saw the doc first.
    assertEquals(0, fused.get(0).luceneDoc());

    Trace.Stage stage = trace.stages().get(0);
    assertEquals("rrf", stage.name());
    assertEquals(4, stage.detail().get("uniqueDocs"));
    assertEquals(2, stage.detail().get("overlap"));
    Map<String, Object> top = stage.results().get(0);
    assertEquals(List.of("bm25", "dense"), top.get("foundBy"));
    assertEquals("1/(60+1) + 1/(60+2) = 0.016393 + 0.016129 = 0.032522", top.get("formula"));
  }

  @Test
  void truncatesToN() {
    var fused = RrfFusion.fuse(List.of(new RrfFusion.Input("bm25", list("a", "b", "c"))),
        new RrfFusion.Params(60, 2), new Trace());
    assertEquals(List.of("a", "b"), fused.stream().map(Candidate::docId).toList());
  }

  @Test
  void tiesBreakByDocIdSoResultsAreStable() {
    var fused = RrfFusion.fuse(List.of(
            new RrfFusion.Input("bm25", list("z", "y")),
            new RrfFusion.Input("dense", list("y", "z"))),
        new RrfFusion.Params(60, 10), new Trace());
    assertEquals(List.of("y", "z"), fused.stream().map(Candidate::docId).toList());
  }

  @Test
  void largerKFlattensTheFusion() {
    // With a small k, being first in one list beats being second in both; with a big k
    // the doc that both lists agree on wins.
    List<RrfFusion.Input> inputs = List.of(
        new RrfFusion.Input("bm25", list("solo", "both", "x1", "x2")),
        new RrfFusion.Input("dense", list("y1", "both", "y2", "y3")));
    assertEquals("both", RrfFusion.fuse(inputs, new RrfFusion.Params(60, 10), new Trace()).get(0).docId());
    var lowK = RrfFusion.fuse(List.of(
            new RrfFusion.Input("bm25", list("solo", "x0", "x1", "both")),
            new RrfFusion.Input("dense", list("y0", "y1", "y2", "both"))),
        new RrfFusion.Params(1, 10), new Trace());
    assertEquals("solo", lowK.get(0).docId());
  }

  @Test
  void emptyInputsFuseToNothing() {
    var fused = RrfFusion.fuse(List.of(new RrfFusion.Input("bm25", List.of())),
        new RrfFusion.Params(60, 10), new Trace());
    assertEquals(List.of(), fused);
  }
}
