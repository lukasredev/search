package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

@SuppressWarnings("unchecked")
class IndexStoryTest {

  private final TestIndex idx = TestIndex.get();

  @Test
  void previewShowsPostingsAgainstTheLiveIndex() throws Exception {
    var story = new IndexStory(idx.reader, new DemoAnalyzer(), idx.nlp.client());
    var out = story.preview("Vitamin pills and vitamin zebrafish", true);

    var postings = (List<Object>) out.get("postings");
    // Sorted by tf, so the repeated term comes first, with both of its positions.
    String first = postings.get(0).toString();
    assertTrue(first.contains("term=vitamin") && first.contains("tf=2")
        && first.contains("positions=[0, 3]") && first.contains("df=2"), first);

    var summary = (Map<String, Object>) out.get("summary");
    assertEquals(4, summary.get("tokensProduced"));
    assertEquals(3, summary.get("uniqueTerms"));
    assertEquals(2, summary.get("termsNewToThisIndex"), "pill and zebrafish are new");
    assertEquals(TestIndex.DOCS.size(), summary.get("docsInIndex"));

    var vector = (Map<String, Object>) out.get("vector");
    assertEquals(FakeNlp.DIM, vector.get("dim"));
    assertEquals(1.0, (Double) vector.get("norm"), 1e-3);
    assertEquals(FakeNlp.DIM * 4, ((Map<String, Object>) out.get("costEstimate")).get("vectorBytes"));
  }

  @Test
  void previewWithoutTheVectorNeverCallsTheSidecar() throws Exception {
    try (FakeNlp nlp = new FakeNlp()) {
      var out = new IndexStory(idx.reader, new DemoAnalyzer(), nlp.client()).preview("hello", false);
      assertEquals(null, out.get("vector"));
      assertEquals(0, nlp.embedCalls.get());
    }
  }

  @Test
  void previewSurvivesADeadSidecar() throws Exception {
    try (FakeNlp nlp = new FakeNlp()) {
      nlp.failStatus = 503;
      var out = new IndexStory(idx.reader, new DemoAnalyzer(), nlp.client()).preview("hello", true);
      assertTrue(((Map<String, Object>) out.get("vector")).containsKey("error"));
      assertEquals(1, ((Map<String, Object>) out.get("summary")).get("tokensProduced"));
    }
  }
}
