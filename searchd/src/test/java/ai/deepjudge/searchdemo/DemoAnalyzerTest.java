package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;

import org.junit.jupiter.api.Test;

class DemoAnalyzerTest {

  private final DemoAnalyzer analyzer = new DemoAnalyzer();

  private List<String> terms(String text) {
    return analyzer.analyze(Fields.CONTENT, text).stream().map(DemoAnalyzer.Token::term).toList();
  }

  @Test
  void lowercasesStemsAndDropsStopWords() {
    assertEquals(List.of("vitamin", "d", "defici"), terms("Vitamin D deficiency"));
    assertEquals(List.of("run", "runner", "run"), terms("the running runner runs"));
  }

  @Test
  void stripsPossessives() {
    assertEquals(List.of("cat", "whisker"), terms("The cat's whiskers"));
  }

  @Test
  void punctuationOnlyQueriesAnalyzeToNothing() {
    // The README's headline example: the shell variables vanish entirely.
    assertEquals(List.of("differ", "between"), terms("difference between $@ and $*"));
    assertEquals(List.of(), terms("$@ $* ??"));
    assertEquals(List.of(), terms(""));
  }

  @Test
  void positionsCountRemovedStopWordsAndOffsetsPointIntoTheInput() {
    String text = "the quick fox";
    var tokens = analyzer.analyze(Fields.CONTENT, text);
    assertEquals(2, tokens.size());
    // "the" is removed but still occupies position 0, so phrase queries stay correct.
    assertEquals(1, tokens.get(0).position());
    assertEquals(2, tokens.get(1).position());
    assertEquals("quick", text.substring(tokens.get(0).start(), tokens.get(0).end()));
    assertEquals("fox", text.substring(tokens.get(1).start(), tokens.get(1).end()));
  }

  @Test
  void traceReplaysTheChainOneFilterAtATime() {
    var steps = analyzer.trace("The Cat's Running");
    assertEquals(List.of(DemoAnalyzer.STEPS), steps.stream().map(DemoAnalyzer.ChainStep::filter).toList());
    assertEquals(List.of("The", "Cat's", "Running"), steps.get(0).tokens());
    assertEquals(List.of("The", "Cat", "Running"), steps.get(1).tokens());
    assertEquals(List.of("the", "cat", "running"), steps.get(2).tokens());
    assertEquals(List.of("cat", "running"), steps.get(3).tokens());
    assertEquals(List.of("cat", "run"), steps.get(4).tokens());
  }

  @Test
  void traceEndsWhereTheRealAnalyzerEnds() {
    // The trace rebuilds the chain by hand; it must not drift from the indexing analyzer.
    for (String text : List.of("Vitamin D's role in bones", "curl vs. wget: downloads",
        "ÜBER-fast Unicode naïve café", "C++ and C# programmers' tools")) {
      var steps = analyzer.trace(text);
      assertEquals(terms(text), steps.get(steps.size() - 1).tokens(), text);
    }
  }

  @Test
  void isSafeToShareAcrossThreads() throws Exception {
    var threads = new java.util.ArrayList<Thread>();
    var failures = new java.util.concurrent.atomic.AtomicInteger();
    for (int t = 0; t < 8; t++) {
      threads.add(Thread.ofPlatform().start(() -> {
        for (int i = 0; i < 500; i++) {
          if (!terms("running runners ran " + i).get(0).equals("run")) failures.incrementAndGet();
        }
      }));
    }
    for (Thread t : threads) t.join();
    assertEquals(0, failures.get());
  }
}
