package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class EvaluatorTest {

  private static List<Candidate> ranking(String... ids) {
    List<Candidate> out = new java.util.ArrayList<>();
    for (int i = 0; i < ids.length; i++) out.add(new Candidate(ids[i], i, 1.0 / (i + 1), i + 1));
    return out;
  }

  private static final double EPS = 1e-9;

  @Test
  void perfectRankingScoresOne() {
    var qrels = Map.of("a", 2, "b", 1);
    assertEquals(1.0, Evaluator.ndcg(ranking("a", "b", "x"), qrels, 10), EPS);
    assertEquals(1.0, Evaluator.recall(ranking("a", "b"), qrels, 10), EPS);
    assertEquals(1.0, Evaluator.reciprocalRank(ranking("a", "b"), qrels), EPS);
  }

  @Test
  void ndcgUsesExponentialGainAndLogDiscount() {
    // Grades a=2 (gain 3), b=1 (gain 1). Ranking: x, b, a.
    var qrels = Map.of("a", 2, "b", 1);
    double dcg = 1 / log2(3) + 3 / log2(4);
    double idcg = 3 / log2(2) + 1 / log2(3);
    assertEquals(dcg / idcg, Evaluator.ndcg(ranking("x", "b", "a"), qrels, 10), EPS);
  }

  @Test
  void ndcgOnlyLooksAtTheTopK() {
    var qrels = Map.of("a", 1);
    assertEquals(0.0, Evaluator.ndcg(ranking("x", "y", "a"), qrels, 2), EPS);
    assertEquals(1 / log2(4), Evaluator.ndcg(ranking("x", "y", "a"), qrels, 3), EPS);
  }

  @Test
  void idealIsCappedAtKSoManyRelevantDocsStillAllowOne() {
    var qrels = Map.of("a", 1, "b", 1, "c", 1);
    assertEquals(1.0, Evaluator.ndcg(ranking("a", "b"), qrels, 2), EPS);
  }

  @Test
  void recallAndMrr() {
    var qrels = Map.of("a", 1, "b", 1, "c", 1, "d", 1);
    assertEquals(0.25, Evaluator.recall(ranking("x", "a", "y"), qrels, 10), EPS);
    assertEquals(0.0, Evaluator.recall(ranking("x", "y", "a"), qrels, 2), EPS);
    assertEquals(0.5, Evaluator.reciprocalRank(ranking("x", "a", "b"), qrels), EPS);
    assertEquals(0.0, Evaluator.reciprocalRank(ranking("x", "y"), qrels), EPS);
  }

  @Test
  void noJudgmentsOrNoResultsScoreZero() {
    assertEquals(0.0, Evaluator.ndcg(ranking("a"), Map.of(), 10), EPS);
    assertEquals(0.0, Evaluator.recall(ranking("a"), Map.of(), 10), EPS);
    assertEquals(0.0, Evaluator.ndcg(List.of(), Map.of("a", 1), 10), EPS);
    assertEquals(0.0, Evaluator.reciprocalRank(List.of(), Map.of("a", 1)), EPS);
  }

  @Test
  void metricsBundleIsRoundedAndCounted() {
    var m = Evaluator.metrics(ranking("x", "a", "y"), Map.of("a", 1, "b", 1, "c", 1));
    assertEquals(0.3333, m.get("recall@10"));
    assertEquals(0.5, m.get("mrr"));
    assertEquals(3, m.get("retrieved"));
    assertEquals(3, m.get("relevantTotal"));
  }

  private static double log2(double v) { return Math.log(v) / Math.log(2); }
}
