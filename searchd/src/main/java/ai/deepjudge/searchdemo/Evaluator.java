package ai.deepjudge.searchdemo;

import java.util.List;
import java.util.Map;

/**
 * Standard IR metrics, computed from the qrels that ship with the dataset.
 *
 * <p>Running these after <em>every</em> stage is the most convincing thing in the demo:
 * it turns "the reranker helps" from a claim into a number.
 */
public final class Evaluator {

  /** nDCG@k with the exponential gain used by TREC and BEIR: gain = 2^rel - 1. */
  public static double ndcg(List<Candidate> ranking, Map<String, Integer> qrels, int k) {
    if (qrels.isEmpty()) return 0;

    double dcg = 0;
    for (int i = 0; i < Math.min(k, ranking.size()); i++) {
      int rel = qrels.getOrDefault(ranking.get(i).docId(), 0);
      if (rel > 0) dcg += (Math.pow(2, rel) - 1) / log2(i + 2);
    }

    // Ideal ranking: all judged-relevant docs sorted by grade, descending.
    double idcg = 0;
    List<Integer> ideal = qrels.values().stream().sorted((a, b) -> b - a).toList();
    for (int i = 0; i < Math.min(k, ideal.size()); i++) {
      idcg += (Math.pow(2, ideal.get(i)) - 1) / log2(i + 2);
    }

    return idcg == 0 ? 0 : dcg / idcg;
  }

  /** Fraction of the judged-relevant documents that appear in the top k. */
  public static double recall(List<Candidate> ranking, Map<String, Integer> qrels, int k) {
    if (qrels.isEmpty()) return 0;
    long found = ranking.stream().limit(k).filter(c -> qrels.containsKey(c.docId())).count();
    return (double) found / qrels.size();
  }

  /** Reciprocal rank of the first relevant document, or 0 if none is retrieved. */
  public static double reciprocalRank(List<Candidate> ranking, Map<String, Integer> qrels) {
    for (int i = 0; i < ranking.size(); i++) {
      if (qrels.containsKey(ranking.get(i).docId())) return 1.0 / (i + 1);
    }
    return 0;
  }

  /** The metric bundle reported for one stage. */
  public static Map<String, Object> metrics(List<Candidate> ranking, Map<String, Integer> qrels) {
    return Trace.map(
        "ndcg@10", r4(ndcg(ranking, qrels, 10)),
        "recall@10", r4(recall(ranking, qrels, 10)),
        "recall@100", r4(recall(ranking, qrels, 100)),
        "mrr", r4(reciprocalRank(ranking, qrels)),
        "retrieved", ranking.size(),
        "relevantTotal", qrels.size());
  }

  private static double log2(double v) {
    return Math.log(v) / Math.log(2);
  }

  static double r4(double v) {
    return Math.round(v * 10000.0) / 10000.0;
  }

  private Evaluator() {}
}
