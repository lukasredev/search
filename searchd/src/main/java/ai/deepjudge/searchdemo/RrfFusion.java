package ai.deepjudge.searchdemo;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reciprocal Rank Fusion: combine several ranked lists using only the ranks.
 *
 * <pre>
 *   RRF(d) = sum over lists L of  1 / (k + rank_L(d))
 * </pre>
 *
 * <p>The appeal for teaching is that it needs no score normalization at all. BM25 scores
 * are unbounded sums of term weights; cosine similarities live in [-1, 1]. Comparing them
 * directly is meaningless, but comparing <em>positions</em> is fine, and that is all RRF
 * uses. The constant {@code k} damps the influence of the very top ranks: a larger k makes
 * the fusion flatter and more democratic.
 */
public final class RrfFusion {

  /** The default from the original RRF paper (Cormack et al., 2009). */
  public static final int DEFAULT_K = 60;

  public record Params(int k, int n) {}

  /** One input ranking. */
  public record Input(String name, List<Candidate> candidates) {}

  public static List<Candidate> fuse(List<Input> inputs, Params params, Trace trace) {
    Trace.Timer timer = new Trace.Timer();

    // docId -> running total, and docId -> per-list contribution for the UI.
    Map<String, Double> scores = new LinkedHashMap<>();
    Map<String, Integer> luceneDocs = new LinkedHashMap<>();
    Map<String, Map<String, Object>> contributions = new LinkedHashMap<>();

    for (Input input : inputs) {
      for (Candidate c : input.candidates()) {
        double term = 1.0 / (params.k() + c.rank());
        scores.merge(c.docId(), term, Double::sum);
        luceneDocs.putIfAbsent(c.docId(), c.luceneDoc());
        contributions
            .computeIfAbsent(c.docId(), d -> new LinkedHashMap<>())
            .put(input.name(), Trace.map(
                "rank", c.rank(),
                "score", Bm25Stage.round(c.score()),
                "term", round6(term)));
      }
    }

    List<Candidate> fused = new ArrayList<>();
    List<Map<String, Object>> results = new ArrayList<>();

    List<Map.Entry<String, Double>> sorted = new ArrayList<>(scores.entrySet());
    // Ties are common when k is large, so break them deterministically by docId.
    sorted.sort(Comparator.<Map.Entry<String, Double>>comparingDouble(e -> -e.getValue())
        .thenComparing(Map.Entry::getKey));

    for (int i = 0; i < Math.min(params.n(), sorted.size()); i++) {
      Map.Entry<String, Double> e = sorted.get(i);
      String docId = e.getKey();
      fused.add(new Candidate(docId, luceneDocs.get(docId), e.getValue(), i + 1));

      Map<String, Object> contrib = contributions.get(docId);
      results.add(Trace.map(
          "docId", docId,
          "rank", i + 1,
          "score", round6(e.getValue()),
          "contrib", contrib,
          // Spelled out so the UI can print the arithmetic verbatim.
          "formula", formula(contrib, params.k(), e.getValue()),
          "foundBy", new ArrayList<>(contrib.keySet())));
    }

    trace.add("rrf", timer.ms(),
        Trace.map("k", params.k(), "n", params.n(),
            "inputs", inputs.stream().map(Input::name).toList()),
        Trace.map("uniqueDocs", scores.size(),
            "overlap", overlap(inputs),
            "formula", "sum over lists of 1 / (k + rank)"),
        results);

    return fused;
  }

  /** e.g. {@code 1/(60+1) + 1/(60+7) = 0.016393 + 0.014925 = 0.031318} */
  private static String formula(Map<String, Object> contrib, int k, double total) {
    List<String> lhs = new ArrayList<>();
    List<String> rhs = new ArrayList<>();
    for (Object v : contrib.values()) {
      @SuppressWarnings("unchecked")
      Map<String, Object> c = (Map<String, Object>) v;
      lhs.add("1/(" + k + "+" + c.get("rank") + ")");
      rhs.add(String.valueOf(c.get("term")));
    }
    return String.join(" + ", lhs) + " = " + String.join(" + ", rhs) + " = " + round6(total);
  }

  /** How many docs both retrievers found — the number that makes fusion interesting. */
  private static int overlap(List<Input> inputs) {
    if (inputs.size() < 2) return 0;
    var first = inputs.get(0).candidates().stream().map(Candidate::docId).collect(
        java.util.stream.Collectors.toCollection(java.util.LinkedHashSet::new));
    for (int i = 1; i < inputs.size(); i++) {
      first.retainAll(inputs.get(i).candidates().stream().map(Candidate::docId).toList());
    }
    return first.size();
  }

  static double round6(double v) {
    return Math.round(v * 1_000_000.0) / 1_000_000.0;
  }

  private RrfFusion() {}
}
