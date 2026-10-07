package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Cross-encoder reranking: the precision stage.
 *
 * <p>Where the bi-encoder scores a query and a document by comparing two independently
 * computed vectors, the cross-encoder feeds {@code [CLS] query [SEP] document [SEP]}
 * through a transformer and reads out one relevance score. It sees the interaction
 * between the two texts, which is why it is more accurate — and why it cannot be
 * precomputed, which is why you only ever run it on a few dozen candidates.
 *
 * <p>The latency this stage reports is the single best argument on the slide for why
 * the pipeline has a candidate-retrieval phase at all.
 */
public final class RerankStage {

  public record Params(int n) {}

  private final NlpClient nlp;

  public RerankStage(NlpClient nlp) {
    this.nlp = nlp;
  }

  public List<Candidate> run(String query, List<Candidate> input, Params params,
                             Trace trace, DocStore docs) throws IOException {
    List<Candidate> pool = input.size() > params.n() ? input.subList(0, params.n()) : input;
    if (pool.isEmpty()) {
      trace.add("rerank", 0, Trace.map("n", params.n()), Trace.map("note", "no candidates"), List.of());
      return List.of();
    }

    List<String> texts = new ArrayList<>(pool.size());
    for (Candidate c : pool) texts.add(docs.fullText(c.docId()));

    Trace.Timer timer = new Trace.Timer();
    NlpClient.RerankResult res = nlp.rerank(query, texts);
    double ms = timer.ms();

    // Rank before this stage, so the UI can draw the movement.
    Map<String, Integer> previousRank = new HashMap<>();
    for (Candidate c : pool) previousRank.put(c.docId(), c.rank());

    List<Candidate> scored = new ArrayList<>(pool.size());
    Map<String, NlpClient.RerankItem> items = new HashMap<>();
    for (int i = 0; i < pool.size(); i++) {
      Candidate c = pool.get(i);
      NlpClient.RerankItem item = res.items().get(i);
      items.put(c.docId(), item);
      scored.add(new Candidate(c.docId(), c.luceneDoc(), item.score(), 0));
    }
    scored.sort(Comparator.comparingDouble(Candidate::score).reversed());

    List<Candidate> ranked = new ArrayList<>(scored.size());
    List<Map<String, Object>> results = new ArrayList<>(scored.size());
    int truncated = 0;

    for (int i = 0; i < scored.size(); i++) {
      Candidate c = scored.get(i).withRank(i + 1);
      ranked.add(c);
      NlpClient.RerankItem item = items.get(c.docId());
      if (item.truncated()) truncated++;

      int prev = previousRank.get(c.docId());
      results.add(Trace.map(
          "docId", c.docId(),
          "rank", i + 1,
          "score", Bm25Stage.round(c.score()),
          "prevRank", prev,
          "delta", prev - (i + 1),
          "docTokens", item.doc_tokens(),
          // The 512-token window is a real limit and a good teaching moment: a long
          // document is judged on its opening only.
          "truncated", item.truncated()));
    }

    trace.add("rerank", ms,
        Trace.map("n", params.n(), "model", res.model(), "maxLength", res.max_length(),
            "scored", pool.size()),
        Trace.map("nlpReportedMs", res.ms(),
            "msPerDoc", Trace.round(ms / pool.size()),
            "truncatedDocs", truncated),
        results);

    return ranked;
  }
}
