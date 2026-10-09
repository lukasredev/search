package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.search.IndexSearcher;

/**
 * The pipeline, end to end:
 *
 * <pre>
 *   analyze -> BM25 top-N  ─┐
 *                           ├─ RRF -> cross-encoder rerank -> top-K
 *           -> dense top-N ─┘
 * </pre>
 *
 * <p>Each stage appends to a single {@link Trace}, which is returned verbatim in the
 * response. The UI renders that trace and computes nothing of its own.
 */
public final class SearchPipeline {

  /** Which retrieval paths to run. Useful for showing what fusion actually buys. */
  public enum Mode { HYBRID, BM25_ONLY, DENSE_ONLY }

  /**
   * Upper bounds on the knobs.
   *
   * <p>Not paranoia: an abandoned request is still computed to completion server-side and
   * only fails when the response is written, so one client timing out and retrying costs
   * the server the full price twice. Without a ceiling, a stray {@code rerank_n=5000}
   * would occupy a worker for minutes. Clamping keeps the worst case bounded.
   */
  public static final int MAX_CANDIDATES = 1000;
  public static final int MAX_RERANK = 200;

  /** Everything the UI can turn. Defaults are the ones the talk uses. */
  public record Request(
      String query,
      Mode mode,
      boolean trace,
      boolean explain,
      boolean rerank,
      float k1,
      float b,
      int bm25N,
      int denseN,
      int efSearch,
      int rrfK,
      int fuseN,
      int rerankN) {

    /** Clamps every knob into a sane range so a bad request cannot wedge a worker. */
    public Request {
      bm25N = clamp(bm25N, 1, MAX_CANDIDATES);
      denseN = clamp(denseN, 1, MAX_CANDIDATES);
      efSearch = clamp(efSearch, 1, MAX_CANDIDATES);
      fuseN = clamp(fuseN, 1, MAX_CANDIDATES);
      rerankN = clamp(rerankN, 0, MAX_RERANK);
      rrfK = clamp(rrfK, 1, 10_000);
      k1 = (float) clamp(k1, 0.0, 10.0);
      b = (float) clamp(b, 0.0, 1.0);
      if (rerankN == 0) rerank = false;
    }

    private static int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }

    private static double clamp(double v, double lo, double hi) {
      return Double.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : lo;
    }

    public static Request of(String query) {
      return new Request(query, Mode.HYBRID, true, true, true,
          1.2f, 0.75f, 100, 100, 100, RrfFusion.DEFAULT_K, 50, 50);
    }
  }

  private final IndexSearcher searcher;
  private final DemoAnalyzer analyzer;
  private final Dataset dataset;
  private final Bm25Stage bm25;
  private final DenseStage dense;
  private final RerankStage reranker;

  public SearchPipeline(IndexSearcher searcher, DemoAnalyzer analyzer, Dataset dataset, NlpClient nlp) {
    this.searcher = searcher;
    this.analyzer = analyzer;
    this.dataset = dataset;
    this.bm25 = new Bm25Stage(searcher, analyzer);
    this.dense = new DenseStage(searcher, nlp);
    this.reranker = new RerankStage(nlp);
  }

  public Map<String, Object> search(Request req) throws IOException {
    Trace trace = new Trace();
    DocStore docs = new DocStore(searcher.getIndexReader());
    List<String> warnings = new ArrayList<>();
    Trace.Timer wall = new Trace.Timer();

    // ---- stage 0: analyze -------------------------------------------------------
    Trace.Timer analyzeTimer = new Trace.Timer();
    List<DemoAnalyzer.ChainStep> chain = analyzer.trace(req.query());
    List<String> terms = chain.get(chain.size() - 1).tokens();
    // Each term becomes one BooleanQuery clause, and Lucene refuses more than this many.
    if (terms.size() > IndexSearcher.getMaxClauseCount()) {
      throw new QueryTooLongException(terms.size(), IndexSearcher.getMaxClauseCount());
    }
    List<Bm25Stage.QueryTerm> queryTerms = bm25.queryTerms(terms);
    trace.add("analyze", analyzeTimer.ms(),
        Trace.map("analyzer", "DemoAnalyzer", "chain", List.of(DemoAnalyzer.STEPS)),
        Trace.map("chain", chain, "terms", queryTerms),
        List.of());

    // ---- stage 1: lexical candidates -------------------------------------------
    List<Candidate> bm25Hits = List.of();
    if (req.mode() != Mode.DENSE_ONLY) {
      bm25Hits = bm25.run(terms, new Bm25Stage.Params(req.k1(), req.b(), req.bm25N()),
          trace, docs, req.explain());
    }

    // ---- stage 2: semantic candidates ------------------------------------------
    List<Candidate> denseHits = List.of();
    if (req.mode() != Mode.BM25_ONLY) {
      try {
        denseHits = dense.run(req.query(),
            new DenseStage.Params(req.denseN(), req.efSearch(), Integer.MAX_VALUE), trace, docs);
      } catch (NlpClient.NlpUnavailableException e) {
        // Degrade rather than fail: a dead sidecar should still leave a working
        // lexical demo on screen.
        warnings.add("dense retrieval unavailable: " + e.getMessage());
      }
    }

    // ---- stage 3: fusion --------------------------------------------------------
    List<Candidate> fused;
    if (!bm25Hits.isEmpty() && !denseHits.isEmpty()) {
      fused = RrfFusion.fuse(
          List.of(new RrfFusion.Input("bm25", bm25Hits), new RrfFusion.Input("dense", denseHits)),
          new RrfFusion.Params(req.rrfK(), req.fuseN()), trace);
    } else {
      // Only one retriever ran, so there is nothing to fuse; carry it through so the
      // downstream stages and metrics still line up.
      List<Candidate> single = bm25Hits.isEmpty() ? denseHits : bm25Hits;
      fused = single.size() > req.fuseN() ? new ArrayList<>(single.subList(0, req.fuseN())) : single;
    }

    // ---- stage 4: rerank --------------------------------------------------------
    List<Candidate> finalRanking = fused;
    if (req.rerank() && !fused.isEmpty()) {
      try {
        finalRanking = reranker.run(req.query(), fused, new RerankStage.Params(req.rerankN()), trace, docs);
      } catch (NlpClient.NlpUnavailableException e) {
        warnings.add("reranking unavailable: " + e.getMessage());
      }
    }

    // ---- response ---------------------------------------------------------------
    String qid = dataset.qidForText(req.query());
    Map<String, Integer> qrels = qid == null ? Map.of() : dataset.qrelsFor(qid);

    Map<String, Object> response = new LinkedHashMap<>();
    response.put("query", req.query());
    response.put("dataset", dataset.name());
    response.put("qid", qid);
    response.put("mode", req.mode().name().toLowerCase());
    response.put("params", Trace.map(
        "k1", req.k1(), "b", req.b(), "bm25N", req.bm25N(), "denseN", req.denseN(),
        "efSearch", req.efSearch(), "rrfK", req.rrfK(), "fuseN", req.fuseN(),
        "rerankN", req.rerankN(), "rerank", req.rerank()));
    response.put("totalMs", Trace.round(wall.ms()));

    if (req.trace()) {
      response.put("stages", trace.stages());
    } else {
      // Production shape: just the answers. Showing this next to trace=true makes the
      // cost of all that instrumentation concrete.
      response.put("results", finalRanking.stream().limit(10)
          .map(c -> Trace.map("docId", c.docId(), "rank", c.rank(),
              "score", Bm25Stage.round(c.score()))).toList());
    }

    response.put("eval", evaluate(qid, qrels, bm25Hits, denseHits, fused, finalRanking));
    response.put("docs", docs.toJson(qrels));
    if (!warnings.isEmpty()) response.put("warnings", warnings);

    return response;
  }

  /** The query analyzed to more terms than a BooleanQuery may hold: a client error, not a 500. */
  public static final class QueryTooLongException extends RuntimeException {
    public QueryTooLongException(int terms, int max) {
      super("query analyzed to " + terms + " terms; at most " + max + " are allowed");
    }
  }

  /** Metrics after each stage, so the per-stage contribution is visible. */
  private Map<String, Object> evaluate(String qid, Map<String, Integer> qrels,
                                       List<Candidate> bm25Hits, List<Candidate> denseHits,
                                       List<Candidate> fused, List<Candidate> finalRanking) {
    if (qid == null) {
      return Trace.map("judged", false,
          "note", "query text does not match a benchmark query, so no qrels apply");
    }
    Map<String, Object> perStage = new LinkedHashMap<>();
    if (!bm25Hits.isEmpty()) perStage.put("bm25", Evaluator.metrics(bm25Hits, qrels));
    if (!denseHits.isEmpty()) perStage.put("dense", Evaluator.metrics(denseHits, qrels));
    perStage.put("rrf", Evaluator.metrics(fused, qrels));
    perStage.put("rerank", Evaluator.metrics(finalRanking, qrels));
    return Trace.map("judged", true, "qid", qid, "relevantTotal", qrels.size(), "perStage", perStage);
  }
}
