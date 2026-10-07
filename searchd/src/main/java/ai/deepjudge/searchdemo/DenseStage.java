package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

/**
 * Semantic candidate retrieval: the HNSW half of the pipeline.
 *
 * <p>Production code would just run {@code new KnnFloatVectorQuery(field, vector, n)}
 * through the {@link IndexSearcher}. We instead call the leaf-level
 * {@link org.apache.lucene.index.LeafReader#searchNearestVectors} directly, because it
 * returns the graph's <em>visited count</em> in {@code totalHits} — the number of nodes
 * the search actually touched. That turns the efSearch / recall / latency tradeoff into
 * something you can show on screen instead of assert on a slide.
 *
 * <p>Separating {@code efSearch} from {@code n} is the whole point: ask the graph for a
 * wider beam than you need, then keep the top {@code n}.
 */
public final class DenseStage {

  /**
   * @param n        how many candidates to return
   * @param efSearch beam width for the graph search; clamped to at least {@code n}
   * @param visitLimit hard cap on visited nodes, after which the search early-terminates
   */
  public record Params(int n, int efSearch, int visitLimit) {
    public Params {
      efSearch = Math.max(efSearch, n);
    }
  }

  private final IndexSearcher searcher;
  private final NlpClient nlp;

  public DenseStage(IndexSearcher searcher, NlpClient nlp) {
    this.searcher = searcher;
    this.nlp = nlp;
  }

  public List<Candidate> run(String query, Params params, Trace trace, DocStore docs)
      throws IOException {
    Trace.Timer total = new Trace.Timer();

    // 1. Embed the query. On CPU this is usually the dominant cost of the stage.
    Trace.Timer embedTimer = new Trace.Timer();
    NlpClient.EmbedResult embedded = nlp.embedQuery(query);
    double embedMs = embedTimer.ms();
    float[] vector = embedded.vectors()[0];

    // 2. Walk the HNSW graph in each segment, then merge.
    Trace.Timer graphTimer = new Trace.Timer();
    List<ScoreDoc> merged = new ArrayList<>();
    long visited = 0;
    boolean earlyTerminated = false;

    for (LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
      AcceptDocs accept = AcceptDocs.fromLiveDocs(leaf.reader().getLiveDocs(), leaf.reader().maxDoc());
      TopDocs hits = leaf.reader().searchNearestVectors(
          Fields.VECTOR, vector, params.efSearch(), accept, params.visitLimit());
      if (hits == null) continue;

      // TopKnnCollector reports visited nodes as totalHits, and marks the relation
      // GREATER_THAN_OR_EQUAL_TO when it hit the visit limit.
      visited += hits.totalHits.value();
      earlyTerminated |= hits.totalHits.relation()
          != org.apache.lucene.search.TotalHits.Relation.EQUAL_TO;

      for (ScoreDoc sd : hits.scoreDocs) {
        merged.add(new ScoreDoc(sd.doc + leaf.docBase, sd.score));
      }
    }
    merged.sort((a, c) -> Double.compare(c.score, a.score));
    double graphMs = graphTimer.ms();

    StoredFields stored = searcher.getIndexReader().storedFields();
    List<Candidate> candidates = new ArrayList<>();
    List<Map<String, Object>> results = new ArrayList<>();

    for (int i = 0; i < Math.min(params.n(), merged.size()); i++) {
      ScoreDoc sd = merged.get(i);
      String docId = stored.document(sd.doc).get(Fields.ID);
      candidates.add(new Candidate(docId, sd.doc, sd.score, i + 1));
      docs.remember(sd.doc, docId);
      results.add(Trace.map("docId", docId, "rank", i + 1, "score", Bm25Stage.round(sd.score)));
    }

    trace.add("dense", total.ms(),
        Trace.map("n", params.n(), "efSearch", params.efSearch(), "visitLimit", params.visitLimit(),
            "model", embedded.model(), "dim", embedded.dim(),
            "similarity", "COSINE", "field", Fields.VECTOR),
        Trace.map("embedMs", Trace.round(embedMs),
            "nlpReportedEmbedMs", embedded.ms(),
            "graphMs", Trace.round(graphMs),
            "visitedCount", visited,
            "visitedFraction", Trace.round(100.0 * visited / Math.max(1, searcher.getIndexReader().numDocs())),
            "earlyTerminated", earlyTerminated,
            "candidatesBeforeTruncation", merged.size()),
        results);

    return candidates;
  }
}
