package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;

/**
 * Lexical candidate retrieval: the inverted index half of the pipeline.
 *
 * <p>The query is analyzed with the same analyzer the documents were, each surviving
 * term becomes a {@code SHOULD} {@link TermQuery}, and BM25 scores the disjunction.
 * We then ask Lucene to {@link IndexSearcher#explain} each hit and decompose its own
 * explanation into the per-term tf / df / idf / field-length numbers the UI draws.
 */
public final class Bm25Stage {

  /** BM25 knobs, exposed so the UI can offer sliders. */
  public record Params(float k1, float b, int n) {}

  private final IndexSearcher searcher;
  private final DemoAnalyzer analyzer;

  public Bm25Stage(IndexSearcher searcher, DemoAnalyzer analyzer) {
    this.searcher = searcher;
    this.analyzer = analyzer;
  }

  /** The analyzed query terms with their collection-level statistics. */
  public record QueryTerm(String term, int df, double idf) {}

  public List<QueryTerm> queryTerms(List<String> terms) throws IOException {
    int numDocs = searcher.getIndexReader().numDocs();
    List<QueryTerm> out = new ArrayList<>(terms.size());
    for (String t : terms) {
      int df = searcher.getIndexReader().docFreq(new Term(Fields.CONTENT, t));
      // Lucene's BM25 idf: log(1 + (N - df + 0.5) / (df + 0.5))
      double idf = Math.log(1 + (numDocs - df + 0.5) / (df + 0.5));
      out.add(new QueryTerm(t, df, round(idf)));
    }
    return out;
  }

  public Query buildQuery(List<String> terms) {
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    for (String t : terms) {
      b.add(new TermQuery(new Term(Fields.CONTENT, t)), BooleanClause.Occur.SHOULD);
    }
    return b.build();
  }

  /**
   * Runs the search and fills in a trace stage. Returns the candidates for the next stage.
   */
  public List<Candidate> run(List<String> terms, Params params, Trace trace,
                             DocStore docs, boolean explain) throws IOException {
    if (terms.isEmpty()) {
      trace.add("bm25", 0, Trace.map("k1", params.k1(), "b", params.b(), "n", params.n()),
          Trace.map("note", "query analyzed to zero terms"), List.of());
      return List.of();
    }

    // A fresh searcher per request would be wasteful, but the similarity is per-request
    // state, so we clone the searcher over the same reader instead.
    IndexSearcher local = new IndexSearcher(searcher.getIndexReader());
    local.setSimilarity(new BM25Similarity(params.k1(), params.b()));

    Query query = buildQuery(terms);
    Trace.Timer timer = new Trace.Timer();
    TopDocs top = local.search(query, params.n());
    double ms = timer.ms();

    StoredFields stored = local.getIndexReader().storedFields();
    List<Candidate> candidates = new ArrayList<>(top.scoreDocs.length);
    List<Map<String, Object>> results = new ArrayList<>(top.scoreDocs.length);

    for (int i = 0; i < top.scoreDocs.length; i++) {
      ScoreDoc sd = top.scoreDocs[i];
      String docId = stored.document(sd.doc).get(Fields.ID);
      candidates.add(new Candidate(docId, sd.doc, sd.score, i + 1));
      docs.remember(sd.doc, docId);

      Map<String, Object> row = Trace.map("docId", docId, "rank", i + 1, "score", round(sd.score));
      if (explain) {
        row.put("explain", explainHit(local, query, sd.doc, params));
      }
      results.add(row);
    }

    trace.add("bm25", ms,
        Trace.map("k1", params.k1(), "b", params.b(), "n", params.n(),
            "similarity", "BM25Similarity", "field", Fields.CONTENT),
        Trace.map("query", query.toString(Fields.CONTENT),
            "totalHits", top.totalHits.value(),
            "totalHitsRelation", top.totalHits.relation().toString()),
        results);

    return candidates;
  }

  /**
   * Turns Lucene's own {@link Explanation} tree into a flat per-term breakdown.
   *
   * <p>The per-term contribution is taken straight from the node value, so it is exact.
   * The tf/idf/dl components underneath are pulled out for display, and the verbatim
   * explanation string goes along too so the UI can show what Lucene actually said.
   */
  private Map<String, Object> explainHit(IndexSearcher searcher, Query query, int doc, Params params)
      throws IOException {
    Explanation root = searcher.explain(query, doc);
    List<Map<String, Object>> terms = new ArrayList<>();
    collectTermExplanations(root, terms);
    return Trace.map(
        "total", round(root.getValue().doubleValue()),
        "k1", params.k1(),
        "b", params.b(),
        "terms", terms,
        "verbatim", root.toString());
  }

  /** Walks the tree looking for the per-term {@code weight(field:term in doc)} nodes. */
  private static void collectTermExplanations(Explanation node, List<Map<String, Object>> out) {
    String desc = node.getDescription();
    if (desc.startsWith("weight(")) {
      Map<String, Object> term = new LinkedHashMap<>();
      term.put("term", parseTerm(desc));
      term.put("contrib", round(node.getValue().doubleValue()));
      // These live at varying depths under the weight node, so just search the subtree.
      for (String key : new String[] {"n", "N", "freq", "k1", "b", "dl", "avgdl"}) {
        Double v = findValue(node, key);
        if (v != null) term.put(label(key), key.equals("n") || key.equals("N") ? (Object) v.longValue() : v);
      }
      Double idf = findDescription(node, "idf");
      if (idf != null) term.put("idf", round(idf));
      Double tf = findDescription(node, "tf");
      if (tf != null) term.put("tfComponent", round(tf));
      out.add(term);
      return;
    }
    for (Explanation child : node.getDetails()) {
      collectTermExplanations(child, out);
    }
  }

  /** {@code weight(content:defici in 412) [BM25Similarity], result of:} -> {@code defici} */
  private static String parseTerm(String desc) {
    int open = desc.indexOf('(');
    int in = desc.lastIndexOf(" in ");
    if (open < 0 || in < 0) return desc;
    String fieldTerm = desc.substring(open + 1, in);
    int colon = fieldTerm.indexOf(':');
    return colon < 0 ? fieldTerm : fieldTerm.substring(colon + 1);
  }

  /** Finds a leaf whose description is exactly {@code key = <number>}. */
  private static Double findValue(Explanation node, String key) {
    if (node.getDescription().startsWith(key + " =")) {
      return node.getValue().doubleValue();
    }
    for (Explanation child : node.getDetails()) {
      Double v = findValue(child, key);
      if (v != null) return v;
    }
    return null;
  }

  /** Finds the value of the node whose description starts with {@code prefix}, e.g. "idf". */
  private static Double findDescription(Explanation node, String prefix) {
    for (Explanation child : node.getDetails()) {
      if (child.getDescription().startsWith(prefix + ",") || child.getDescription().equals(prefix)) {
        return child.getValue().doubleValue();
      }
      Double v = findDescription(child, prefix);
      if (v != null) return v;
    }
    return null;
  }

  private static String label(String key) {
    return switch (key) {
      case "n" -> "df";
      case "N" -> "numDocs";
      case "freq" -> "tf";
      default -> key;
    };
  }

  static double round(double v) {
    return Math.round(v * 10000.0) / 10000.0;
  }
}
