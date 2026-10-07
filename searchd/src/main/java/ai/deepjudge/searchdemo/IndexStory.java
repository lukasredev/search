package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;

/**
 * Answers "what happens to one document when it is indexed?"
 *
 * <p>Runs a piece of text through exactly the same analyzer the {@link Indexer} uses and
 * reports everything the index would store for it: the token stream after each filter, the
 * postings that would be written, and the embedding that would go into the HNSW graph.
 *
 * <p>Nothing is written — this is a dry run, so a speaker can paste arbitrary text mid-talk
 * and show what the engine would do with it.
 */
public final class IndexStory {

  private final DirectoryReader reader;
  private final DemoAnalyzer analyzer;
  private final NlpClient nlp;

  public IndexStory(DirectoryReader reader, DemoAnalyzer analyzer, NlpClient nlp) {
    this.reader = reader;
    this.analyzer = analyzer;
    this.nlp = nlp;
  }

  /** A term as this document would contribute it to the inverted index. */
  private record Posting(String term, int tf, List<Integer> positions, int df, double idf) {}

  public Map<String, Object> preview(String text, boolean withVector) throws IOException {
    Trace.Timer timer = new Trace.Timer();

    // 1. Analysis: the same chain that ran at index time, replayed one filter at a time.
    List<DemoAnalyzer.ChainStep> chain = analyzer.trace(text);
    List<DemoAnalyzer.Token> tokens = analyzer.analyze(Fields.CONTENT, text);

    // 2. Postings: group the tokens by term. This is literally what gets appended to each
    //    term's posting list — the document id, the frequency, and the positions.
    Map<String, List<Integer>> byTerm = new LinkedHashMap<>();
    for (DemoAnalyzer.Token t : tokens) {
      byTerm.computeIfAbsent(t.term(), k -> new ArrayList<>()).add(t.position());
    }

    int numDocs = reader.numDocs();
    List<Posting> postings = new ArrayList<>(byTerm.size());
    int newTerms = 0;
    for (Map.Entry<String, List<Integer>> e : byTerm.entrySet()) {
      int df = reader.docFreq(new Term(Fields.CONTENT, e.getKey()));
      if (df == 0) newTerms++;
      double idf = Math.log(1 + (numDocs - df + 0.5) / (df + 0.5));
      postings.add(new Posting(e.getKey(), e.getValue().size(), e.getValue(), df,
          Bm25Stage.round(idf)));
    }
    postings.sort((a, b) -> b.tf() != a.tf() ? b.tf() - a.tf() : a.term().compareTo(b.term()));

    Map<String, Object> out = Trace.map(
        "text", text,
        "chars", text.length(),
        "analyzer", Trace.map("chain", chain, "steps", List.of(DemoAnalyzer.STEPS)),
        "tokens", tokens,
        "postings", postings,
        "summary", Trace.map(
            "tokensProduced", tokens.size(),
            "uniqueTerms", byTerm.size(),
            "termsNewToThisIndex", newTerms,
            "docsInIndex", numDocs));

    // 3. The vector half: the same text, embedded by the same model the indexer used.
    if (withVector) {
      try {
        NlpClient.EmbedResult embedded = nlp.embed(List.of(text));
        float[] v = embedded.vectors()[0];
        out.put("vector", Trace.map(
            "model", embedded.model(),
            "dim", embedded.dim(),
            "ms", embedded.ms(),
            "bytes", embedded.dim() * 4,
            "norm", Bm25Stage.round(norm(v)),
            "values", v));
      } catch (NlpClient.NlpUnavailableException e) {
        out.put("vector", Map.of("error", e.getMessage()));
      }
    }

    // 4. Rough cost. Postings compression makes the lexical side hard to predict, so this
    //    is labelled an estimate; the vector side is exact.
    out.put("costEstimate", Trace.map(
        "vectorBytes", withVector ? 384 * 4 : 0,
        "storedBytes", text.getBytes(java.nio.charset.StandardCharsets.UTF_8).length,
        "note", "postings are delta-encoded and compressed, so their true cost only shows "
              + "up in the whole-index figures from /api/stats"));
    out.put("ms", Trace.round(timer.ms()));
    return out;
  }

  /** Reads back the vector actually stored for an indexed document. */
  public float[] storedVector(int luceneDoc) throws IOException {
    for (LeafReaderContext leaf : reader.leaves()) {
      if (luceneDoc < leaf.docBase || luceneDoc >= leaf.docBase + leaf.reader().maxDoc()) continue;
      FloatVectorValues values = leaf.reader().getFloatVectorValues(Fields.VECTOR);
      if (values == null) return null;
      int target = luceneDoc - leaf.docBase;
      var it = values.iterator();
      if (it.advance(target) == target) {
        return values.vectorValue(it.index());
      }
    }
    return null;
  }

  private static double norm(float[] v) {
    double sum = 0;
    for (float f : v) sum += f * f;
    return Math.sqrt(sum);
  }
}
