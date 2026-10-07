package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import io.javalin.Javalin;
import io.javalin.http.Context;

/** HTTP API, and the static host for the built UI. */
public final class Main {

  private static final com.fasterxml.jackson.databind.ObjectMapper JSON =
      new com.fasterxml.jackson.databind.ObjectMapper();

  public static void main(String[] args) throws Exception {
    String datasetName = Config.dataset();
    Path indexDir = Config.indexRoot().resolve(datasetName);
    Path dataDir = Config.dataRoot().resolve(datasetName);

    if (!Files.exists(indexDir.resolve("segments_1")) && !Files.exists(indexDir)) {
      System.err.println("no index at " + indexDir + " — run the indexer first");
      System.exit(1);
    }

    var dir = FSDirectory.open(indexDir);
    var reader = DirectoryReader.open(dir);
    var searcher = new IndexSearcher(reader);
    var analyzer = new DemoAnalyzer();
    var dataset = Dataset.load(dataDir);
    var nlp = new NlpClient(Config.nlpUrl(), Duration.ofMinutes(2));
    var pipeline = new SearchPipeline(searcher, analyzer, dataset, nlp);

    System.out.printf("serving %s: %d docs, %d judged queries%n",
        datasetName, reader.numDocs(), dataset.queries().size());

    Javalin app = Javalin.create(config -> {
      config.showJavalinBanner = false;
      config.bundledPlugins.enableCors(cors -> cors.addRule(it -> it.anyHost()));
      String uiDir = Config.uiDir();
      if (uiDir != null) {
        config.staticFiles.add(uiDir, io.javalin.http.staticfiles.Location.EXTERNAL);
        config.spaRoot.addFile("/", uiDir + "/index.html", io.javalin.http.staticfiles.Location.EXTERNAL);
      }
    });

    app.get("/api/health", ctx -> {
      Map<String, Object> out = new LinkedHashMap<>();
      out.put("ok", true);
      out.put("dataset", datasetName);
      out.put("docs", reader.numDocs());
      try {
        out.put("nlp", nlp.health());
      } catch (Exception e) {
        out.put("nlp", Map.of("ok", false, "error", String.valueOf(e.getMessage())));
      }
      ctx.json(out);
    });

    app.get("/api/search", ctx -> ctx.json(pipeline.search(parse(ctx))));

    app.get("/api/analyze", ctx -> {
      String text = ctx.queryParamAsClass("text", String.class).getOrDefault("");
      ctx.json(Trace.map(
          "text", text,
          "chain", analyzer.trace(text),
          "tokens", analyzer.analyze(Fields.CONTENT, text)));
    });

    app.get("/api/term", ctx -> ctx.json(termInfo(reader, searcher,
        ctx.queryParamAsClass("t", String.class).getOrDefault(""),
        ctx.queryParamAsClass("limit", Integer.class).getOrDefault(20))));

    app.get("/api/doc/{id}", ctx -> {
      var hits = searcher.search(new org.apache.lucene.search.TermQuery(
          new Term(Fields.ID, ctx.pathParam("id"))), 1);
      if (hits.scoreDocs.length == 0) {
        ctx.status(404).json(Map.of("error", "no such doc"));
        return;
      }
      int luceneDoc = hits.scoreDocs[0].doc;
      var d = reader.storedFields().document(luceneDoc);
      ctx.json(Trace.map(
          "id", d.get(Fields.ID),
          "title", d.get(Fields.TITLE),
          "text", d.get(Fields.TEXT),
          "luceneDoc", luceneDoc,
          "topTerms", topTermsForDoc(reader, luceneDoc, 25)));
    });

    app.get("/api/stats", ctx -> {
      Path stats = indexDir.resolve("index_stats.json");
      if (Files.exists(stats)) {
        ctx.contentType("application/json").result(Files.readString(stats));
      } else {
        ctx.json(Map.of("docs", reader.numDocs(), "note", "index_stats.json missing"));
      }
    });

    app.get("/api/benchmark", ctx -> {
      Path cached = indexDir.resolve("benchmark.json");
      if (Files.exists(cached)) {
        ctx.contentType("application/json").result(Files.readString(cached));
      } else {
        ctx.status(404).json(Map.of("error", "no cached benchmark; run the Benchmark main class"));
      }
    });

    app.get("/api/queries", ctx -> {
      List<Map<String, Object>> out = new ArrayList<>();
      dataset.queries().forEach((qid, text) -> out.add(Trace.map(
          "qid", qid, "text", text, "relevantDocs", dataset.qrelsFor(qid).size())));
      ctx.json(Trace.map("dataset", datasetName, "queries", out));
    });

    app.exception(NlpClient.NlpUnavailableException.class, (e, ctx) ->
        ctx.status(503).json(Map.of("error", "nlp sidecar unavailable", "detail", e.getMessage())));

    app.start(Config.port());
  }

  private static SearchPipeline.Request parse(Context ctx) {
    String q = ctx.queryParamAsClass("q", String.class).getOrDefault("");
    SearchPipeline.Request d = SearchPipeline.Request.of(q);
    return new SearchPipeline.Request(
        q,
        SearchPipeline.Mode.valueOf(ctx.queryParamAsClass("mode", String.class)
            .getOrDefault("hybrid").toUpperCase().replace("-", "_")),
        !"off".equals(ctx.queryParam("trace")),
        ctx.queryParamAsClass("explain", Boolean.class).getOrDefault(d.explain()),
        ctx.queryParamAsClass("rerank", Boolean.class).getOrDefault(d.rerank()),
        ctx.queryParamAsClass("k1", Float.class).getOrDefault(d.k1()),
        ctx.queryParamAsClass("b", Float.class).getOrDefault(d.b()),
        ctx.queryParamAsClass("bm25_n", Integer.class).getOrDefault(d.bm25N()),
        ctx.queryParamAsClass("dense_n", Integer.class).getOrDefault(d.denseN()),
        ctx.queryParamAsClass("ef_search", Integer.class).getOrDefault(d.efSearch()),
        ctx.queryParamAsClass("rrf_k", Integer.class).getOrDefault(d.rrfK()),
        ctx.queryParamAsClass("fuse_n", Integer.class).getOrDefault(d.fuseN()),
        ctx.queryParamAsClass("rerank_n", Integer.class).getOrDefault(d.rerankN()));
  }

  /** df, idf, total term frequency, and which documents contain the term. */
  private static Map<String, Object> termInfo(DirectoryReader reader, IndexSearcher searcher,
                                              String rawTerm, int limit) throws IOException {
    // Analyze the input so "deficiency" finds the stemmed term "defici".
    List<DemoAnalyzer.Token> analyzed = new DemoAnalyzer().analyze(Fields.CONTENT, rawTerm);
    String term = analyzed.isEmpty() ? rawTerm : analyzed.get(0).term();

    Term t = new Term(Fields.CONTENT, term);
    int df = reader.docFreq(t);
    long totalTf = reader.totalTermFreq(t);
    int numDocs = reader.numDocs();
    double idf = Math.log(1 + (numDocs - df + 0.5) / (df + 0.5));

    List<Map<String, Object>> docs = new ArrayList<>();
    for (var leaf : reader.leaves()) {
      Terms terms = leaf.reader().terms(Fields.CONTENT);
      if (terms == null) continue;
      TermsEnum te = terms.iterator();
      if (!te.seekExact(new BytesRef(term))) continue;
      PostingsEnum postings = te.postings(null, PostingsEnum.FREQS);
      while (postings.nextDoc() != DocIdSetIterator.NO_MORE_DOCS && docs.size() < limit) {
        int global = postings.docID() + leaf.docBase;
        docs.add(Trace.map(
            "docId", reader.storedFields().document(global).get(Fields.ID),
            "tf", postings.freq(),
            "luceneDoc", global));
      }
    }

    return Trace.map(
        "input", rawTerm,
        "analyzedTo", term,
        "df", df,
        "numDocs", numDocs,
        "idf", Bm25Stage.round(idf),
        "totalTermFreq", totalTf,
        "postings", docs);
  }

  /** The document's own terms, highest in-document frequency first. */
  private static List<Map<String, Object>> topTermsForDoc(DirectoryReader reader, int luceneDoc, int limit)
      throws IOException {
    Terms vector = reader.termVectors().get(luceneDoc, Fields.CONTENT);
    if (vector == null) return List.of();

    List<Map<String, Object>> out = new ArrayList<>();
    TermsEnum te = vector.iterator();
    BytesRef bytes;
    while ((bytes = te.next()) != null) {
      String term = bytes.utf8ToString();
      out.add(Trace.map("term", term, "tf", te.totalTermFreq(),
          "df", reader.docFreq(new Term(Fields.CONTENT, term))));
    }
    out.sort((a, b) -> Long.compare(((Number) b.get("tf")).longValue(),
        ((Number) a.get("tf")).longValue()));
    return out.subList(0, Math.min(limit, out.size()));
  }

  private Main() {}
}
