package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
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

  /** The UI dev server, the only page that calls the API cross-origin. */
  static final String DEV_ORIGIN = "http://localhost:5173";

  /**
   * Most postings /api/term returns. Each one costs a stored-fields read, and a common
   * term on Quora has hundreds of thousands of them.
   */
  static final int MAX_POSTINGS = 1000;

  public static void main(String[] args) throws Exception {
    String datasetName = Config.dataset();
    Path indexDir = Config.indexRoot().resolve(datasetName);
    Path dataDir = Config.dataRoot().resolve(datasetName);

    // Checked before opening: the index volume is mounted read-only in Docker, so
    // FSDirectory.open() would fail trying to create a missing directory.
    if (!Files.isDirectory(indexDir)) {
      System.err.printf("No index directory at %s — run the indexer for '%s' first.%n",
          indexDir.toAbsolutePath(), datasetName);
      System.exit(1);
    }
    var dir = FSDirectory.open(indexDir);
    if (!DirectoryReader.indexExists(dir)) {
      System.err.printf("""
          No Lucene index at %s.

          Build one first:
            docker compose run --rm indexer %s     (in Docker)
            make index DATASET=%s                  (locally)
          %n""", indexDir.toAbsolutePath(), datasetName, datasetName);
      System.exit(1);
    }
    var reader = DirectoryReader.open(dir);
    var dataset = Dataset.load(dataDir);
    var nlp = new NlpClient(Config.nlpUrl(), Duration.ofMinutes(2));

    System.out.printf("serving %s: %d docs, %d judged queries%n",
        datasetName, reader.numDocs(), dataset.queries().size());

    Javalin app = createApp(reader, dataset, nlp, indexDir, Config.uiDir(),
        System.getenv("AUTH_USER"), System.getenv("AUTH_PASS"), Config.stuckRequestAfter());
    Runtime.getRuntime().addShutdownHook(new Thread(app::stop, "shutdown"));
    app.start(Config.listenHost(), Config.port());
  }

  /**
   * Every route, wired to an open index. Separate from {@link #main} so the tests can
   * stand the real HTTP surface up against a small index and a fake sidecar.
   *
   * @param uiDir    directory holding the UI, or null to serve only the API
   * @param authUser with {@code authPass}, turns on basic auth; null or blank leaves it off
   */
  static Javalin createApp(DirectoryReader reader, Dataset dataset, NlpClient nlp, Path indexDir,
                           String uiDir, String authUser, String authPass, Duration stuckAfter) {
    String datasetName = dataset.name();
    var searcher = new IndexSearcher(reader);
    var analyzer = new DemoAnalyzer();
    var pipeline = new SearchPipeline(searcher, analyzer, dataset, nlp);
    var indexStory = new IndexStory(reader, analyzer, nlp);

    Javalin app = Javalin.create(config -> {
      config.showJavalinBanner = false;
      if (uiDir != null) {
        config.staticFiles.add(uiDir, io.javalin.http.staticfiles.Location.EXTERNAL);
        config.spaRoot.addFile("/", uiDir + "/index.html", io.javalin.http.staticfiles.Location.EXTERNAL);
      }
      // On SIGTERM, finish in-flight searches (a rerank takes seconds) before exiting.
      config.jetty.modifyServer(server -> server.setStopTimeout(Config.shutdownGrace().toMillis()));
    });

    new Probes(stuckAfter).register(app, searcher, datasetName);
    app.before(ctx -> {
      // The API returns JSON and the UI renders it with innerHTML, so a browser must never
      // sniff a response into HTML. Framing stays allowed: the demo may live in a slide.
      ctx.header("X-Content-Type-Options", "nosniff");
      ctx.header("Referrer-Policy", "no-referrer");
      // Only the UI dev server gets CORS headers. Allowing any origin would let every page
      // open in the browser query a local, auth-less instance. Not Javalin's CORS plugin:
      // it answers 400 to any Origin it does not know, and browsers send Origin on
      // same-origin POSTs too, which broke POST /api/index-preview everywhere but :5173.
      if (DEV_ORIGIN.equals(ctx.header("Origin"))) {
        ctx.header("Access-Control-Allow-Origin", DEV_ORIGIN);
        ctx.header("Vary", "Origin");
      }
    });
    // The dev server's POSTs carry a JSON content type, so the browser preflights them.
    app.options("/api/*", ctx -> {
      if (DEV_ORIGIN.equals(ctx.header("Origin"))) {
        ctx.header("Access-Control-Allow-Methods", "GET, POST");
        ctx.header("Access-Control-Allow-Headers", "content-type");
      }
      ctx.status(204);
    });
    requireAuthIfConfigured(app, authUser, authPass);

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

    app.get("/api/term", ctx -> ctx.json(termInfo(reader, analyzer,
        ctx.queryParamAsClass("t", String.class).getOrDefault(""),
        Math.max(0, Math.min(MAX_POSTINGS,
            ctx.queryParamAsClass("limit", Integer.class).getOrDefault(20))))));

    // The indexing story: run arbitrary text through the real analyzer and embedding
    // model and show everything the index would store, without writing anything.
    // A POST, because a whole document does not fit in a URL: Jetty rejects request
    // lines over 8 KB, and 2% of the CQADupStack posts are longer than that.
    app.post("/api/index-preview", ctx -> {
      // The validator turns an unparseable body into a 400, where bodyAsClass would 500.
      PreviewRequest req = ctx.bodyValidator(PreviewRequest.class).get();
      ctx.json(indexStory.preview(req.text() == null ? "" : req.text(),
          req.vector() == null || req.vector()));
    });

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
          "topTerms", topTermsForDoc(reader, luceneDoc, 25),
          // The vector Lucene actually stored for this document, read back from the index.
          "vector", indexStory.storedVector(luceneDoc)));
    });

    app.get("/api/stats", ctx -> {
      Path stats = indexDir.resolve("index_stats.json");
      if (Files.exists(stats)) {
        ctx.contentType("application/json").result(Files.readString(stats));
      } else {
        ctx.json(Map.of("docs", reader.numDocs(), "note", "index_stats.json missing"));
      }
    });

    // Curated examples live beside the index, not in the UI bundle: they are produced
    // from the index by scripts/find_demo_queries.py and change whenever it is rebuilt.
    app.get("/api/demo-queries", ctx -> {
      Path demos = indexDir.resolve("demo_queries.json");
      if (Files.exists(demos)) {
        ctx.contentType("application/json").result(Files.readString(demos));
      } else {
        ctx.json(Map.of("dataset", datasetName, "note",
            "none generated yet; run scripts/find_demo_queries.py"));
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
      // Quora ships 10k judged queries; the dropdown only needs a sample of them.
      int limit = ctx.queryParamAsClass("limit", Integer.class).getOrDefault(500);
      List<Map<String, Object>> out = new ArrayList<>();
      for (Map.Entry<String, String> e : dataset.queries().entrySet()) {
        if (out.size() >= limit) break;
        out.add(Trace.map("qid", e.getKey(), "text", e.getValue(),
            "relevantDocs", dataset.qrelsFor(e.getKey()).size()));
      }
      ctx.json(Trace.map("dataset", datasetName, "total", dataset.queries().size(),
          "returned", out.size(), "queries", out));
    });

    app.exception(NlpClient.NlpUnavailableException.class, (e, ctx) ->
        ctx.status(503).json(Map.of("error", "nlp sidecar unavailable", "detail", e.getMessage())));
    app.exception(SearchPipeline.QueryTooLongException.class, (e, ctx) ->
        ctx.status(400).json(Map.of("error", e.getMessage())));

    return app;
  }

  /**
   * Turns on HTTP basic auth when AUTH_USER and AUTH_PASS are both set. Enough to keep a
   * public demo from being crawled and burning CPU on reranking; put TLS in front of it.
   */
  private static void requireAuthIfConfigured(Javalin app, String user, String pass) {
    if (user == null || user.isBlank() || pass == null || pass.isBlank()) {
      System.out.println("auth: disabled (set AUTH_USER and AUTH_PASS to enable)");
      return;
    }
    byte[] expected = ("Basic " + java.util.Base64.getEncoder()
        .encodeToString((user + ":" + pass).getBytes(StandardCharsets.UTF_8)))
        .getBytes(StandardCharsets.UTF_8);

    app.before(ctx -> {
      if (Probes.PATHS.contains(ctx.path())) return;
      String got = ctx.header("Authorization");
      // Constant-time compare so the header cannot be guessed byte by byte.
      if (got == null
          || !java.security.MessageDigest.isEqual(expected, got.getBytes(StandardCharsets.UTF_8))) {
        ctx.header("WWW-Authenticate", "Basic realm=\"search demo\", charset=\"UTF-8\"");
        throw new io.javalin.http.UnauthorizedResponse();
      }
    });
    System.out.println("auth: basic auth required");
  }

  /** Body of POST /api/index-preview. {@code vector} defaults to true. */
  record PreviewRequest(String text, Boolean vector) {}

  private static SearchPipeline.Request parse(Context ctx) {
    String q = ctx.queryParamAsClass("q", String.class).getOrDefault("");
    SearchPipeline.Request d = SearchPipeline.Request.of(q);
    return new SearchPipeline.Request(
        q,
        mode(ctx.queryParamAsClass("mode", String.class).getOrDefault("hybrid")),
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

  private static SearchPipeline.Mode mode(String raw) {
    try {
      return SearchPipeline.Mode.valueOf(raw.toUpperCase(java.util.Locale.ROOT).replace("-", "_"));
    } catch (IllegalArgumentException e) {
      throw new io.javalin.http.BadRequestResponse(
          "unknown mode '" + raw + "'; expected hybrid, bm25_only or dense_only");
    }
  }

  /** df, idf, total term frequency, and which documents contain the term. */
  private static Map<String, Object> termInfo(DirectoryReader reader, DemoAnalyzer analyzer,
                                              String rawTerm, int limit) throws IOException {
    // Analyze the input so "deficiency" finds the stemmed term "defici".
    List<DemoAnalyzer.Token> analyzed = analyzer.analyze(Fields.CONTENT, rawTerm);
    String term = analyzed.isEmpty() ? rawTerm : analyzed.get(0).term();

    Term t = new Term(Fields.CONTENT, term);
    int df = reader.docFreq(t);
    long totalTf = reader.totalTermFreq(t);
    int numDocs = reader.numDocs();
    double idf = Math.log(1 + (numDocs - df + 0.5) / (df + 0.5));

    List<Map<String, Object>> docs = new ArrayList<>();
    var stored = reader.storedFields();
    for (var leaf : reader.leaves()) {
      if (docs.size() >= limit) break;
      Terms terms = leaf.reader().terms(Fields.CONTENT);
      if (terms == null) continue;
      TermsEnum te = terms.iterator();
      if (!te.seekExact(new BytesRef(term))) continue;
      PostingsEnum postings = te.postings(null, PostingsEnum.FREQS);
      while (postings.nextDoc() != DocIdSetIterator.NO_MORE_DOCS && docs.size() < limit) {
        int global = postings.docID() + leaf.docBase;
        docs.add(Trace.map(
            "docId", stored.document(global).get(Fields.ID),
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
