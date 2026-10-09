package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.javalin.Javalin;

/** The HTTP surface, served by the real {@link Main#createApp} on a random port. */
class ApiTest {

  private static final ObjectMapper JSON = new ObjectMapper();
  private static final HttpClient HTTP = HttpClient.newBuilder()
      .version(HttpClient.Version.HTTP_1_1).build();

  private final TestIndex idx = TestIndex.get();
  private Javalin app;

  @TempDir Path tmp;

  @AfterEach
  void stop() {
    if (app != null) app.stop();
  }

  private String start(NlpClient nlp, String uiDir, String user, String pass, Duration stuck) {
    app = Main.createApp(idx.reader, idx.dataset, nlp, idx.indexDir, uiDir, user, pass, stuck);
    return start(app);
  }

  private static String start(Javalin app) {
    app.start("127.0.0.1", 0);
    return "http://127.0.0.1:" + app.port();
  }

  private String start() {
    return start(idx.nlp.client(), null, null, null, Duration.ofMinutes(10));
  }

  private static HttpResponse<String> get(String url, String... headers) throws Exception {
    var b = HttpRequest.newBuilder(URI.create(url)).timeout(Duration.ofSeconds(30));
    if (headers.length > 0) b.headers(headers);
    return HTTP.send(b.build(), HttpResponse.BodyHandlers.ofString());
  }

  private static HttpResponse<String> post(String url, String body) throws Exception {
    return HTTP.send(HttpRequest.newBuilder(URI.create(url))
        .header("content-type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
  }

  private static JsonNode json(HttpResponse<String> res) throws Exception {
    return JSON.readTree(res.body());
  }

  private static String enc(String s) {
    return java.net.URLEncoder.encode(s, StandardCharsets.UTF_8);
  }

  // ---- search ------------------------------------------------------------------

  @Test
  void searchReturnsTheFullTraceAsJson() throws Exception {
    String base = start();
    var res = get(base + "/api/search?q=" + enc(TestIndex.Q_VITAMIN));
    assertEquals(200, res.statusCode());
    assertTrue(res.headers().firstValue("content-type").orElse("").startsWith("application/json"));
    var body = json(res);
    List<String> stages = new java.util.ArrayList<>();
    body.get("stages").forEach(s -> stages.add(s.get("name").asText()));
    assertEquals(List.of("analyze", "bm25", "dense", "rrf", "rerank"), stages);
    assertEquals("q1", body.get("qid").asText());
    assertEquals("d1", body.get("stages").get(4).get("results").get(0).get("docId").asText());
    assertTrue(body.get("eval").get("judged").asBoolean());
  }

  @Test
  void searchKnobsArriveAndAreClamped() throws Exception {
    String base = start();
    var body = json(get(base + "/api/search?q=vitamin&mode=bm25-only&rerank=false&explain=false"
        + "&k1=99&b=0.3&bm25_n=999999&fuse_n=1&rerank_n=-4&trace=off"));
    assertEquals("bm25_only", body.get("mode").asText());
    assertEquals(10.0, body.get("params").get("k1").asDouble());
    assertEquals(0.3, body.get("params").get("b").asDouble(), 1e-6);
    assertEquals(SearchPipeline.MAX_CANDIDATES, body.get("params").get("bm25N").asInt());
    assertEquals(0, body.get("params").get("rerankN").asInt());
    assertNull(body.get("stages"));
    assertEquals(1, body.get("results").size());
  }

  @Test
  void badParametersAreClientErrors() throws Exception {
    String base = start();
    var mode = get(base + "/api/search?q=x&mode=quantum");
    assertEquals(400, mode.statusCode());
    assertTrue(mode.body().contains("unknown mode"), mode.body());
    assertEquals(400, get(base + "/api/search?q=x&k1=lots").statusCode());
    assertEquals(400, get(base + "/api/search?q=x&bm25_n=1e9").statusCode());
    assertEquals(400, get(base + "/api/term?t=x&limit=many").statusCode());
    var tooLong = get(base + "/api/search?q=" + enc("w ".repeat(1100)));
    assertEquals(400, tooLong.statusCode());
    assertTrue(tooLong.body().contains("at most"), tooLong.body());
  }

  @Test
  void modeParsingIgnoresTheDefaultLocale() throws Exception {
    // In a Turkish locale "hybrid".toUpperCase() is "HYBRİD", which matches no enum constant.
    var saved = java.util.Locale.getDefault();
    java.util.Locale.setDefault(java.util.Locale.forLanguageTag("tr-TR"));
    try {
      String base = start();
      assertEquals(200, get(base + "/api/search?q=x&mode=hybrid&rerank=false").statusCode());
    } finally {
      java.util.Locale.setDefault(saved);
    }
  }

  @Test
  void searchDegradesWhenTheSidecarIsDown() throws Exception {
    try (FakeNlp down = new FakeNlp()) {
      down.failStatus = 502;
      String base = start(down.client(), null, null, null, Duration.ofMinutes(10));
      var res = get(base + "/api/search?q=vitamin");
      assertEquals(200, res.statusCode());
      assertEquals(2, json(res).get("warnings").size());

      var health = json(get(base + "/api/health"));
      assertTrue(health.get("ok").asBoolean());
      assertEquals(TestIndex.DOCS.size(), health.get("docs").asInt());
    }
  }

  @Test
  void healthReportsTheSidecar() throws Exception {
    String base = start();
    var body = json(get(base + "/api/health"));
    assertTrue(body.get("nlp").get("ok").asBoolean());
    assertEquals(TestIndex.DATASET, body.get("dataset").asText());
  }

  @Test
  void healthSurvivesAnUnreachableSidecar() throws Exception {
    FakeNlp gone = new FakeNlp();
    NlpClient client = gone.client();
    gone.close();
    String base = start(client, null, null, null, Duration.ofMinutes(10));
    var body = json(get(base + "/api/health"));
    assertTrue(body.get("ok").asBoolean());
    assertFalse(body.get("nlp").get("ok").asBoolean());
  }

  // ---- inspection endpoints ----------------------------------------------------

  @Test
  void analyzeShowsTheChain() throws Exception {
    String base = start();
    var body = json(get(base + "/api/analyze?text=" + enc("The Cat's whiskers")));
    assertEquals(DemoAnalyzer.STEPS.length, body.get("chain").size());
    assertEquals("cat", body.get("tokens").get(0).get("term").asText());
    assertEquals(0, json(get(base + "/api/analyze")).get("tokens").size());
  }

  @Test
  void termAnalyzesItsInputAndCapsPostings() throws Exception {
    String base = start();
    var body = json(get(base + "/api/term?t=Deficiencies"));
    assertEquals("defici", body.get("analyzedTo").asText());
    assertEquals(2, body.get("df").asInt());
    assertEquals(2, body.get("postings").size());
    assertEquals(1, json(get(base + "/api/term?t=deficiency&limit=1")).get("postings").size());
    assertEquals(0, json(get(base + "/api/term?t=deficiency&limit=-5")).get("postings").size());
    assertEquals(2, json(get(base + "/api/term?t=deficiency&limit=99999999")).get("postings").size());
    assertEquals(0, json(get(base + "/api/term?t=nosuchterm")).get("df").asInt());
    assertEquals(200, get(base + "/api/term").statusCode());
  }

  @Test
  void docReturnsStoredFieldsTermsAndVector() throws Exception {
    String base = start();
    var body = json(get(base + "/api/doc/d7"));
    assertEquals("Running processes", body.get("title").asText());
    // "Running ... running" and "processes ... processes": two terms tie at tf 2.
    assertEquals(2, body.get("topTerms").get(0).get("tf").asInt());
    assertEquals(2, body.get("topTerms").get(1).get("tf").asInt());
    assertEquals("run", body.get("topTerms").get(1).get("term").asText());
    assertEquals(FakeNlp.DIM, body.get("vector").size());

    var missing = get(base + "/api/doc/nope");
    assertEquals(404, missing.statusCode());
    assertEquals(404, get(base + "/api/doc/" + enc("../../etc/passwd")).statusCode());
  }

  @Test
  void indexPreviewTakesAJsonBody() throws Exception {
    String base = start();
    var res = post(base + "/api/index-preview", "{\"text\":\"vitamin zebrafish\",\"vector\":false}");
    assertEquals(200, res.statusCode());
    var body = json(res);
    assertEquals(2, body.get("summary").get("uniqueTerms").asInt());
    assertNull(body.get("vector"));

    assertTrue(json(post(base + "/api/index-preview", "{\"text\":\"x\"}")).has("vector"));
    assertEquals(0, json(post(base + "/api/index-preview", "{}")).get("chars").asInt());
  }

  @Test
  void malformedPreviewBodiesAreRejectedNotCrashed() throws Exception {
    String base = start();
    for (String body : List.of("not json", "{\"text\": 5, \"vector\": \"maybe\"}", "[]")) {
      int status = post(base + "/api/index-preview", body).statusCode();
      assertTrue(status >= 400 && status < 500, body + " -> " + status);
    }
    // Javalin caps request bodies at 1 MB, which bounds what one preview can cost.
    String huge = "{\"text\":\"" + "a ".repeat(600_000) + "\",\"vector\":false}";
    assertEquals(413, post(base + "/api/index-preview", huge).statusCode());
  }

  @Test
  void queriesAreSampledInFileOrder() throws Exception {
    String base = start();
    var body = json(get(base + "/api/queries?limit=2"));
    assertEquals(3, body.get("total").asInt());
    assertEquals(2, body.get("returned").asInt());
    assertEquals("q1", body.get("queries").get(0).get("qid").asText());
    assertEquals(2, body.get("queries").get(0).get("relevantDocs").asInt());
    assertEquals(0, json(get(base + "/api/queries?limit=-1")).get("returned").asInt());
  }

  @Test
  void fileBackedEndpointsServeWhatTheIndexerAndScriptsWrote() throws Exception {
    String base = start();
    var stats = json(get(base + "/api/stats"));
    assertEquals(TestIndex.DOCS.size(), stats.get("docs").asInt());
    assertTrue(stats.has("bytesByComponent"));

    // Neither file exists for the test index.
    assertEquals(404, get(base + "/api/benchmark").statusCode());
    assertTrue(json(get(base + "/api/demo-queries")).has("note"));

    Path demos = idx.indexDir.resolve("demo_queries.json");
    Files.writeString(demos, "{\"dataset\":\"tiny\",\"lexical-win\":[]}");
    try {
      assertEquals("tiny", json(get(base + "/api/demo-queries")).get("dataset").asText());
    } finally {
      Files.delete(demos);
    }
  }

  // ---- probes ------------------------------------------------------------------

  @Test
  void readinessRunsARealQuery() throws Exception {
    String base = start();
    var res = get(base + "/readyz");
    assertEquals(200, res.statusCode());
    assertEquals(TestIndex.DOCS.size(), json(res).get("docs").asInt());
    assertEquals(200, get(base + "/livez").statusCode());
  }

  @Test
  void livenessFailsWhileARequestIsStuck() throws Exception {
    app = Main.createApp(idx.reader, idx.dataset, idx.nlp.client(), idx.indexDir, null, null, null,
        Duration.ofMillis(200));
    CountDownLatch release = new CountDownLatch(1);
    app.get("/test/hang", ctx -> {
      release.await(10, TimeUnit.SECONDS);
      ctx.result("done");
    });
    String base = start(app);

    CompletableFuture<HttpResponse<String>> hung = HTTP.sendAsync(
        HttpRequest.newBuilder(URI.create(base + "/test/hang")).build(),
        HttpResponse.BodyHandlers.ofString());
    Thread.sleep(400);
    var stuck = get(base + "/livez");
    assertEquals(503, stuck.statusCode());
    assertEquals(1, json(stuck).get("stuck").asInt());

    release.countDown();
    assertEquals(200, hung.get(10, TimeUnit.SECONDS).statusCode());
    var ok = get(base + "/livez");
    assertEquals(200, ok.statusCode(), ok.body());
    assertEquals(1, json(ok).get("inFlight").asInt(), "only the probe itself");
  }

  @Test
  void failedRequestsAreNotLeftInFlight() throws Exception {
    app = Main.createApp(idx.reader, idx.dataset, idx.nlp.client(), idx.indexDir, null, null, null,
        Duration.ofMillis(200));
    String base = start(app);
    assertEquals(400, get(base + "/api/search?q=x&mode=bogus").statusCode());
    assertEquals(404, get(base + "/api/doc/nope").statusCode());
    assertEquals(404, get(base + "/no/such/route").statusCode());
    Thread.sleep(300);
    assertEquals(200, get(base + "/livez").statusCode());
  }

  // ---- browser-facing security ---------------------------------------------------

  @Test
  void responsesForbidContentSniffing() throws Exception {
    String base = start();
    var res = get(base + "/api/search?q=x&rerank=false");
    assertEquals("nosniff", res.headers().firstValue("X-Content-Type-Options").orElse(null));
    assertEquals("no-referrer", res.headers().firstValue("Referrer-Policy").orElse(null));
  }

  @Test
  void corsOnlyAllowsTheUiDevServer() throws Exception {
    String base = start();
    var dev = get(base + "/api/health", "Origin", "http://localhost:5173");
    assertEquals("http://localhost:5173",
        dev.headers().firstValue("Access-Control-Allow-Origin").orElse(null));
    var evil = get(base + "/api/health", "Origin", "https://evil.example");
    assertTrue(evil.headers().firstValue("Access-Control-Allow-Origin").isEmpty());
  }

  @Test
  void servesTheUiWithoutEscapingItsDirectory() throws Exception {
    Path ui = Files.createDirectories(tmp.resolve("site/ui"));
    Files.writeString(ui.resolve("index.html"), "<html>demo</html>");
    Files.writeString(tmp.resolve("site/secret.txt"), "TOP SECRET");
    String base = start(idx.nlp.client(), ui.toString(), null, null, Duration.ofMinutes(10));

    assertTrue(get(base + "/").body().contains("demo"));
    for (String path : List.of("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt",
        "/%2e%2e%2fsecret.txt", "/ui/../../secret.txt")) {
      var res = HTTP.send(HttpRequest.newBuilder(URI.create(base + path)).build(),
          HttpResponse.BodyHandlers.ofString());
      assertFalse(res.body().contains("TOP SECRET"), path + " leaked the file");
    }
  }

  @Nested
  class BasicAuth {

    private String base;
    private final String good = "Basic " + Base64.getEncoder()
        .encodeToString("talk:s3cret pass".getBytes(StandardCharsets.UTF_8));

    private void startWithAuth() throws Exception {
      Path ui = Files.createDirectories(tmp.resolve("ui"));
      Files.writeString(ui.resolve("index.html"), "<html>demo</html>");
      base = start(idx.nlp.client(), ui.toString(), "talk", "s3cret pass", Duration.ofMinutes(10));
    }

    @Test
    void everyApiRouteAndTheUiRequireCredentials() throws Exception {
      startWithAuth();
      for (String path : List.of("/", "/index.html", "/api/health", "/api/search?q=x",
          "/api/doc/d1", "/api/term?t=x", "/api/stats", "/api/queries", "/no/such/route",
          "/livez/", "/LIVEZ", "/livez/../api/stats", "/api/../api/stats")) {
        var res = get(base + path);
        assertEquals(401, res.statusCode(), path);
        assertTrue(res.headers().firstValue("WWW-Authenticate").orElse("").startsWith("Basic"), path);
        assertFalse(res.body().contains("demo"), path);
      }
      assertEquals(401, post(base + "/api/index-preview", "{\"text\":\"x\"}").statusCode());
    }

    @Test
    void wrongCredentialsAreRejected() throws Exception {
      startWithAuth();
      for (String creds : List.of("talk:wrong", "TALK:s3cret pass", "talk:s3cret pas", "talk:s3cret pass ")) {
        String header = "Basic " + Base64.getEncoder().encodeToString(creds.getBytes(StandardCharsets.UTF_8));
        assertEquals(401, get(base + "/api/health", "Authorization", header).statusCode(), creds);
      }
      assertEquals(401, get(base + "/api/health", "Authorization", "Bearer x").statusCode());
    }

    @Test
    void rightCredentialsGetThrough() throws Exception {
      startWithAuth();
      assertEquals(200, get(base + "/api/health", "Authorization", good).statusCode());
      assertEquals(200, get(base + "/api/search?q=x&rerank=false", "Authorization", good).statusCode());
      assertTrue(get(base + "/", "Authorization", good).body().contains("demo"));
    }

    @Test
    void probesStayOpenForTheKubelet() throws Exception {
      startWithAuth();
      assertEquals(200, get(base + "/livez").statusCode());
      assertEquals(200, get(base + "/readyz").statusCode());
    }

    @Test
    void aBlankPasswordLeavesAuthOff() throws Exception {
      base = start(idx.nlp.client(), null, "talk", " ", Duration.ofMinutes(10));
      assertEquals(200, get(base + "/api/health").statusCode());
    }
  }
}
