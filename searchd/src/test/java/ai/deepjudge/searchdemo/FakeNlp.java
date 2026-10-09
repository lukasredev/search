package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * A stand-in for the Python sidecar, speaking the same JSON over real HTTP.
 *
 * <p>Embeddings are a normalized hashed bag of words, so texts sharing words are close
 * and the HNSW search has a meaningful answer. The reranker scores a document by how many
 * query words it contains. Both are deterministic, which is all the tests need.
 */
final class FakeNlp implements AutoCloseable {

  static final int DIM = 32;
  static final String EMBED_MODEL = "fake/bag-of-words";
  static final String RERANK_MODEL = "fake/word-overlap";
  /** Docs with more words than this are reported as truncated, like the 512-token window. */
  static final int MAX_DOC_WORDS = 40;

  private static final ObjectMapper JSON = new ObjectMapper();

  private final HttpServer server;
  final AtomicInteger embedCalls = new AtomicInteger();
  final AtomicInteger embeddedTexts = new AtomicInteger();
  final AtomicInteger rerankCalls = new AtomicInteger();

  /** Non-zero makes every model endpoint answer with this status. */
  volatile int failStatus = 0;
  /** Makes /rerank return one score fewer than it was given documents. */
  volatile boolean shortRerank = false;
  volatile String embedModel = EMBED_MODEL;

  FakeNlp() throws IOException {
    server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext("/embed", this::embed);
    server.createContext("/rerank", this::rerank);
    server.createContext("/health", ex -> reply(ex, 200, Map.of("ok", true,
        "embed_model", embedModel, "rerank_model", RERANK_MODEL)));
    server.start();
  }

  String url() {
    return "http://127.0.0.1:" + server.getAddress().getPort();
  }

  NlpClient client() {
    return new NlpClient(url(), Duration.ofSeconds(10));
  }

  static List<String> words(String text) {
    List<String> out = new ArrayList<>();
    for (String w : text.toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
      if (!w.isEmpty()) out.add(w);
    }
    return out;
  }

  static float[] vector(String text) {
    float[] v = new float[DIM];
    for (String w : words(text)) v[Math.floorMod(w.hashCode(), DIM)] += 1;
    double norm = 0;
    for (float f : v) norm += f * f;
    // An empty text still needs a valid unit vector: COSINE rejects the zero vector.
    if (norm == 0) {
      v[0] = 1;
      return v;
    }
    for (int i = 0; i < DIM; i++) v[i] /= (float) Math.sqrt(norm);
    return v;
  }

  static double overlap(String query, String doc) {
    List<String> docWords = words(doc);
    double score = 0;
    for (String w : words(query)) if (docWords.contains(w)) score++;
    return score;
  }

  private void embed(HttpExchange ex) throws IOException {
    if (failed(ex)) return;
    embedCalls.incrementAndGet();
    JsonNode body = JSON.readTree(ex.getRequestBody());
    List<float[]> vectors = new ArrayList<>();
    for (JsonNode t : body.get("texts")) vectors.add(vector(t.asText()));
    embeddedTexts.addAndGet(vectors.size());
    reply(ex, 200, Map.of("model", embedModel, "dim", DIM, "ms", 1.0, "vectors", vectors));
  }

  private void rerank(HttpExchange ex) throws IOException {
    if (failed(ex)) return;
    rerankCalls.incrementAndGet();
    JsonNode body = JSON.readTree(ex.getRequestBody());
    String query = body.get("query").asText();
    List<Map<String, Object>> items = new ArrayList<>();
    for (JsonNode d : body.get("docs")) {
      int n = words(d.asText()).size();
      items.add(Map.of("score", overlap(query, d.asText()), "doc_tokens", n,
          "truncated", n > MAX_DOC_WORDS));
    }
    if (shortRerank && !items.isEmpty()) items.remove(items.size() - 1);
    reply(ex, 200, Map.of("model", RERANK_MODEL, "max_length", 512, "ms", 2.0, "items", items));
  }

  private boolean failed(HttpExchange ex) throws IOException {
    int status = failStatus;
    if (status == 0) return false;
    byte[] msg = "model exploded".getBytes(StandardCharsets.UTF_8);
    ex.sendResponseHeaders(status, msg.length);
    ex.getResponseBody().write(msg);
    ex.close();
    return true;
  }

  private static void reply(HttpExchange ex, int status, Object body) throws IOException {
    byte[] bytes = JSON.writeValueAsBytes(body);
    ex.getResponseHeaders().add("content-type", "application/json");
    ex.sendResponseHeaders(status, bytes.length);
    ex.getResponseBody().write(bytes);
    ex.close();
  }

  @Override
  public void close() {
    server.stop(0);
  }
}
