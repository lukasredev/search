package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;

/** Thin HTTP client for the Python sidecar that owns the two transformer models. */
public final class NlpClient {

  @JsonIgnoreProperties(ignoreUnknown = true)
  public record EmbedResult(String model, int dim, double ms, float[][] vectors) {}

  @JsonIgnoreProperties(ignoreUnknown = true)
  public record RerankItem(double score, int doc_tokens, boolean truncated) {}

  @JsonIgnoreProperties(ignoreUnknown = true)
  public record RerankResult(String model, int max_length, double ms, List<RerankItem> items) {}

  private static final ObjectMapper JSON = new ObjectMapper();

  private final URI base;
  private final HttpClient http;

  public NlpClient(String baseUrl, Duration timeout) {
    this.base = URI.create(baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl);
    // uvicorn/h11 drops the request body when Java's client attempts its h2c upgrade,
    // so pin HTTP/1.1 rather than debugging a silently empty POST.
    this.http = HttpClient.newBuilder()
        .version(HttpClient.Version.HTTP_1_1)
        .connectTimeout(Duration.ofSeconds(5))
        .build();
    this.timeout = timeout;
  }

  private final Duration timeout;

  public EmbedResult embed(List<String> texts) {
    return post("/embed", Map.of("texts", texts), EmbedResult.class);
  }

  /** Convenience for the one text we embed on the hot path: the query. */
  public EmbedResult embedQuery(String query) {
    return embed(List.of(query));
  }

  public RerankResult rerank(String query, List<String> docs) {
    return post("/rerank", Map.of("query", query, "docs", docs), RerankResult.class);
  }

  public Map<String, Object> health() {
    HttpRequest req = HttpRequest.newBuilder(base.resolve(base.getPath() + "/health"))
        .timeout(Duration.ofSeconds(5)).GET().build();
    return send(req, new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {});
  }

  private <T> T post(String path, Object body, Class<T> type) {
    try {
      HttpRequest req = HttpRequest.newBuilder(URI.create(base + path))
          .timeout(timeout)
          .header("content-type", "application/json")
          .POST(HttpRequest.BodyPublishers.ofByteArray(JSON.writeValueAsBytes(body)))
          .build();
      HttpResponse<byte[]> res = http.send(req, HttpResponse.BodyHandlers.ofByteArray());
      if (res.statusCode() != 200) {
        throw new NlpUnavailableException(
            "nlp " + path + " returned " + res.statusCode() + ": " + new String(res.body()));
      }
      return JSON.readValue(res.body(), type);
    } catch (IOException e) {
      throw new NlpUnavailableException("nlp " + path + " failed: " + e.getMessage(), e);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new NlpUnavailableException("interrupted calling nlp " + path, e);
    }
  }

  private <T> T send(HttpRequest req, com.fasterxml.jackson.core.type.TypeReference<T> type) {
    try {
      HttpResponse<byte[]> res = http.send(req, HttpResponse.BodyHandlers.ofByteArray());
      return JSON.readValue(res.body(), type);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(e);
    }
  }

  /** Signals the sidecar is down, so the API can degrade to lexical-only instead of 500ing. */
  public static final class NlpUnavailableException extends RuntimeException {
    public NlpUnavailableException(String message) { super(message); }
    public NlpUnavailableException(String message, Throwable cause) { super(message, cause); }
  }
}
