package ai.deepjudge.searchdemo;

import java.nio.file.Path;
import java.time.Duration;

/** Environment-driven configuration. Defaults are the local dev layout. */
public final class Config {

  public static String dataset() { return env("DATASET", "cqadupstack-unix"); }

  public static Path dataRoot() { return Path.of(env("DATA_ROOT", "../data")); }

  public static Path indexRoot() { return Path.of(env("INDEX_ROOT", "../data/index")); }

  public static String nlpUrl() { return env("NLP_URL", "http://localhost:8081"); }

  public static int port() { return Integer.parseInt(env("PORT", "8080")); }

  /** A request running longer than this fails /livez. NLP calls time out after 2 min. */
  public static Duration stuckRequestAfter() {
    return Duration.ofSeconds(Long.parseLong(env("STUCK_REQUEST_SECONDS", "600")));
  }

  /** How long a SIGTERM waits for in-flight requests. Keep under the pod's grace period. */
  public static Duration shutdownGrace() {
    return Duration.ofSeconds(Long.parseLong(env("SHUTDOWN_GRACE_SECONDS", "20")));
  }

  /** Directory holding ui/index.html and its fonts, or null to serve only the API. */
  public static String uiDir() { return System.getenv("UI_DIR"); }

  private static String env(String key, String fallback) {
    String v = System.getenv(key);
    return v == null || v.isBlank() ? fallback : v;
  }

  private Config() {}
}
