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

  /**
   * Interface to listen on. Loopback by default, because auth is off by default and a
   * laptop on conference Wi-Fi should not offer its API to the room. The container sets
   * 0.0.0.0 and leaves exposure to the published port. Not {@code HOST}: zsh sets that to
   * the machine's hostname.
   */
  public static String listenHost() { return env("LISTEN_HOST", "127.0.0.1"); }

  /** A request running longer than this fails /livez. NLP calls time out after 2 min. */
  public static Duration stuckRequestAfter() {
    return Duration.ofSeconds(Long.parseLong(env("STUCK_REQUEST_SECONDS", "600")));
  }

  /** How long a SIGTERM waits for in-flight requests. Keep under the pod's grace period. */
  public static Duration shutdownGrace() {
    return Duration.ofSeconds(Long.parseLong(env("SHUTDOWN_GRACE_SECONDS", "20")));
  }

  /** The git commit the image was built from (set by the Docker build), or null. */
  public static String commit() {
    String v = System.getenv("GIT_SHA");
    return v == null || v.isBlank() ? null : v;
  }

  /** Directory holding ui/index.html and its fonts, or null to serve only the API. */
  public static String uiDir() { return System.getenv("UI_DIR"); }

  private static String env(String key, String fallback) {
    String v = System.getenv(key);
    return v == null || v.isBlank() ? fallback : v;
  }

  private Config() {}
}
