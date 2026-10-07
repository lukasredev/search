package ai.deepjudge.searchdemo;

import java.nio.file.Path;

/** Environment-driven configuration. Defaults are the local dev layout. */
public final class Config {

  public static String dataset() { return env("DATASET", "scifact"); }

  public static Path dataRoot() { return Path.of(env("DATA_ROOT", "../data")); }

  public static Path indexRoot() { return Path.of(env("INDEX_ROOT", "../data/index")); }

  public static String nlpUrl() { return env("NLP_URL", "http://localhost:8081"); }

  public static int port() { return Integer.parseInt(env("PORT", "8080")); }

  /** Directory of built UI assets to serve, or null to skip (dev uses the Vite server). */
  public static String uiDir() { return System.getenv("UI_DIR"); }

  private static String env(String key, String fallback) {
    String v = System.getenv(key);
    return v == null || v.isBlank() ? fallback : v;
  }

  private Config() {}
}
