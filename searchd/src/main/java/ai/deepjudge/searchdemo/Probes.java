package ai.deepjudge.searchdemo;

import java.time.Duration;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;

import io.javalin.config.RoutesConfig;

/**
 * Kubernetes probes. {@code /livez} answers "should this process be restarted?" and
 * {@code /readyz} "should it get traffic?".
 *
 * <p>Neither calls the NLP sidecar. Without it, search degrades to BM25 with a warning,
 * so failing readiness would turn a degraded demo into an outage, and restarting this
 * process would not bring the sidecar back.
 */
final class Probes {

  /** Exempt from basic auth: the kubelet does not send credentials. */
  static final Set<String> PATHS = Set.of("/livez", "/readyz");

  /**
   * Start time of every request still being handled. A request that never finishes is
   * how this server fails: if a handler and its exception handler both throw, Javalin
   * 6.4 retries the exception handler forever and the thread spins at full CPU, while
   * every other endpoint keeps answering. Liveness is the only thing that notices.
   */
  private final Map<Object, Long> inFlight = new ConcurrentHashMap<>();
  private final long stuckNanos;

  Probes(Duration stuckAfter) {
    this.stuckNanos = stuckAfter.toNanos();
  }

  /** Register before any other handler, so every request is tracked. */
  void register(RoutesConfig routes, IndexSearcher searcher, String dataset) {
    routes.before(ctx -> inFlight.put(ctx.req(), System.nanoTime()));
    // After-handlers run even when the request threw, so only a stuck request stays.
    routes.after(ctx -> inFlight.remove(ctx.req()));

    routes.get("/livez", ctx -> {
      long now = System.nanoTime();
      long stuck = inFlight.values().stream().filter(t -> now - t > stuckNanos).count();
      ctx.status(stuck == 0 ? 200 : 503)
          .json(Map.of("ok", stuck == 0, "inFlight", inFlight.size(), "stuck", stuck));
    });

    routes.get("/readyz", ctx -> {
      // A real query, not just an open reader: proves the searcher can answer.
      int docs = searcher.count(new MatchAllDocsQuery());
      ctx.status(docs > 0 ? 200 : 503)
          .json(Map.of("ok", docs > 0, "dataset", dataset, "docs", docs));
    });
  }
}
