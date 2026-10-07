package ai.deepjudge.searchdemo;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The record of what each pipeline stage did. Every number the UI displays comes from
 * here, so "open the network tab" is itself a demo step — the frontend invents nothing.
 */
public final class Trace {

  /** One pipeline stage: its knobs, how long it took, and the ranking it produced. */
  public record Stage(
      String name,
      double ms,
      Map<String, Object> params,
      Map<String, Object> detail,
      List<Map<String, Object>> results) {}

  private final List<Stage> stages = new ArrayList<>();

  public List<Stage> stages() { return stages; }

  public void add(String name, double ms, Map<String, Object> params,
                  Map<String, Object> detail, List<Map<String, Object>> results) {
    stages.add(new Stage(name, round(ms), params, detail, results));
  }

  public static double round(double v) {
    return Math.round(v * 100.0) / 100.0;
  }

  /** Ordered map helper, so the JSON keys come out in a readable order. */
  public static Map<String, Object> map(Object... kv) {
    Map<String, Object> m = new LinkedHashMap<>();
    for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
    return m;
  }

  /** Simple wall-clock timer in milliseconds. */
  public static final class Timer {
    private final long start = System.nanoTime();

    public double ms() { return (System.nanoTime() - start) / 1e6; }
  }
}
