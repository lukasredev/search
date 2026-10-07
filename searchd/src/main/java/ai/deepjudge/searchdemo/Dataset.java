package ai.deepjudge.searchdemo;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The normalized dataset on disk: {@code corpus.jsonl}, {@code queries.tsv}, {@code qrels.tsv}.
 * Written by {@code scripts/fetch_dataset.py}; nothing here knows about BEIR.
 */
public final class Dataset {

  public record Doc(String id, String title, String text) {
    /** What we index for BM25 and embed for the vector field. */
    String content() {
      return title.isEmpty() ? text : title + "\n\n" + text;
    }
  }

  private final String name;
  private final Map<String, String> queries;
  private final Map<String, Map<String, Integer>> qrels;

  private Dataset(String name, Map<String, String> queries, Map<String, Map<String, Integer>> qrels) {
    this.name = name;
    this.queries = queries;
    this.qrels = qrels;
  }

  public String name() { return name; }

  /** qid -> query text, in file order. */
  public Map<String, String> queries() { return queries; }

  /** qid -> (docid -> relevance grade). Absent docs are non-relevant by assumption. */
  public Map<String, Map<String, Integer>> qrels() { return qrels; }

  public Map<String, Integer> qrelsFor(String qid) {
    return qrels.getOrDefault(qid, Map.of());
  }

  /** Looks up the qid of a query whose text matches exactly, so eval can light up for free. */
  public String qidForText(String text) {
    String needle = text.strip();
    for (Map.Entry<String, String> e : queries.entrySet()) {
      if (e.getValue().equalsIgnoreCase(needle)) return e.getKey();
    }
    return null;
  }

  public static Dataset load(Path dir) throws IOException {
    Map<String, String> queries = new LinkedHashMap<>();
    for (String line : Files.readAllLines(dir.resolve("queries.tsv"), StandardCharsets.UTF_8)) {
      if (line.isBlank()) continue;
      String[] parts = line.split("\t", 2);
      queries.put(parts[0], parts[1]);
    }

    Map<String, Map<String, Integer>> qrels = new LinkedHashMap<>();
    for (String line : Files.readAllLines(dir.resolve("qrels.tsv"), StandardCharsets.UTF_8)) {
      if (line.isBlank()) continue;
      String[] parts = line.split("\t");
      int grade = (int) Double.parseDouble(parts[2]);
      if (grade > 0) {
        qrels.computeIfAbsent(parts[0], k -> new LinkedHashMap<>()).put(parts[1], grade);
      }
    }

    return new Dataset(dir.getFileName().toString(), queries, qrels);
  }

  /** Streams the corpus so a large dataset never has to fit in memory. */
  public static void forEachDoc(Path dir, java.util.function.Consumer<Doc> consumer) throws IOException {
    com.fasterxml.jackson.databind.ObjectMapper json = new com.fasterxml.jackson.databind.ObjectMapper();
    try (BufferedReader r = Files.newBufferedReader(dir.resolve("corpus.jsonl"), StandardCharsets.UTF_8)) {
      String line;
      while ((line = r.readLine()) != null) {
        if (line.isBlank()) continue;
        var node = json.readTree(line);
        consumer.accept(new Doc(
            node.get("id").asText(),
            node.path("title").asText(""),
            node.path("text").asText("")));
      }
    }
  }

  /** Reads the corpus fully into a list. Fine for the datasets we ship (<6k docs). */
  public static List<Doc> readDocs(Path dir) throws IOException {
    List<Doc> docs = new ArrayList<>();
    forEachDoc(dir, docs::add);
    return docs;
  }
}
