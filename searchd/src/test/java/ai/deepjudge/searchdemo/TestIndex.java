package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.FSDirectory;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * A tiny dataset in the on-disk format {@code fetch_dataset.py} writes, indexed once per
 * test run by the real {@link Indexer} against a {@link FakeNlp}.
 */
final class TestIndex {

  static final String DATASET = "tiny";

  /** The judged queries. q1 has a grade-0 judgment, which must be dropped on load. */
  static final String Q_VITAMIN = "vitamin d deficiency";
  static final String Q_SHELL = "difference between $@ and $* in bash";

  /** Long enough to be snipped and to count as truncated by the fake reranker. */
  static final String LONG_TEXT = ("Running a long benchmark takes patience and a lot of coffee. ")
      .repeat(12).strip();

  static final List<Map<String, String>> DOCS = List.of(
      doc("d1", "Vitamin D deficiency", "Low vitamin D levels are linked to bone disease."),
      doc("d2", "Bone health", "Calcium and vitamin intake keep bones strong."),
      doc("d3", "Bash positional parameters", "The difference between $@ and $* in shell scripts."),
      doc("d4", "Shell scripting", "Quoting arguments in bash scripts avoids word splitting."),
      doc("d5", "curl vs wget", "What is the difference between curl and wget downloads?"),
      doc("d6", "Cats", "The cat's whiskers are sensitive."),
      doc("d7", "Running processes", "The runner was running quickly while processes ran."),
      doc("d8", "", "Untitled document about deficiency in sleep."),
      doc("d9", "Benchmarks", LONG_TEXT));

  final Path root;
  final Path dataDir;
  final Path indexDir;
  final DirectoryReader reader;
  final IndexSearcher searcher;
  final Dataset dataset;
  final FakeNlp nlp;

  private static TestIndex instance;

  /** Built once and shared: the index is read-only once written. */
  static synchronized TestIndex get() {
    if (instance == null) {
      try {
        instance = new TestIndex();
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
    return instance;
  }

  private TestIndex() throws IOException {
    root = Files.createTempDirectory("searchd-test");
    dataDir = writeDataset(root.resolve("data").resolve(DATASET));
    indexDir = root.resolve("index").resolve(DATASET);
    nlp = new FakeNlp();
    Indexer.build(dataDir, indexDir, nlp.client());
    reader = DirectoryReader.open(FSDirectory.open(indexDir));
    searcher = new IndexSearcher(reader);
    dataset = Dataset.load(dataDir);
  }

  SearchPipeline pipeline() {
    return pipeline(nlp.client());
  }

  SearchPipeline pipeline(NlpClient client) {
    return new SearchPipeline(searcher, new DemoAnalyzer(), dataset, client);
  }

  static Path writeDataset(Path dir) throws IOException {
    Files.createDirectories(dir);
    ObjectMapper json = new ObjectMapper();
    StringBuilder corpus = new StringBuilder();
    for (Map<String, String> d : DOCS) corpus.append(json.writeValueAsString(d)).append('\n');
    Files.writeString(dir.resolve("corpus.jsonl"), corpus);
    Files.writeString(dir.resolve("queries.tsv"),
        "q1\t" + Q_VITAMIN + "\n" + "q2\t" + Q_SHELL + "\n" + "q3\tcats and dogs\n");
    Files.writeString(dir.resolve("qrels.tsv"),
        "q1\td1\t2\n" + "q1\td2\t1\n" + "q1\td6\t0\n" + "q2\td3\t1\n" + "q3\td6\t1\n");
    return dir;
  }

  private static Map<String, String> doc(String id, String title, String text) {
    return Map.of("id", id, "title", title, "text", text);
  }
}
