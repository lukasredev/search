package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DatasetTest {

  @TempDir Path tmp;

  @Test
  void loadsQueriesAndPositiveQrelsOnly() throws Exception {
    Dataset ds = Dataset.load(TestIndex.writeDataset(tmp.resolve("tiny")));
    assertEquals("tiny", ds.name());
    assertEquals(List.of("q1", "q2", "q3"), List.copyOf(ds.queries().keySet()));
    // The grade-0 judgment of d6 is not a relevant document.
    assertEquals(Map.of("d1", 2, "d2", 1), ds.qrelsFor("q1"));
    assertEquals(Map.of(), ds.qrelsFor("nope"));
  }

  @Test
  void matchesQueryTextIgnoringCaseAndSurroundingSpace() throws Exception {
    Dataset ds = Dataset.load(TestIndex.writeDataset(tmp.resolve("tiny")));
    assertEquals("q1", ds.qidForText("  Vitamin D Deficiency "));
    assertNull(ds.qidForText("vitamin d"));
  }

  @Test
  void readsTheCorpusAndJoinsTitleAndText() throws Exception {
    Path dir = TestIndex.writeDataset(tmp.resolve("tiny"));
    // Missing title/text fields and blank lines are tolerated.
    Files.writeString(dir.resolve("corpus.jsonl"), Files.readString(dir.resolve("corpus.jsonl"))
        + "\n{\"id\":\"bare\"}\n");
    var docs = Dataset.readDocs(dir);
    assertEquals(TestIndex.DOCS.size() + 1, docs.size());
    assertEquals("Vitamin D deficiency\n\nLow vitamin D levels are linked to bone disease.",
        docs.get(0).content());
    assertEquals("Untitled document about deficiency in sleep.", docs.get(7).content());
    assertEquals("", docs.get(docs.size() - 1).content());
  }

  @Test
  void acceptsFractionalGrades() throws Exception {
    Path dir = TestIndex.writeDataset(tmp.resolve("tiny"));
    Files.writeString(dir.resolve("qrels.tsv"), "q1\td1\t2.0\n");
    assertEquals(Map.of("d1", 2), Dataset.load(dir).qrelsFor("q1"));
  }
}
