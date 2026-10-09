package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.FSDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.fasterxml.jackson.databind.ObjectMapper;

class IndexerTest {

  @TempDir Path tmp;

  @Test
  @SuppressWarnings("unchecked")
  void writesBothIndexesIntoOneSegmentAndRecordsStats() throws Exception {
    TestIndex idx = TestIndex.get();
    assertEquals(TestIndex.DOCS.size(), idx.reader.numDocs());
    assertEquals(1, idx.reader.leaves().size());

    var fields = idx.reader.leaves().get(0).reader().getFieldInfos();
    assertEquals(FakeNlp.DIM, fields.fieldInfo(Fields.VECTOR).getVectorDimension());
    assertTrue(fields.fieldInfo(Fields.CONTENT).hasTermVectors());

    var hit = new IndexSearcher(idx.reader).search(new TermQuery(new Term(Fields.ID, "d1")), 1);
    var doc = idx.reader.storedFields().document(hit.scoreDocs[0].doc);
    assertEquals("Vitamin D deficiency", doc.get(Fields.TITLE));
    assertEquals("Low vitamin D levels are linked to bone disease.", doc.get(Fields.TEXT));

    Map<String, Object> stats = new ObjectMapper().readValue(
        idx.indexDir.resolve("index_stats.json").toFile(), Map.class);
    assertEquals(TestIndex.DOCS.size(), stats.get("docs"));
    assertEquals(true, stats.get("hasVectors"));
    assertEquals(FakeNlp.DIM, stats.get("vectorDim"));
    var components = (Map<String, Number>) stats.get("bytesByComponent");
    for (String c : new String[] {"inverted index", "hnsw vectors", "stored fields", "term vectors"}) {
      assertTrue(components.containsKey(c) && components.get(c).longValue() > 0, c + ": " + components);
    }
    long sum = components.values().stream().mapToLong(Number::longValue).sum();
    assertEquals(((Number) stats.get("bytesTotal")).longValue(), sum);
  }

  @Test
  void reusesTheEmbeddingCacheUntilTheModelChanges() throws Exception {
    Path data = TestIndex.writeDataset(tmp.resolve("data/tiny"));
    Path index = tmp.resolve("index/tiny");
    try (FakeNlp nlp = new FakeNlp()) {
      Indexer.build(data, index, nlp.client());
      int corpus = TestIndex.DOCS.size();
      assertEquals(corpus + 1, nlp.embeddedTexts.get(), "the probe plus every doc");
      assertTrue(Files.exists(data.resolve("vectors.f32.json")));

      // Same model, same corpus: only the probe is embedded.
      Indexer.build(data, index, nlp.client());
      assertEquals(corpus + 2, nlp.embeddedTexts.get());

      // A different model must not reuse the old model's vectors.
      nlp.embedModel = "fake/other";
      Indexer.build(data, index, nlp.client());
      assertEquals(2 * corpus + 3, nlp.embeddedTexts.get());

      // A cache from before model tracking is adopted rather than re-embedded.
      Files.delete(data.resolve("vectors.f32.json"));
      Indexer.build(data, index, nlp.client());
      assertEquals(2 * corpus + 4, nlp.embeddedTexts.get());
      assertTrue(Files.exists(data.resolve("vectors.f32.json")));
    }
  }

  @Test
  void aCorpusOfADifferentSizeIsReEmbedded() throws Exception {
    Path data = TestIndex.writeDataset(tmp.resolve("data/tiny"));
    Path index = tmp.resolve("index/tiny");
    try (FakeNlp nlp = new FakeNlp()) {
      Indexer.build(data, index, nlp.client());
      Files.writeString(data.resolve("corpus.jsonl"), "{\"id\":\"only\",\"title\":\"t\",\"text\":\"x\"}\n");
      int before = nlp.embeddedTexts.get();
      Indexer.build(data, index, nlp.client());
      assertEquals(before + 2, nlp.embeddedTexts.get());
      try (var r = DirectoryReader.open(FSDirectory.open(index))) {
        assertEquals(1, r.numDocs(), "the rebuild replaces the old index");
      }
    }
  }

  @Test
  void buildsALexicalOnlyIndexWithoutTheSidecar() throws Exception {
    Path data = TestIndex.writeDataset(tmp.resolve("data/tiny"));
    Path index = tmp.resolve("index/tiny");
    Indexer.build(data, index, (NlpClient) null);
    try (var r = DirectoryReader.open(FSDirectory.open(index))) {
      assertEquals(TestIndex.DOCS.size(), r.numDocs());
      assertNull(r.leaves().get(0).reader().getFieldInfos().fieldInfo(Fields.VECTOR));
      assertNull(new IndexStory(r, new DemoAnalyzer(), null).storedVector(0));
    }
    assertTrue(Files.readString(index.resolve("index_stats.json")).contains("\"hasVectors\" : false"));
  }

  @Test
  void failsBeforeTouchingTheIndexWhenTheSidecarIsDown() throws Exception {
    Path data = TestIndex.writeDataset(tmp.resolve("data/tiny"));
    Path index = tmp.resolve("index/tiny");
    try (FakeNlp nlp = new FakeNlp()) {
      nlp.failStatus = 503;
      assertThrows(NlpClient.NlpUnavailableException.class,
          () -> Indexer.build(data, index, nlp.client()));
    }
    assertTrue(Files.notExists(index));
  }

  @Test
  void storesTheVectorItWasGiven() throws Exception {
    TestIndex idx = TestIndex.get();
    var hit = new IndexSearcher(idx.reader).search(new TermQuery(new Term(Fields.ID, "d3")), 1);
    float[] stored = new IndexStory(idx.reader, new DemoAnalyzer(), null).storedVector(hit.scoreDocs[0].doc);
    var d3 = TestIndex.DOCS.get(2);
    assertArrayEquals(FakeNlp.vector(d3.get("title") + "\n\n" + d3.get("text")), stored, 1e-6f);
  }
}
