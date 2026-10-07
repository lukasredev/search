package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Builds the index. Deliberately linear and readable top to bottom — this class is
 * meant to be shown on a slide.
 *
 * <p>The thing worth pointing at: one {@link IndexWriter} writes both the inverted
 * index (the {@code content} {@link TextField}) and the HNSW graph (the {@code vector}
 * {@link KnnFloatVectorField}). A modern engine is one index with two retrieval paths,
 * not two separate systems.
 *
 * <pre>
 *   mvn exec:java -Dexec.mainClass=ai.deepjudge.searchdemo.Indexer -Dexec.args="scifact"
 * </pre>
 */
public final class Indexer {

  /** HNSW graph parameters. Small graph, fast build; plenty for a few thousand docs. */
  static final int HNSW_MAX_CONN = 16;
  static final int HNSW_BEAM_WIDTH = 100;

  /**
   * How many docs we hand the embedding service at a time. Short documents (Quora
   * questions average 60 characters) are dominated by per-request overhead at small
   * batch sizes: 64 gives 301 docs/s, 256 gives 385.
   */
  private static final int EMBED_BATCH = Integer.parseInt(
      System.getenv().getOrDefault("EMBED_BATCH", "256"));

  public static void main(String[] args) throws Exception {
    String dataset = args.length > 0 ? args[0] : Config.dataset();
    Path dataDir = Config.dataRoot().resolve(dataset);
    Path indexDir = Config.indexRoot().resolve(dataset);
    boolean withVectors = !"false".equalsIgnoreCase(System.getenv().getOrDefault("WITH_VECTORS", "true"));

    System.out.printf("indexing %s%n  from %s%n  into %s%n  vectors: %s%n",
        dataset, dataDir, indexDir, withVectors);
    long t0 = System.nanoTime();
    build(dataDir, indexDir, withVectors);
    System.out.printf("done in %.1fs%n", (System.nanoTime() - t0) / 1e9);
  }

  public static void build(Path dataDir, Path indexDir, boolean withVectors) throws IOException {
    List<Dataset.Doc> docs = Dataset.readDocs(dataDir);
    System.out.printf("read %d docs%n", docs.size());

    // Embed first, in batches, so a sidecar failure happens before we touch the index.
    // Cached to disk: re-indexing to try a different analyzer or HNSW setting should not
    // cost another pass over the embedding model.
    float[][] vectors = withVectors ? embedCached(docs, dataDir.resolve("vectors.f32")) : null;

    Files.createDirectories(indexDir);
    try (Directory dir = FSDirectory.open(indexDir);
         IndexWriter writer = new IndexWriter(dir, writerConfig())) {
      writer.deleteAll();

      FieldType contentType = contentFieldType();
      for (int i = 0; i < docs.size(); i++) {
        Dataset.Doc doc = docs.get(i);
        Document luceneDoc = new Document();

        // Untokenized identifier, so we can look a doc up exactly.
        luceneDoc.add(new StringField(Fields.ID, doc.id(), org.apache.lucene.document.Field.Store.YES));

        // The searchable text: analyzed, with term vectors so the UI can inspect postings.
        luceneDoc.add(new org.apache.lucene.document.Field(Fields.CONTENT, doc.content(), contentType));

        // Stored copies for display and for feeding the cross-encoder.
        luceneDoc.add(new StoredField(Fields.TITLE, doc.title()));
        luceneDoc.add(new StoredField(Fields.TEXT, doc.text()));

        if (vectors != null) {
          luceneDoc.add(new KnnFloatVectorField(Fields.VECTOR, vectors[i], VectorSimilarityFunction.COSINE));
        }

        writer.addDocument(luceneDoc);
        if ((i + 1) % 1000 == 0) System.out.printf("  indexed %d/%d%n", i + 1, docs.size());
      }

      // One segment keeps the index stable and inspectable, and makes the stats below
      // easy to attribute. You would not do this on a write-heavy production index.
      writer.forceMerge(1);
      writer.commit();
    }

    writeStats(indexDir, withVectors);
  }

  private static IndexWriterConfig writerConfig() {
    IndexWriterConfig config = new IndexWriterConfig(new DemoAnalyzer());
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    config.setRAMBufferSizeMB(256);
    // Indexing-time similarity only decides how norms (field lengths) are encoded, so
    // k1 and b remain free to change per query at search time.
    config.setSimilarity(new BM25Similarity());
    config.setCodec(hnswCodec());
    // Keep each component in its own file. Lucene would otherwise bundle a small
    // segment into one .cfs, which is better for file handles but would collapse the
    // inverted-index / HNSW / stored-fields byte breakdown we report below into a
    // single opaque number.
    config.setUseCompoundFile(false);
    config.getMergePolicy().setNoCFSRatio(0.0);
    return config;
  }

  /** Default codec, but with explicit HNSW parameters for the vector field. */
  private static Codec hnswCodec() {
    return new Lucene104Codec() {
      private final KnnVectorsFormat knn = new Lucene99HnswVectorsFormat(HNSW_MAX_CONN, HNSW_BEAM_WIDTH);

      @Override
      public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
        return Fields.VECTOR.equals(field) ? knn : super.getKnnVectorsFormatForField(field);
      }
    };
  }

  private static FieldType contentFieldType() {
    FieldType type = new FieldType(TextField.TYPE_NOT_STORED);
    type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
    // Term vectors are what let the UI show "which terms does this document contain,
    // and where" without re-analyzing the stored text.
    type.setStoreTermVectors(true);
    type.setStoreTermVectorPositions(true);
    type.setStoreTermVectorOffsets(true);
    type.freeze();
    return type;
  }

  /** Reads the embedding cache if it matches the corpus, otherwise embeds and writes it. */
  private static float[][] embedCached(List<Dataset.Doc> docs, Path cache) throws IOException {
    if (Files.exists(cache)) {
      float[][] cached = readVectors(cache, docs.size());
      if (cached != null) {
        System.out.printf("reusing %d cached vectors from %s%n", cached.length, cache);
        return cached;
      }
      System.out.println("cache at " + cache + " does not match the corpus; re-embedding");
    }
    float[][] vectors = embedAll(docs);
    writeVectors(cache, vectors);
    System.out.println("cached vectors -> " + cache);
    return vectors;
  }

  /** Flat little format: int count, int dim, then count*dim big-endian floats. */
  private static float[][] readVectors(Path path, int expectedCount) throws IOException {
    try (var in = new java.io.DataInputStream(new java.io.BufferedInputStream(Files.newInputStream(path)))) {
      int count = in.readInt();
      int dim = in.readInt();
      if (count != expectedCount) return null;
      float[][] out = new float[count][dim];
      for (int i = 0; i < count; i++) {
        for (int d = 0; d < dim; d++) out[i][d] = in.readFloat();
      }
      return out;
    } catch (java.io.EOFException e) {
      return null;
    }
  }

  private static void writeVectors(Path path, float[][] vectors) throws IOException {
    try (var out = new java.io.DataOutputStream(new java.io.BufferedOutputStream(Files.newOutputStream(path)))) {
      out.writeInt(vectors.length);
      out.writeInt(vectors[0].length);
      for (float[] v : vectors) {
        for (float f : v) out.writeFloat(f);
      }
    }
  }

  private static float[][] embedAll(List<Dataset.Doc> docs) {
    NlpClient nlp = new NlpClient(Config.nlpUrl(), java.time.Duration.ofMinutes(10));
    float[][] out = new float[docs.size()][];
    long t0 = System.nanoTime();

    for (int start = 0; start < docs.size(); start += EMBED_BATCH) {
      int end = Math.min(start + EMBED_BATCH, docs.size());
      List<String> batch = new ArrayList<>(end - start);
      for (int i = start; i < end; i++) batch.add(docs.get(i).content());

      NlpClient.EmbedResult result = nlp.embed(batch);
      for (int i = start; i < end; i++) out[i] = result.vectors()[i - start];

      double elapsed = (System.nanoTime() - t0) / 1e9;
      double rate = end / Math.max(elapsed, 1e-6);
      System.out.printf("\r  embedded %d/%d  (%.0f docs/s, eta %.0fs)   ",
          end, docs.size(), rate, (docs.size() - end) / Math.max(rate, 1e-6));
    }
    System.out.println();
    return out;
  }

  /**
   * Writes {@code index_stats.json}: doc/term counts plus the on-disk byte breakdown
   * grouped into inverted index vs HNSW graph vs stored fields. This is the data behind
   * the "what does semantic search actually cost you?" slide.
   */
  private static void writeStats(Path indexDir, boolean withVectors) throws IOException {
    Map<String, Object> stats = new LinkedHashMap<>();

    try (Directory dir = FSDirectory.open(indexDir);
         DirectoryReader reader = DirectoryReader.open(dir)) {
      stats.put("docs", reader.numDocs());
      stats.put("segments", reader.leaves().size());

      long terms = 0;
      long postings = 0;
      for (LeafReader leaf : reader.leaves().stream().map(c -> c.reader()).toList()) {
        Terms t = leaf.terms(Fields.CONTENT);
        if (t != null) {
          terms += t.size();
          postings += t.getSumDocFreq();
        }
      }
      stats.put("uniqueTerms", terms);
      stats.put("postings", postings);
      stats.put("hasVectors", withVectors);
      if (withVectors) {
        stats.put("vectorDim", reader.leaves().get(0).reader()
            .getFieldInfos().fieldInfo(Fields.VECTOR).getVectorDimension());
        stats.put("hnswMaxConn", HNSW_MAX_CONN);
        stats.put("hnswBeamWidth", HNSW_BEAM_WIDTH);
      }
    }

    // Attribute every file in the directory to a component by its extension.
    Map<String, Long> byComponent = new TreeMap<>();
    Map<String, Long> byExtension = new TreeMap<>();
    long total = 0;
    try (var files = Files.list(indexDir)) {
      for (Path f : files.toList()) {
        if (Files.isDirectory(f)) continue;
        long size = Files.size(f);
        String ext = f.getFileName().toString().replaceFirst("^.*\\.", "");
        byExtension.merge(ext, size, Long::sum);
        byComponent.merge(component(ext), size, Long::sum);
        total += size;
      }
    }
    stats.put("bytesTotal", total);
    stats.put("bytesByComponent", byComponent);
    stats.put("bytesByExtension", byExtension);

    Path out = indexDir.resolve("index_stats.json");
    Files.writeString(out, new com.fasterxml.jackson.databind.ObjectMapper()
        .writerWithDefaultPrettyPrinter().writeValueAsString(stats), StandardCharsets.UTF_8);

    System.out.println("index stats -> " + out);
    System.out.printf("  %d docs, %s unique terms, %.1f MB total%n",
        stats.get("docs"), stats.get("uniqueTerms"), total / 1e6);
    byComponent.forEach((k, v) -> System.out.printf("    %-16s %7.1f MB%n", k, v / 1e6));
  }

  /** Maps a Lucene file extension to the component it belongs to. */
  private static String component(String ext) {
    return switch (ext) {
      case "tim", "tip", "tmd", "doc", "pos", "pay", "psm" -> "inverted index";
      case "vec", "vem", "vex", "veq", "vemf", "vemq" -> "hnsw vectors";
      case "fdt", "fdx", "fdm" -> "stored fields";
      case "tvd", "tvx", "tvm" -> "term vectors";
      case "nvd", "nvm" -> "norms";
      case "dvd", "dvm" -> "doc values";
      case "json", "fnm", "si", "lock" -> "metadata";
      default -> ext.startsWith("segments") ? "metadata" : "other (" + ext + ")";
    };
  }
}
