package ai.deepjudge.searchdemo;

/** Index field names, in one place so the indexer and the search stages cannot drift. */
public final class Fields {
  /** Document identifier from the dataset. Indexed untokenized, stored. */
  public static final String ID = "id";
  /** Title and body concatenated. This is what BM25 searches. */
  public static final String CONTENT = "content";
  /** Stored only, for display and for feeding the cross-encoder. */
  public static final String TITLE = "title";
  public static final String TEXT = "text";
  /** 384-dimensional normalized embedding of title + text. */
  public static final String VECTOR = "vector";

  private Fields() {}
}
