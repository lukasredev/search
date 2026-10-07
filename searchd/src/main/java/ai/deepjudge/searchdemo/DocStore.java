package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.StoredFields;

/**
 * Per-request accumulator for the documents any stage touched.
 *
 * <p>Stages record which docs they saw; at the end the API emits one {@code docs} map so
 * the four stage rankings can reference documents by id without repeating their text
 * four times over.
 */
public final class DocStore {

  /** Longest snippet we send to the browser. The full text goes to the reranker only. */
  private static final int SNIPPET_CHARS = 420;

  private final IndexReader reader;
  private final Map<String, Integer> luceneDocByDocId = new LinkedHashMap<>();

  public DocStore(IndexReader reader) {
    this.reader = reader;
  }

  public void remember(int luceneDoc, String docId) {
    luceneDocByDocId.putIfAbsent(docId, luceneDoc);
  }

  public Integer luceneDoc(String docId) {
    return luceneDocByDocId.get(docId);
  }

  /** Full title + body, as handed to the cross-encoder. */
  public String fullText(String docId) throws IOException {
    Document d = reader.storedFields().document(luceneDocByDocId.get(docId));
    String title = d.get(Fields.TITLE);
    String text = d.get(Fields.TEXT);
    return (title == null || title.isEmpty()) ? text : title + "\n\n" + text;
  }

  /**
   * Builds the response's document map.
   *
   * @param qrels relevance grades for the current query, used to mark gold documents so
   *              the UI can show relevance migrating up the ranking
   */
  public Map<String, Object> toJson(Map<String, Integer> qrels) throws IOException {
    StoredFields stored = reader.storedFields();
    Map<String, Object> out = new LinkedHashMap<>();
    for (Map.Entry<String, Integer> e : luceneDocByDocId.entrySet()) {
      Document d = stored.document(e.getValue());
      String text = d.get(Fields.TEXT);
      out.put(e.getKey(), Trace.map(
          "title", d.get(Fields.TITLE),
          "snippet", snippet(text),
          "chars", text == null ? 0 : text.length(),
          "luceneDoc", e.getValue(),
          "relevance", qrels.getOrDefault(e.getKey(), 0)));
    }
    return out;
  }

  private static String snippet(String text) {
    if (text == null) return "";
    if (text.length() <= SNIPPET_CHARS) return text;
    int cut = text.lastIndexOf(' ', SNIPPET_CHARS);
    return text.substring(0, cut < SNIPPET_CHARS / 2 ? SNIPPET_CHARS : cut) + "…";
  }
}
