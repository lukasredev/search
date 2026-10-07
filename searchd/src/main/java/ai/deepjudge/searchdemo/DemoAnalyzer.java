package ai.deepjudge.searchdemo;

import java.io.IOException;
import java.io.StringReader;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;

/**
 * The analyzer used for both indexing and querying, plus the machinery to replay its
 * work one filter at a time.
 *
 * <p>The chain is spelled out rather than inherited from {@link EnglishAnalyzer} for a
 * reason: {@link #trace} rebuilds it incrementally so the UI can show what each filter
 * does to the tokens. If you change the chain here, change {@link #STEPS} to match.
 */
public final class DemoAnalyzer extends Analyzer {

  /** Filter names in application order. Index 0 is the tokenizer itself. */
  static final String[] STEPS = {
    "StandardTokenizer", "EnglishPossessiveFilter", "LowerCaseFilter", "StopFilter", "PorterStemFilter"
  };

  private static final CharArraySet STOP_WORDS = EnglishAnalyzer.getDefaultStopSet();

  /** One token as it exists at some point in the chain. */
  public record Token(String term, int position, int start, int end) {}

  /** The token stream as it looks after {@code filter} has been applied. */
  public record ChainStep(String filter, List<String> tokens) {}

  @Override
  protected TokenStreamComponents createComponents(String field) {
    Tokenizer source = new StandardTokenizer();
    return new TokenStreamComponents(source, wrap(source, STEPS.length - 1));
  }

  /** Applies the first {@code filters} filters of the chain on top of the tokenizer. */
  private static TokenStream wrap(TokenStream source, int filters) {
    TokenStream ts = source;
    if (filters >= 1) ts = new EnglishPossessiveFilter(ts);
    if (filters >= 2) ts = new LowerCaseFilter(ts);
    if (filters >= 3) ts = new StopFilter(ts, STOP_WORDS);
    if (filters >= 4) ts = new PorterStemFilter(ts);
    return ts;
  }

  /**
   * Runs {@code text} through the chain once per prefix length, so the caller sees the
   * tokens after the tokenizer, then after each filter in turn. Re-running the whole
   * chain per step is wasteful but this is only ever called on a single query string.
   */
  public List<ChainStep> trace(String text) {
    List<ChainStep> steps = new ArrayList<>(STEPS.length);
    for (int i = 0; i < STEPS.length; i++) {
      Tokenizer source = new StandardTokenizer();
      source.setReader(new StringReader(text));
      try (TokenStream ts = wrap(source, i)) {
        steps.add(new ChainStep(STEPS[i], drain(ts).stream().map(Token::term).toList()));
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
    return steps;
  }

  /** The final analyzed tokens, with positions and offsets, as the index would store them. */
  public List<Token> analyze(String field, String text) {
    try (TokenStream ts = tokenStream(field, new StringReader(text))) {
      return drain(ts);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static List<Token> drain(TokenStream ts) throws IOException {
    CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
    PositionIncrementAttribute pos = ts.addAttribute(PositionIncrementAttribute.class);
    OffsetAttribute off = ts.addAttribute(OffsetAttribute.class);

    List<Token> tokens = new ArrayList<>();
    int position = -1;
    ts.reset();
    while (ts.incrementToken()) {
      position += pos.getPositionIncrement();
      tokens.add(new Token(term.toString(), position, off.startOffset(), off.endOffset()));
    }
    ts.end();
    return tokens;
  }
}
