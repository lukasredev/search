package ai.deepjudge.searchdemo;

/**
 * A document moving through the pipeline.
 *
 * @param docId   dataset identifier
 * @param luceneDoc internal Lucene docid, needed for explain() and stored-field lookups
 * @param score   score from whichever stage produced this candidate
 * @param rank    1-based rank within that stage
 */
public record Candidate(String docId, int luceneDoc, double score, int rank) {

  public Candidate withRank(int newRank) {
    return new Candidate(docId, luceneDoc, score, newRank);
  }

  public Candidate withScore(double newScore) {
    return new Candidate(docId, luceneDoc, newScore, rank);
  }
}
