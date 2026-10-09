package ai.deepjudge.searchdemo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.List;

import org.junit.jupiter.api.Test;

class NlpClientTest {

  @Test
  void embedsAndReranksOverHttp() throws Exception {
    try (FakeNlp nlp = new FakeNlp()) {
      var e = nlp.client().embed(List.of("a b", "c"));
      assertEquals(FakeNlp.EMBED_MODEL, e.model());
      assertEquals(FakeNlp.DIM, e.dim());
      assertEquals(2, e.vectors().length);

      var r = nlp.client().rerank("vitamin d", List.of("vitamin d", "nothing"));
      assertEquals(List.of(2.0, 0.0), r.items().stream().map(NlpClient.RerankItem::score).toList());
      assertEquals(2, r.items().get(0).doc_tokens());
      assertEquals(512, r.max_length());

      assertEquals(true, nlp.client().health().get("ok"));
    }
  }

  @Test
  void toleratesATrailingSlashInTheBaseUrl() throws Exception {
    try (FakeNlp nlp = new FakeNlp()) {
      var client = new NlpClient(nlp.url() + "/", Duration.ofSeconds(5));
      assertEquals(1, client.embedQuery("x").vectors().length);
      assertEquals(true, client.health().get("ok"));
    }
  }

  @Test
  void anErrorStatusBecomesUnavailableWithTheBody() throws Exception {
    try (FakeNlp nlp = new FakeNlp()) {
      nlp.failStatus = 500;
      var e = assertThrows(NlpClient.NlpUnavailableException.class, () -> nlp.client().embedQuery("x"));
      assertTrue(e.getMessage().contains("500") && e.getMessage().contains("model exploded"),
          e.getMessage());
    }
  }

  @Test
  void aRefusedConnectionBecomesUnavailable() throws Exception {
    FakeNlp nlp = new FakeNlp();
    NlpClient client = nlp.client();
    nlp.close();
    assertThrows(NlpClient.NlpUnavailableException.class, () -> client.rerank("q", List.of("d")));
  }
}
