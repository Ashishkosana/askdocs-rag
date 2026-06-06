from askdocs.config import Settings
from askdocs.pipeline import RagPipeline
from askdocs.store import Hit


def test_answer_grounds_in_retrieved_context(
    fake_embedder, fake_store_factory, fake_client_factory
):
    store = fake_store_factory(
        hits=[Hit("Growth is $49 per seat.", "nimbus_billing.md", 2, 0.91)], count=10
    )
    client = fake_client_factory("The Growth plan is $49 per seat per month. [1]")
    pipe = RagPipeline(Settings(), embedder=fake_embedder, store=store, client=client)

    ans = pipe.answer("How much is the Growth plan?")

    assert "49" in ans.text
    assert ans.contexts[0].source == "nimbus_billing.md"
    assert ans.usage.input_tokens == 100 and ans.usage.output_tokens == 20

    # the retrieved passage must actually be in the prompt sent to Claude
    sent = client.messages.calls[0]
    assert "Growth is $49 per seat." in sent["messages"][0]["content"]
    assert sent["model"] == "claude-opus-4-8"
    assert sent["thinking"] == {"type": "adaptive"}


def test_index_embeds_and_stores(
    tmp_path, fake_embedder, fake_store_factory, fake_client_factory
):
    (tmp_path / "doc.md").write_text("Alpha fact here.\n\nBeta fact here.")
    store = fake_store_factory()
    pipe = RagPipeline(
        Settings(), embedder=fake_embedder, store=store, client=fake_client_factory()
    )

    added = pipe.index(tmp_path)

    assert added >= 1
    assert store.added  # chunks + embeddings were upserted
    chunks, embeddings = store.added[0]
    assert len(chunks) == len(embeddings)
