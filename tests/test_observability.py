from askdocs.observability import Usage, cost_usd


def test_input_and_output_pricing():
    assert cost_usd("claude-opus-4-8", 1_000_000, 0) == 5.0
    assert cost_usd("claude-opus-4-8", 0, 1_000_000) == 25.0


def test_unknown_model_is_free_not_an_error():
    assert cost_usd("made-up-model", 1000, 1000) == 0.0


def test_usage_as_dict_roundtrip():
    usage = Usage("claude-haiku-4-5", 1_000_000, 0, 12.34)
    assert usage.cost_usd == 1.0
    d = usage.as_dict()
    assert d["model"] == "claude-haiku-4-5"
    assert d["latency_ms"] == 12.3
    assert d["cost_usd"] == 1.0
