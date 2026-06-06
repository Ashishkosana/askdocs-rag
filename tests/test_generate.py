from askdocs.generate import build_user_prompt, generate_answer
from askdocs.store import Hit


def test_prompt_numbers_each_context():
    hits = [Hit("Alpha", "x.md", 0, 0.5), Hit("Beta", "y.md", 1, 0.4)]
    prompt = build_user_prompt("What is alpha?", hits)
    assert "[1]" in prompt and "[2]" in prompt
    assert "x.md" in prompt and "Alpha" in prompt
    assert "What is alpha?" in prompt


def test_prompt_handles_empty_context():
    assert "(no context retrieved)" in build_user_prompt("Q?", [])


def test_generate_answer_returns_text_and_usage(fake_client_factory):
    client = fake_client_factory("Grounded answer [1]")
    ans = generate_answer(
        client,
        "Q?",
        [Hit("ctx", "x.md", 0, 0.5)],
        model="claude-opus-4-8",
        effort="low",
        max_tokens=256,
    )
    assert ans.text == "Grounded answer [1]"
    assert ans.usage.output_tokens == 20
    assert ans.usage.cost_usd > 0
