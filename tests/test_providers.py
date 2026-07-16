from types import SimpleNamespace

from token_efficient_nlp.providers import GroqProvider


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"label":"incendio","confidence":0.93}'
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=31, completion_tokens=9),
        )


def test_groq_provider_returns_validated_result_and_usage():
    completions = FakeCompletions()
    provider = GroqProvider.__new__(GroqProvider)
    provider.model = "llama-3.1-8b-instant"
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    result = provider.classify(
        "fumaça no painel",
        ["queda", "incendio"],
    )

    assert result.label == "incendio"
    assert result.confidence == 0.93
    assert result.input_tokens == 31
    assert result.output_tokens == 9
    assert result.provider == "groq"
    assert completions.kwargs["model"] == "llama-3.1-8b-instant"
    assert completions.kwargs["max_completion_tokens"] == 60
    assert completions.kwargs["response_format"] == {"type": "json_object"}
