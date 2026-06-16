"""LLM client factories used by benchmark runners."""

from __future__ import annotations

from collections.abc import Callable

from rflow.clients import AnthropicClient, LLMClient, LLMUsage, OpenAIClient


class SmartFakeClient(LLMClient):
    """Deterministic local client for harness smoke tests.

    It recognizes the synthetic S-NIAH prompt shape and emits a REPL block for
    rflow runs, or a direct answer for vanilla chat runs.
    """

    def __init__(self) -> None:
        self.last_usage = LLMUsage()

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        prompt = "\n\n".join(m.get("content", "") for m in messages)
        self.last_usage = LLMUsage(
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=32,
        )
        answer = self._extract_secret(prompt)
        wants_repl = "```repl" in prompt or "done(" in prompt or "Python REPL" in prompt
        if wants_repl and answer == "UNKNOWN" and 'INPUTS["haystack"]' in prompt:
            return '```repl\nprint(INPUTS["haystack"])\n```'
        if wants_repl:
            return f'```repl\ndone("{answer}")\n```'
        return answer

    @staticmethod
    def _extract_secret(prompt: str) -> str:
        marker = ""
        for token in prompt.replace("`", " ").split():
            if token.startswith("needle-marker-"):
                marker = token.strip(".,:;()[]{}")
                break
        if marker:
            lines = prompt.splitlines()
            for index, line in enumerate(lines):
                if line.strip() == f"marker: {marker}":
                    for candidate in lines[index + 1 : index + 5]:
                        if candidate.startswith("secret: "):
                            return candidate.split("secret: ", 1)[1].strip()
        return "UNKNOWN"


class ClientFactory:
    """Registry-backed factory for runner LLM clients."""

    def __init__(
        self,
        providers: dict[str, Callable[[str], LLMClient]] | None = None,
    ) -> None:
        self._providers = {
            "fake": lambda _model: SmartFakeClient(),
            "openai": lambda model: OpenAIClient(model=model),
            "anthropic": lambda model: AnthropicClient(model=model),
        }
        if providers:
            self._providers.update(providers)

    def create(self, provider: str, model: str) -> LLMClient:
        try:
            factory = self._providers[provider]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers))
            raise ValueError(f"provider must be one of: {available}") from exc
        return factory(model)


def create_client(provider: str, model: str) -> LLMClient:
    """Build an LLM client from CLI provider/model arguments."""

    return ClientFactory().create(provider, model)


__all__ = ["ClientFactory", "SmartFakeClient", "create_client"]
