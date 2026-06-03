from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from rlmflow.llm import TinkerClient


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeFuture:
    def __init__(self, output):
        self.output = output
        self.timeout = None

    def result(self, timeout=None):
        self.timeout = timeout
        return self.output


class _FakeSamplingClient:
    def __init__(self):
        self.future = _FakeFuture(
            SimpleNamespace(sequences=[SimpleNamespace(tokens=[101, 102, 103])])
        )
        self.calls = []

    def sample(self, *, prompt, num_samples, sampling_params):
        self.calls.append(
            {
                "prompt": prompt,
                "num_samples": num_samples,
                "sampling_params": sampling_params,
            }
        )
        return self.future


class _FakeRenderer:
    def build_generation_prompt(self, messages):
        assert messages[-1]["content"] == "hello"
        return SimpleNamespace(tokens=[1, 2, 3, 4])

    def get_stop_sequences(self):
        return ["<|end|>"]

    def parse_response(self, tokens):
        assert tokens == [101, 102, 103]
        return {"role": "assistant", "content": "hi from tinker"}, True


def test_tinker_client_samples_and_tracks_usage(monkeypatch):
    fake_tinker = ModuleType("tinker")
    fake_tinker.types = SimpleNamespace(SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    sampling = _FakeSamplingClient()
    client = TinkerClient(
        sampling_client=sampling,
        renderer_obj=_FakeRenderer(),
        max_tokens=32,
        temperature=0.2,
    )

    text, usage = client.completion(
        [{"role": "user", "content": "hello"}],
        timeout=5,
    )

    assert text == "hi from tinker"
    assert usage.input_tokens == 4
    assert usage.output_tokens == 3
    assert sampling.future.timeout == 5
    assert sampling.calls[0]["num_samples"] == 1
    params = sampling.calls[0]["sampling_params"]
    assert params.kwargs == {
        "max_tokens": 32,
        "temperature": 0.2,
        "stop": ["<|end|>"],
    }
