from __future__ import annotations

import threading
import time

import pytest

from rflow.llm import LLMClient, LLMUsage
from rflow.llm_channel import LLMChannel


class _ObservedLLM(LLMClient):
    def __init__(self, *, thread_safe: bool, delay: float = 0.01) -> None:
        self.thread_safe = thread_safe
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def chat(self, messages, *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
        prompt = messages[-1]["content"]
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            sleep_for = self.delay * 3 if prompt == "slow" else self.delay
            timeout = kwargs.get("timeout")
            if timeout is not None and sleep_for > timeout:
                time.sleep(timeout)
                raise TimeoutError(f"request timed out after {timeout}s")
            time.sleep(sleep_for)
        finally:
            with self.lock:
                self.active -= 1
        return prompt.upper(), LLMUsage(input_tokens=len(prompt), output_tokens=1)


class _OrderedUnsafeLLM(LLMClient):
    thread_safe = False

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, messages, *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        return prompt.upper(), LLMUsage(input_tokens=len(prompt), output_tokens=1)


class _KwargsLLM(LLMClient):
    thread_safe = True

    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def chat(self, messages, *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
        self.kwargs.append(dict(kwargs))
        return messages[-1]["content"], LLMUsage(input_tokens=1, output_tokens=1)


def test_llm_channel_preserves_batch_order_when_calls_finish_out_of_order():
    client = _ObservedLLM(thread_safe=True, delay=0.01)
    channel = LLMChannel(
        {"default": client},
        max_concurrency=2,
    )
    try:
        pairs = channel.batch("default", ["slow", "fast"])
    finally:
        channel.shutdown()

    assert [text for text, _usage in pairs] == ["SLOW", "FAST"]
    assert client.max_active == 2


def test_llm_channel_forwards_sampling_kwargs_to_batch_requests():
    client = _KwargsLLM()
    channel = LLMChannel(
        {"default": client},
        max_concurrency=2,
        request_timeout=5,
    )
    try:
        pairs = channel.batch(
            "default",
            ["a", "b"],
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
            stop=["DONE"],
        )
    finally:
        channel.shutdown()

    assert [text for text, _usage in pairs] == ["a", "b"]
    assert client.kwargs == [
        {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 128,
            "stop": ["DONE"],
            "timeout": 5,
        },
        {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 128,
            "stop": ["DONE"],
            "timeout": 5,
        },
    ]


def test_llm_channel_global_cap_holds_across_nested_callers():
    client = _ObservedLLM(thread_safe=True, delay=0.01)
    channel = LLMChannel(
        {"default": client},
        max_concurrency=2,
    )
    outputs: list[list[str]] = []
    outputs_lock = threading.Lock()

    def run_batch(index: int) -> None:
        pairs = channel.batch(
            "default",
            [f"{index}-a", f"{index}-b", f"{index}-c"],
        )
        with outputs_lock:
            outputs.append([text for text, _usage in pairs])

    threads = [threading.Thread(target=run_batch, args=(i,)) for i in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        channel.shutdown()

    assert len(outputs) == 4
    assert client.max_active <= 2


def test_llm_channel_serializes_unsafe_clients():
    client = _ObservedLLM(thread_safe=False, delay=0.01)
    channel = LLMChannel(
        {"default": client},
        max_concurrency=4,
    )
    try:
        pairs = channel.batch("default", ["a", "b", "c", "d"])
    finally:
        channel.shutdown()

    assert [text for text, _usage in pairs] == ["A", "B", "C", "D"]
    assert client.max_active == 1


def test_llm_channel_preserves_unsafe_client_call_order():
    client = _OrderedUnsafeLLM()
    channel = LLMChannel(
        {"default": client},
        max_concurrency=4,
    )
    try:
        pairs = channel.batch("default", ["a", "b", "c", "d"])
    finally:
        channel.shutdown()

    assert [text for text, _usage in pairs] == ["A", "B", "C", "D"]
    assert client.prompts == ["a", "b", "c", "d"]


class _UsageLLM(LLMClient):
    thread_safe = True

    def chat(self, messages, *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
        prompt = messages[-1]["content"]
        usage = LLMUsage(input_tokens=int(prompt), output_tokens=1)
        self.last_usage = LLMUsage(input_tokens=999, output_tokens=999)
        return prompt, usage


def test_llm_channel_uses_per_request_usage_not_shared_last_usage():
    channel = LLMChannel(
        {"default": _UsageLLM()},
        max_concurrency=3,
    )
    try:
        pairs = channel.batch("default", ["1", "2", "3"])
    finally:
        channel.shutdown()

    assert [text for text, _usage in pairs] == ["1", "2", "3"]
    assert sum(usage.input_tokens for _text, usage in pairs) == 6
    assert sum(usage.output_tokens for _text, usage in pairs) == 3


def test_llm_channel_times_out_stuck_batch_request():
    client = _ObservedLLM(thread_safe=True, delay=0.05)
    channel = LLMChannel(
        {"default": client},
        max_concurrency=1,
        request_timeout=0.01,
    )
    try:
        with pytest.raises(TimeoutError, match="LLM request timed out"):
            channel.batch("default", ["slow"])
    finally:
        channel.shutdown()
