from __future__ import annotations

from rflow.llm import is_retryable


class APIConnectionError(Exception):
    pass


class RateLimitError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class ReadTimeout(Exception):
    pass


def test_retry_transient_keeps_connection_and_rate_errors_retryable():
    assert is_retryable(APIConnectionError("connection dropped"))
    assert is_retryable(RateLimitError("rate limited"))


def test_retry_transient_does_not_retry_timeout_errors():
    assert not is_retryable(APITimeoutError("request timed out"))
    assert not is_retryable(ReadTimeout("read timed out"))
    assert not is_retryable(TimeoutError("outer timeout"))


def test_retry_transient_does_not_retry_timeout_causes():
    exc = RuntimeError("wrapped")
    exc.__cause__ = ReadTimeout("read timed out")

    assert not is_retryable(exc)
