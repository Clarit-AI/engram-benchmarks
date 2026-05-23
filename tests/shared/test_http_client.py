"""Tests for the streaming HTTP client layer."""

from __future__ import annotations

import pytest
from engram_benchmarks.shared.http_client import (
    ChatResult,
    chat_completion,
    make_dry_run_mock,
)


def test_dry_run_mock_returns_chat_result():
    mock = make_dry_run_mock(answer="hello world", ttft_s=0.01, total_latency_s=0.05)
    result = chat_completion(
        model_url="http://unused",
        messages=[{"role": "user", "content": "test"}],
        mock_fn=mock,
    )
    assert isinstance(result, ChatResult)
    assert result.text == "hello world"
    assert result.ttft_s == pytest.approx(0.01)
    assert result.total_latency_s == pytest.approx(0.05)


def test_ttft_strictly_less_than_total_latency_in_mock():
    """Mock enforces that TTFT < total latency (first token before full response)."""
    mock = make_dry_run_mock(ttft_s=0.042, total_latency_s=0.150)
    result = chat_completion("http://unused", [{"role": "user", "content": "q"}], mock_fn=mock)
    assert result.ttft_s < result.total_latency_s, (
        "TTFT must be < total latency — streaming first-token fires before response completes"
    )


def test_input_tokens_counted_from_prompt():
    mock = make_dry_run_mock(answer="yes")
    result = chat_completion(
        "http://unused",
        [{"role": "user", "content": "the quick brown fox"}],
        mock_fn=mock,
    )
    assert result.input_tokens == 4  # "the quick brown fox" = 4 words


def test_output_tokens_from_answer():
    mock = make_dry_run_mock(answer="A B C D E")
    result = chat_completion("http://unused", [{"role": "user", "content": "q"}], mock_fn=mock)
    assert result.output_tokens == 5
