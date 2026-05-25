"""Streaming HTTP client for Engram benchmark harnesses.

TTFT contract
-------------
TTFT is the time from request start to receipt of the first non-empty delta
token in the SSE stream.  This is the DEFAULT measurement path — streaming is
always used.  Measuring wall-clock around the whole response and calling it
TTFT is a correctness bug; this module does not support that.

For dry-run / unit-test contexts, inject a ``mock_http_fn`` that returns a
``ChatResult`` with synthetic values.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ChatResult:
    """Return value of a single chat completion call."""

    text: str
    ttft_s: float          # time-to-first-token (streaming, seconds)
    total_latency_s: float # wall-clock for the full response
    input_tokens: int      # caller-provided count (token proxy or tokenizer)
    output_tokens: int     # word-split proxy of the response text


# Signature for injectable mock clients used in tests.
MockHttpFn = Callable[[str, list, str], ChatResult]


def chat_completion(
    model_url: str,
    messages: list[dict],
    model: str = "default",
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 120.0,
    input_token_count: int = 0,
    rid: Optional[str] = None,
    mock_fn: Optional[MockHttpFn] = None,
) -> ChatResult:
    """Send a streaming chat completion and return a ChatResult.

    Parameters
    ----------
    model_url:
        Base URL of the OpenAI-compatible server (e.g. ``http://localhost:30000``).
    messages:
        OpenAI-format message list.
    model:
        Model name sent in the request payload.
    max_tokens:
        Maximum tokens to generate.
    temperature:
        Sampling temperature.
    timeout:
        HTTP request timeout in seconds.
    input_token_count:
        Pre-computed input token count (from caller; avoids re-tokenising).
    rid:
        Optional request ID to embed in the payload.  When provided, the
        Engram server tracks the request under this key so a subsequent
        ``/save_snapshot`` call with ``rid=<same>`` can locate the state.
        Omit for baseline and warm passes; always set for the cold pass.
    mock_fn:
        If provided, called instead of making an HTTP request.  Signature:
        ``(model_url, messages, model) -> ChatResult``.  Used by tests.
    """
    if mock_fn is not None:
        return mock_fn(model_url, messages, model)

    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("'requests' is required for live HTTP calls.") from exc

    url = model_url.rstrip("/") + "/v1/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,  # always stream; TTFT requires first-token timing
    }
    if rid is not None:
        payload["rid"] = rid
    headers = {"Content-Type": "application/json"}

    t_start = time.perf_counter()
    t_first_token: Optional[float] = None
    chunks: list[str] = []

    with requests.post(
        url, json=payload, headers=headers, stream=True, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line: str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta: str = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    chunks.append(delta)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    t_end = time.perf_counter()
    # If no delta token arrived (empty response), TTFT = total latency.
    ttft = (t_first_token if t_first_token is not None else t_end) - t_start
    total = t_end - t_start
    text = "".join(chunks)

    return ChatResult(
        text=text,
        ttft_s=ttft,
        total_latency_s=total,
        input_tokens=input_token_count,
        output_tokens=_word_count(text),
    )


def _word_count(text: str) -> int:
    return len(text.split())


def make_dry_run_mock(
    answer: str = "Dry-run answer.",
    ttft_s: float = 0.042,
    total_latency_s: float = 0.150,
) -> MockHttpFn:
    """Return a mock HTTP function for CPU dry-run / unit tests."""

    def _mock(model_url: str, messages: list, model: str) -> ChatResult:
        prompt_text = " ".join(
            m.get("content", "") for m in messages if isinstance(m, dict)
        )
        return ChatResult(
            text=answer,
            ttft_s=ttft_s,
            total_latency_s=total_latency_s,
            input_tokens=_word_count(prompt_text),
            output_tokens=_word_count(answer),
        )

    return _mock
