"""LongMemEval judge — LLM-as-judge scorer per paper §3.3.

Environment variables
---------------------
JUDGE_MODEL
    OpenAI model name to use for judging.
    Default: ``gpt-4o-2024-08-06`` (official model from the paper).

JUDGE_BASE_URL
    Base URL for an OpenAI-compatible endpoint.
    Default: ``None`` → uses the official OpenAI endpoint.

OPENAI_API_KEY
    API key for the judge endpoint.
    Required when using ``LLMJudge``; raises ``RuntimeError`` at construction
    if not set.

Scoring contract
----------------
The judge sends the question, reference answer, and model answer to the
judge model and checks whether the response contains "yes" (case-insensitive).
Returns 1.0 for a match, 0.0 otherwise.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


_JUDGE_PROMPT_TEMPLATE = """\
You are a judge evaluating whether a model's answer correctly answers a question.

Question: {question}
Reference answer: {reference}
Model's answer: {answer}

Does the model's answer correctly answer the question given the reference? \
Reply with only "Yes" or "No".
"""


class BaseJudge(ABC):
    """Abstract base for LongMemEval judges."""

    @abstractmethod
    def score(self, question: str, reference: str, answer: str) -> float:
        """Score a model answer against the reference.

        Parameters
        ----------
        question:
            The original memory question.
        reference:
            The ground-truth reference answer.
        answer:
            The model's predicted answer.

        Returns
        -------
        float
            1.0 if the answer is judged correct, 0.0 otherwise.
        """
        ...


class LLMJudge(BaseJudge):
    """LLM-as-judge scorer following LongMemEval paper §3.3.

    Reads configuration from environment variables at construction time:

    - ``JUDGE_MODEL`` (default: ``gpt-4o-2024-08-06``)
    - ``JUDGE_BASE_URL`` (default: ``None`` → official OpenAI endpoint)
    - ``OPENAI_API_KEY`` (required; raises ``RuntimeError`` if absent)
    """

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it to your OpenAI API key before using LLMJudge. "
                "For dry-run testing without a key, use MockJudge instead."
            )
        self._model = os.environ.get("JUDGE_MODEL", "gpt-4o-2024-08-06")
        base_url = os.environ.get("JUDGE_BASE_URL") or None

        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required for LLMJudge. "
                "Install it with: pip install openai"
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def score(self, question: str, reference: str, answer: str) -> float:
        """Send the judge prompt and return 1.0 if the response contains 'yes'."""
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            reference=reference,
            answer=answer,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content or ""
        return 1.0 if "yes" in verdict.lower() else 0.0


class MockJudge(BaseJudge):
    """Dry-run judge for CPU tests — no API calls.

    Returns 1.0 for any non-empty answer, 0.0 for empty answers.
    """

    def score(self, question: str, reference: str, answer: str) -> float:  # noqa: ARG002
        return 1.0 if answer.strip() else 0.0
