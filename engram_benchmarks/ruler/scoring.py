"""RULER scoring — programmatic string-match scorers.

No LLM judge, no OPENAI_API_KEY required.

Implemented from the official RULER constants.py
(https://github.com/hsiehjackson/RULER):

- string_match_all : fraction of reference strings found in the prediction.
- string_match_part: 1.0 if *any* reference is a substring, else 0.0.

Task routing
------------
qa_1 / qa_2 → StringMatchPartScorer  (partial credit; any answer match passes)
all others  → StringMatchAllScorer   (strict; every reference must appear)
"""

from __future__ import annotations

from typing import Union

from engram_benchmarks.shared.scoring import BaseScorer


class StringMatchAllScorer(BaseScorer):
    """string_match_all: fraction of reference strings found as substrings.

    If ``reference`` is a list, the score is the fraction of references found
    (case-insensitive) anywhere in the prediction string.  If all are found,
    score == 1.0.  If none are found, score == 0.0.

    A single-string reference is treated as a one-element list.
    """

    def score(self, prediction: str, reference: Union[list[str], str]) -> float:
        if isinstance(reference, str):
            reference = [reference]
        if not reference:
            return 0.0
        pred_lower = prediction.lower()
        found = sum(1 for ref in reference if ref.lower() in pred_lower)
        return found / len(reference)


class StringMatchPartScorer(BaseScorer):
    """string_match_part: 1.0 if any reference is a substring, else 0.0.

    Used for qa_1 / qa_2 where any acceptable answer string passing is a hit.
    """

    def score(self, prediction: str, reference: Union[list[str], str]) -> float:
        if isinstance(reference, str):
            reference = [reference]
        if not reference:
            return 0.0
        pred_lower = prediction.lower()
        return 1.0 if any(ref.lower() in pred_lower for ref in reference) else 0.0


def get_scorer(task_name: str) -> BaseScorer:
    """Factory: return the correct RULER scorer for a given task name.

    Parameters
    ----------
    task_name:
        One of the 13 RULER task names.

    Returns
    -------
    BaseScorer
        ``StringMatchPartScorer`` for qa_1 / qa_2; ``StringMatchAllScorer``
        for all other tasks.
    """
    if task_name in ("qa_1", "qa_2"):
        return StringMatchPartScorer()
    return StringMatchAllScorer()
