"""Token F1 scorer for LoCoMo benchmark.

Token F1 is the standard QA metric (also used by SQuAD-style benchmarks):
- Normalize both prediction and reference (lowercase, strip punctuation,
  remove articles, collapse whitespace).
- Compute token-level precision and recall via word overlap.
- Return the harmonic mean (F1).

When multiple reference answers are provided, return the maximum F1 over all
references (matching the original LoCoMo evaluation protocol).

No LLM judge is needed — this metric is fully deterministic and requires no
OPENAI_API_KEY.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Union

from engram_benchmarks.shared.scoring import BaseScorer


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    return " ".join(s.split())


class TokenF1Scorer(BaseScorer):
    """Token-level F1 between prediction and reference (standard QA metric).

    Matches the LoCoMo paper evaluation: when reference is a list, max F1 over
    all references is returned.
    """

    def score(self, prediction: str, reference: Union[str, list]) -> float:
        """Return token F1 in [0.0, 1.0].

        Parameters
        ----------
        prediction:
            Model output string.
        reference:
            Ground-truth string, or a list of acceptable ground-truth strings.
            When a list is provided, returns the maximum F1 over all references.
        """
        if isinstance(reference, list):
            if not reference:
                return 0.0
            return max(self._f1_single(prediction, ref) for ref in reference)
        return self._f1_single(prediction, reference)

    def _f1_single(self, pred: str, ref: str) -> float:
        """Token F1 between a single prediction/reference pair."""
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(ref_tokens)
        n_common = sum(common.values())

        if n_common == 0:
            return 0.0

        precision = n_common / len(pred_tokens)
        recall = n_common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)
