"""GraphWalks scoring — set-overlap F1.

Implements the official scoring logic from the openai/graphwalks HF dataset card.

Parse contract
--------------
The model must end its response with exactly::

    Final Answer: [node1, node2, ...]

``parse_answer`` extracts the node list from the **last line** of the response.
Any response that does not contain that exact footer on the last line returns an
empty list, which scores 0.0.

F1 contract
-----------
- precision = |predicted ∩ truth| / |predicted|
- recall    = |predicted ∩ truth| / |truth|
- f1        = 2 * P * R / (P + R)
- Both sets empty → 1.0 (trivially correct).
"""

from __future__ import annotations

import re
from typing import Union

from engram_benchmarks.shared.scoring import BaseScorer


def parse_answer(response: str) -> list[str]:
    """Extract node list from 'Final Answer: [node1, node2, ...]' on the last line.

    Parameters
    ----------
    response:
        Full model response text.

    Returns
    -------
    list[str]
        Parsed node names; empty list when the footer is missing or malformed.
    """
    line = response.split("\n")[-1]
    if "Final Answer:" not in line:
        return []
    match = re.search(r"Final Answer: ?\[.*\]", line)
    if not match:
        return []
    inner = match.group(0).removeprefix("Final Answer:").strip().strip("[]")
    return [x.strip() for x in inner.split(",") if x.strip()]


class SetF1Scorer(BaseScorer):
    """Set-overlap F1 between predicted node set and ground-truth node set.

    Scoring formula (from openai/graphwalks dataset card):
        precision = |predicted ∩ truth| / |predicted|
        recall    = |predicted ∩ truth| / |truth|
        f1        = 2 * precision * recall / (precision + recall)

    Edge cases:
        - Both sets empty → 1.0 (perfect trivial match)
        - Only predicted is empty → 0.0
        - Only truth is empty → 0.0 (unexpected; model predicted when it shouldn't)
    """

    def score(self, prediction: str, reference: Union[str, list]) -> float:
        """Compute set-F1 between parsed prediction and ground-truth node list.

        Parameters
        ----------
        prediction:
            Raw model response text; ``parse_answer`` is applied internally.
        reference:
            Ground-truth node list (list[str]) or a single node (str).

        Returns
        -------
        float
            F1 score in [0.0, 1.0].
        """
        pred_nodes = set(parse_answer(prediction))
        if isinstance(reference, str):
            truth_nodes = {reference}
        else:
            truth_nodes = set(reference)

        # Both empty → trivially correct
        if not pred_nodes and not truth_nodes:
            return 1.0

        # One side empty → score 0
        if not pred_nodes or not truth_nodes:
            return 0.0

        intersection = pred_nodes & truth_nodes
        precision = len(intersection) / len(pred_nodes)
        recall = len(intersection) / len(truth_nodes)

        if precision + recall == 0.0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)
