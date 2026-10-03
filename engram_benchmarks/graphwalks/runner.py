"""GraphWalks two-phase runner.

Implements the warm-tier protocol defined in ``BaseTwoPhaseRunner`` for the
openai/graphwalks benchmark (BFS reachability / parent-finding tasks).

Question format
---------------
Each ``Question`` carries:
- ``graph_text`` — the adjacency-list representation of the graph (the long context).
- ``question``   — the BFS/parents query.
- ``answer``     — ground-truth node list.
- ``num_hops``   — BFS depth used to generate the answer.
- ``num_nodes``  — size of the graph.

Prompt contract
---------------
The model MUST end its response with::

    Final Answer: [node1, node2, ...]

Both the full and warm prompts include this instruction.

Synthetic questions
-------------------
``generate_synthetic_question`` creates a deterministic path graph (A→B→C→D→E)
and a BFS-depth-2 reachability question for dry-run / unit tests.  The answer
is always deterministic and the data dict is compatible with ``Question``.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from engram_benchmarks.shared.http_client import MockHttpFn
from engram_benchmarks.shared.runner import BaseTwoPhaseRunner

from .scoring import SetF1Scorer
from .results import GraphWalksResult

_FINAL_ANSWER_INSTRUCTION = (
    "End your response with exactly: Final Answer: [node1, node2, ...]"
)


@dataclass
class Question:
    """A single GraphWalks question/task."""

    question_id: str
    graph_text: str
    question: str
    answer: List[str]
    num_hops: int
    num_nodes: int


class GraphWalksRunner(BaseTwoPhaseRunner[Question]):
    """Two-phase runner for the GraphWalks benchmark."""

    _result_cls = GraphWalksResult

    def __init__(
        self,
        model_url: str,
        snapshot_dir: Path,
        model: str = "default",
        max_tokens: int = 512,
        mock_fn: Optional[MockHttpFn] = None,
    ) -> None:
        super().__init__(
            model_url=model_url,
            snapshot_dir=snapshot_dir,
            scorer=SetF1Scorer(),
            model=model,
            max_tokens=max_tokens,
            mock_fn=mock_fn,
        )

    # ------------------------------------------------------------------ #
    # BaseTwoPhaseRunner interface                                         #
    # ------------------------------------------------------------------ #

    def _item_id(self, item: Question) -> str:
        return item.question_id

    def _build_full_prompt(self, item: Question) -> str:
        """Full prompt: graph context + question + format instruction."""
        return (
            f"{item.graph_text}\n\n"
            f"{item.question}\n\n"
            f"{_FINAL_ANSWER_INSTRUCTION}"
        )

    def _build_warm_prompt(self, item: Question) -> str:
        """Warm prompt: question only (graph context is in the snapshot)."""
        return f"{item.question}\n\n{_FINAL_ANSWER_INSTRUCTION}"

    def _reference_answer(self, item: Question) -> list[str]:
        return item.answer

    def _snapshot_metadata(self, item: Question, full_prompt: str) -> dict:
        return {
            "question_id": item.question_id,
            "num_nodes": item.num_nodes,
            "num_hops": item.num_hops,
            "prompt_length": len(full_prompt),
        }

    def _extra_result_fields(self, item: Question) -> dict:
        return {
            "graph_size": item.num_nodes,
            "num_hops": item.num_hops,
        }


# ------------------------------------------------------------------ #
# Synthetic question generator for dry-run / tests                   #
# ------------------------------------------------------------------ #

def generate_synthetic_question(
    num_nodes: int = 5,
    num_hops: int = 2,
    seed: int = 42,
) -> dict:
    """Build a deterministic path-graph question for dry-run / unit tests.

    Graph structure: a simple directed path  A → B → C → D → E  (up to
    ``num_nodes`` nodes, labelled with uppercase letters starting at A).

    BFS question: "Starting from node A, which nodes are reachable at exactly
    depth ``num_hops``?"

    The answer is deterministic from graph structure: node at index ``num_hops``
    in the path (e.g. depth 2 from A on A→B→C gives C).

    Parameters
    ----------
    num_nodes:
        Number of nodes in the path graph (2 ≤ num_nodes ≤ 26).
    num_hops:
        BFS depth for the reachability question.
    seed:
        Unused (reserved for future random-graph variants); kept for API
        consistency.

    Returns
    -------
    dict
        Keys: ``question_id``, ``graph_text``, ``question``, ``answer``
        (list[str]), ``num_hops``, ``num_nodes``.
    """
    num_nodes = max(2, min(num_nodes, 26))
    num_hops = max(1, min(num_hops, num_nodes - 1))

    labels = list(string.ascii_uppercase[:num_nodes])

    # Build adjacency list text
    edges = [(labels[i], labels[i + 1]) for i in range(num_nodes - 1)]
    graph_lines = [f"Graph with {num_nodes} nodes and directed edges:"]
    for src, dst in edges:
        graph_lines.append(f"  {src} -> {dst}")
    graph_text = "\n".join(graph_lines)

    question = (
        f"Starting from node {labels[0]}, which nodes are reachable at "
        f"exactly depth {num_hops}?"
    )

    # In a simple path A→B→C→D→E, the only node at depth k from A is labels[k]
    answer = [labels[num_hops]] if num_hops < num_nodes else []

    question_id = f"synthetic_n{num_nodes}_h{num_hops}_s{seed}"

    return {
        "question_id": question_id,
        "graph_text": graph_text,
        "question": question,
        "answer": answer,
        "num_hops": num_hops,
        "num_nodes": num_nodes,
    }
