"""GraphWalks dataset download and dry-run data generation.

Dataset
-------
openai/graphwalks on Hugging Face Hub (MIT license).
Tasks: BFS reachability and parent-finding, up to 128K token contexts.

Usage
-----
For a real run (requires HF credentials)::

    from engram_benchmarks.graphwalks.download import download_dataset
    questions = download_dataset(split="test")

For dry-run / CPU tests (no HF token needed)::

    from engram_benchmarks.graphwalks.download import generate_dry_run_data
    generate_dry_run_data("/tmp/graphwalks-dry", n=5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .runner import Question, generate_synthetic_question


# ------------------------------------------------------------------ #
# Dry-run data generation (no HF token required)                     #
# ------------------------------------------------------------------ #

def generate_dry_run_data(data_dir: "str | Path", n: int = 5) -> List[Question]:
    """Generate ``n`` synthetic questions and save them to ``data_dir``.

    Files are saved as ``<data_dir>/synthetic_<i>.json``.  Each file contains
    the dict returned by ``generate_synthetic_question``.

    Parameters
    ----------
    data_dir:
        Directory to save the synthetic question files.
    n:
        Number of synthetic questions to generate.

    Returns
    -------
    List[Question]
        The generated ``Question`` objects (also saved to disk).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    questions: List[Question] = []
    for i in range(n):
        num_nodes = 5 + i            # 5, 6, 7, 8, 9 nodes
        num_hops = min(2, num_nodes - 1)
        raw = generate_synthetic_question(num_nodes=num_nodes, num_hops=num_hops, seed=i)
        out_path = data_dir / f"synthetic_{i}.json"
        out_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        questions.append(
            Question(
                question_id=raw["question_id"],
                graph_text=raw["graph_text"],
                question=raw["question"],
                answer=raw["answer"],
                num_hops=raw["num_hops"],
                num_nodes=raw["num_nodes"],
            )
        )

    return questions


def load_dry_run_data(data_dir: "str | Path") -> List[Question]:
    """Load synthetic questions previously saved by ``generate_dry_run_data``."""
    data_dir = Path(data_dir)
    questions: List[Question] = []
    for p in sorted(data_dir.glob("synthetic_*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        questions.append(
            Question(
                question_id=raw["question_id"],
                graph_text=raw["graph_text"],
                question=raw["question"],
                answer=raw["answer"],
                num_hops=raw["num_hops"],
                num_nodes=raw["num_nodes"],
            )
        )
    return questions


# ------------------------------------------------------------------ #
# Real dataset download (requires HF token / datasets library)       #
# ------------------------------------------------------------------ #

def download_dataset(
    split: str = "test",
    cache_dir: "str | Path | None" = None,
) -> List[Question]:
    """Download the openai/graphwalks dataset from Hugging Face and return Questions.

    Requires ``datasets`` library (``pip install datasets``) and a Hugging Face
    token with access to the openai/graphwalks repo.

    Parameters
    ----------
    split:
        Dataset split to load (e.g. ``"test"``, ``"validation"``).
    cache_dir:
        Optional local cache directory for HF datasets.

    Returns
    -------
    List[Question]
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required for dataset download. "
            "Install it with: pip install datasets"
        ) from exc

    ds = load_dataset(
        "openai/graphwalks",
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=False,
    )

    questions: List[Question] = []
    for i, row in enumerate(ds):
        # Field names may vary; handle common variants gracefully.
        graph_text = row.get("graph", row.get("graph_text", ""))
        question = row.get("question", row.get("query", ""))
        raw_answer = row.get("answer", row.get("answer_nodes", []))
        if isinstance(raw_answer, str):
            raw_answer = [raw_answer]
        num_hops = int(row.get("num_hops", row.get("hops", 0)))
        num_nodes = int(row.get("num_nodes", row.get("graph_size", 0)))
        question_id = str(row.get("id", row.get("question_id", f"{split}_{i}")))

        questions.append(
            Question(
                question_id=question_id,
                graph_text=graph_text,
                question=question,
                answer=raw_answer,
                num_hops=num_hops,
                num_nodes=num_nodes,
            )
        )

    return questions
