"""RULER benchmark runner.

Subclasses ``BaseTwoPhaseRunner[TaskInstance]`` using the deliberate warm
protocol:

  1. Baseline — full context + question → stateless score + TTFT.
  2. Cold Engram pass — full context, saves snapshot stub.
  3. Warm Engram pass — question only, snapshot present → warm score + TTFT.

``_result_cls = RULERResult`` routes ``BaseTwoPhaseRunner.run_all`` to produce
``RULERResult`` instances without a subclass override.  ``_scorer_for`` swaps
the match strategy per task_name (string_match_all vs string_match_part).

Usage (dry-run)::

    python -m engram_benchmarks.ruler.runner \\
        --url http://localhost:30000 \\
        --model default \\
        --tasks niah_single_1 niah_multikey_2 \\
        --context-lengths 4096 8192 \\
        --dry-run \\
        --output results/ruler_dry.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional

from engram_benchmarks.shared.http_client import MockHttpFn, make_dry_run_mock
from engram_benchmarks.shared.runner import BaseTwoPhaseRunner
from engram_benchmarks.shared.scoring import BaseScorer

from .results import RULERResult, RULERRunSummary
from .scoring import StringMatchAllScorer, get_scorer
from .tasks import (
    CONTEXT_LENGTHS,
    RULER_TASKS,
    TaskInstance,
    generate_synthetic_task,
)

logger = logging.getLogger(__name__)


class RULERRunner(BaseTwoPhaseRunner[TaskInstance]):
    """Two-phase RULER runner.

    Parameters
    ----------
    model_url:
        Base URL of the OpenAI-compatible endpoint.
    snapshot_dir:
        Directory for snapshot stubs.
    model:
        Model name sent in requests.
    max_tokens:
        Max tokens per generation.
    mock_fn:
        Inject a mock HTTP function for dry-run / unit tests.
    """

    _result_cls = RULERResult

    # ------------------------------------------------------------------ #
    # BaseTwoPhaseRunner abstract interface                                #
    # ------------------------------------------------------------------ #

    def _item_id(self, item: TaskInstance) -> str:
        return item.task_id

    def _build_full_prompt(self, item: TaskInstance) -> str:
        return (
            f"Read the following passage carefully.\n\n"
            f"{item.context_text}\n\n"
            f"Question: {item.question}\n"
            f"Answer:"
        )

    def _build_warm_prompt(self, item: TaskInstance) -> str:
        """Warm prompt: question only, no long context (snapshot assumed present)."""
        return (
            f"Using the document you already processed, answer this question:\n\n"
            f"Question: {item.question}\n"
            f"Answer:"
        )

    def _reference_answer(self, item: TaskInstance) -> Any:
        return item.answer

    def _snapshot_metadata(self, item: TaskInstance, full_prompt: str) -> dict:
        return {
            "task_id": item.task_id,
            "task_name": item.task_name,
            "context_length": item.context_length,
            "prompt_length_chars": len(full_prompt),
        }

    def _extra_result_fields(self, item: TaskInstance) -> dict:
        return {
            "task_name": item.task_name,
            "context_length": item.context_length,
        }

    def _scorer_for(self, item: TaskInstance) -> BaseScorer:
        """Per-task scorer: RULER uses different match strategies per task_name."""
        return get_scorer(item.task_name)


# ------------------------------------------------------------------ #
# High-level helper                                                    #
# ------------------------------------------------------------------ #


def run_ruler(
    model_url: str,
    task_names: Optional[List[str]] = None,
    context_lengths: Optional[List[int]] = None,
    sample_indices: Optional[List[int]] = None,
    model: str = "default",
    snapshot_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    mock_fn: Optional[MockHttpFn] = None,
) -> RULERRunSummary:
    """Generate tasks, run all phases, return a ``RULERRunSummary``.

    Parameters
    ----------
    model_url:
        OpenAI-compatible endpoint URL.
    task_names:
        Subset of ``RULER_TASKS`` to run.  Defaults to all 13.
    context_lengths:
        Context lengths to test.  Defaults to ``CONTEXT_LENGTHS``.
    sample_indices:
        Sample indices per (task, context_length) cell.  Defaults to [0].
    model:
        Model name.
    snapshot_dir:
        Where to write snapshot stubs.  Defaults to ``/tmp/ruler_snapshots``.
    output_path:
        If provided, write a JSONL results file here.
    mock_fn:
        Mock HTTP function for dry-run / tests.

    Returns
    -------
    RULERRunSummary
    """
    task_names = task_names or RULER_TASKS
    context_lengths = context_lengths or CONTEXT_LENGTHS
    sample_indices = sample_indices or [0]
    snapshot_dir = snapshot_dir or Path("/tmp/ruler_snapshots")

    items: List[TaskInstance] = []
    for task_name in task_names:
        for ctx_len in context_lengths:
            for sidx in sample_indices:
                items.append(generate_synthetic_task(task_name, ctx_len, sidx))

    logger.info("Generated %d RULER task instances.", len(items))

    runner = RULERRunner(
        model_url=model_url,
        snapshot_dir=snapshot_dir,
        scorer=StringMatchAllScorer(),  # default; _scorer_for overrides per item
        model=model,
        mock_fn=mock_fn,
    )

    results = runner.run_all(items)

    summary = RULERRunSummary(
        benchmark="ruler",
        model=model,
        results=results,  # type: ignore[arg-type]
    )

    if output_path is not None:
        summary.to_jsonl(output_path)
        logger.info("Results written to %s", output_path)

    return summary


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m engram_benchmarks.ruler.runner",
        description="Run the RULER long-context benchmark harness.",
    )
    p.add_argument("--url", default="http://localhost:30000", help="Model server URL.")
    p.add_argument("--model", default="default", help="Model name.")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=RULER_TASKS,
        choices=RULER_TASKS,
        metavar="TASK",
        help="Task names to run (default: all 13).",
    )
    p.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[4096, 8192],
        metavar="N",
        help="Context lengths in tokens (default: 4096 8192).",
    )
    p.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=[0],
        metavar="IDX",
        help="Sample indices per (task, ctx_len) cell (default: 0).",
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("/tmp/ruler_snapshots"),
        help="Directory for snapshot stubs.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/ruler.jsonl"),
        help="Path for JSONL results output.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock HTTP client (no server required).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    mock_fn = make_dry_run_mock() if args.dry_run else None

    summary = run_ruler(
        model_url=args.url,
        task_names=args.tasks,
        context_lengths=args.context_lengths,
        sample_indices=args.samples,
        model=args.model,
        snapshot_dir=args.snapshot_dir,
        output_path=args.output,
        mock_fn=mock_fn,
    )

    d = summary.to_dict()
    print("\n=== RULER Results ===")
    print(f"Tasks run      : {d['n_total']} total ({d['n_warm']} warm, {d['n_cold']} cold)")
    print(f"Warm token red.: {d['warm_token_reduction']}")
    print(f"Warm TTFT spup.: {d['warm_ttft_speedup']}")
    amort = d.get("compute_amortization")
    if amort:
        print(f"Break-even     : {amort['break_even_restores']} warm restores")
    print(f"Output         : {args.output}")


if __name__ == "__main__":
    main()
