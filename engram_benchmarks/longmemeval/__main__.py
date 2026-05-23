"""CLI entry point for the LongMemEval benchmark harness.

Usage
-----
Dry-run (no GPU, no API key required)::

    python -m engram_benchmarks.longmemeval --dry-run

Live GPU run::

    OPENAI_API_KEY=<key> python -m engram_benchmarks.longmemeval \\
        --model-url http://localhost:30000 \\
        --snapshot-dir /tmp/lme_snapshots \\
        --output results/longmemeval.jsonl \\
        --num-questions 500

CLI fix — judge branch
----------------------
Dry-run  → MockJudge (no API calls, no OPENAI_API_KEY required)
Live run → LLMJudge  (reads JUDGE_MODEL, JUDGE_BASE_URL, OPENAI_API_KEY from env)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from engram_benchmarks.shared.http_client import make_dry_run_mock

from .judge import LLMJudge, MockJudge
from .results import LMERunSummary
from .runner import LMERunner, generate_dry_run_questions, load_questions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongMemEval benchmark harness for Engram stateful inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic questions and a mock HTTP client (no GPU, no API key needed).",
    )
    parser.add_argument(
        "--model-url",
        default="http://localhost:30000",
        help="Base URL of the Engram inference server.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="/tmp/lme_snapshots",
        help="Directory for snapshot stubs.",
    )
    parser.add_argument(
        "--output",
        default="results/longmemeval.jsonl",
        help="Path to write the JSONL results.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Limit the number of questions (default: all).",
    )
    parser.add_argument(
        "--mode",
        choices=["warm", "baseline"],
        default="warm",
        help="'warm' runs the full three-pass protocol; 'baseline' runs baseline only.",
    )
    parser.add_argument(
        "--model",
        default="default",
        help="Model name sent in the request payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------ #
    # Judge selection — the key CLI fix                                   #
    # dry_run=True  → MockJudge (no API calls)                           #
    # dry_run=False → LLMJudge  (reads env vars: JUDGE_MODEL, etc.)      #
    # ------------------------------------------------------------------ #
    if args.dry_run:
        judge = MockJudge()
        logger.info("Dry-run mode: using MockJudge (no OPENAI_API_KEY required)")
    else:
        judge = LLMJudge()  # raises RuntimeError if OPENAI_API_KEY is not set
        logger.info("Live mode: using LLMJudge")

    mock_fn = make_dry_run_mock() if args.dry_run else None

    runner = LMERunner(
        model_url=args.model_url,
        snapshot_dir=Path(args.snapshot_dir),
        judge=judge,
        model=args.model,
        mock_fn=mock_fn,
    )

    # ------------------------------------------------------------------ #
    # Load questions                                                      #
    # ------------------------------------------------------------------ #
    if args.dry_run:
        n = args.num_questions or 5
        questions = generate_dry_run_questions(n)
        logger.info("Generated %d dry-run questions", len(questions))
    else:
        if not args.num_questions:
            logger.warning("--num-questions not set; loading all 500 questions")
        questions = load_questions(Path("data/longmemeval/test.jsonl"), limit=args.num_questions)
        logger.info("Loaded %d questions from dataset", len(questions))

    # ------------------------------------------------------------------ #
    # Run                                                                 #
    # ------------------------------------------------------------------ #
    if args.mode == "warm":
        results = runner.run_all(questions)
    else:
        results = runner.run_baseline_only(questions)

    # ------------------------------------------------------------------ #
    # Summarise and write output                                          #
    # ------------------------------------------------------------------ #
    summary = LMERunSummary(
        benchmark="longmemeval",
        model=args.model,
        results=results,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_jsonl(out_path)

    d = summary.to_dict()
    print(json.dumps(d, indent=2, default=str))
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    sys.exit(main())
