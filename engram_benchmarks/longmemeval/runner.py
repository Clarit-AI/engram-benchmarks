"""LongMemEval two-phase runner.

Subclasses BaseTwoPhaseRunner for the LongMemEval benchmark.  Wires the
LLM-as-judge scorer into the three-pass warm-tier protocol.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from engram_benchmarks.shared.http_client import MockHttpFn, _word_count
from engram_benchmarks.shared.results import BaseResult
from engram_benchmarks.shared.runner import BaseTwoPhaseRunner
from engram_benchmarks.shared.scoring import BaseScorer

from .judge import BaseJudge, MockJudge
from .results import LMEResult, LMERunSummary

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """A single LongMemEval question."""

    question_id: str
    memory_text: str   # long memory context (115K–1.5M tokens)
    question: str
    answer: str        # ground-truth reference answer
    memory_type: str   # episodic | semantic | temporal | spatial | factual


class _JudgeScorer(BaseScorer):
    """Adapts a BaseJudge into the BaseScorer interface for a single question."""

    def __init__(self, judge: BaseJudge, question: str) -> None:
        self._judge = judge
        self._question = question

    def score(self, prediction: str, reference: str) -> float:  # type: ignore[override]
        return self._judge.score(self._question, reference, prediction)


class LMERunner(BaseTwoPhaseRunner[Question]):
    """Two-phase runner for LongMemEval.

    Uses an LLM-as-judge scorer (MockJudge in dry-run mode, LLMJudge in live
    mode) instead of the simple string-match BaseScorer.

    Parameters
    ----------
    model_url:
        Base URL of the Engram inference server.
    snapshot_dir:
        Directory where snapshot stubs are written.
    judge:
        Judge instance to use for scoring.  Pass MockJudge() for dry-run.
    model:
        Model name sent in the request payload.
    max_tokens:
        Maximum tokens to generate per call.
    mock_fn:
        Inject a mock HTTP function for dry-run / unit tests.
    """

    def __init__(
        self,
        model_url: str,
        snapshot_dir: Path,
        judge: Optional[BaseJudge] = None,
        model: str = "default",
        max_tokens: int = 256,
        mock_fn: Optional[MockHttpFn] = None,
    ) -> None:
        # Use a placeholder scorer; the real per-question scorer is set in run_all.
        self._judge = judge or MockJudge()
        super().__init__(
            model_url=model_url,
            snapshot_dir=snapshot_dir,
            scorer=_JudgeScorer(self._judge, ""),  # placeholder
            model=model,
            max_tokens=max_tokens,
            mock_fn=mock_fn,
        )

    # ------------------------------------------------------------------ #
    # BaseTwoPhaseRunner abstract method implementations                  #
    # ------------------------------------------------------------------ #

    def _item_id(self, item: Question) -> str:
        return item.question_id

    def _build_full_prompt(self, item: Question) -> str:
        return f"{item.memory_text}\n\nQuestion: {item.question}\nAnswer:"

    def _build_warm_prompt(self, item: Question) -> str:
        return f"Question: {item.question}\nAnswer:"

    def _reference_answer(self, item: Question) -> str:
        return item.answer

    def _snapshot_metadata(self, item: Question, full_prompt: str) -> dict:
        return {
            "question_id": item.question_id,
            "memory_type": item.memory_type,
            "token_proxy": _word_count(full_prompt),
        }

    def _extra_result_fields(self, item: Question) -> dict:
        return {"memory_type": item.memory_type}

    # ------------------------------------------------------------------ #
    # Override run_all to wire the judge per question                     #
    # ------------------------------------------------------------------ #

    def run_all(self, items: List[Question]) -> List[LMEResult]:  # type: ignore[override]
        """Run baseline → cold → warm for each question.

        Wires the judge with the per-question text before each scoring call,
        then constructs LMEResult (not BaseResult) so memory_type is preserved.
        """
        results: List[LMEResult] = []
        for item in items:
            item_id = self._item_id(item)
            full_prompt = self._build_full_prompt(item)
            warm_prompt = self._build_warm_prompt(item)
            ref = self._reference_answer(item)

            judge_scorer = _JudgeScorer(self._judge, item.question)

            # --- 1. Baseline ---
            b = self._call(full_prompt)
            b_score = judge_scorer.score(b.text, ref)

            # --- 2. Cold Engram pass (establishes snapshot; metrics not reported) ---
            self._call(full_prompt)  # discard cold latency
            self._write_snapshot(item, full_prompt)

            # --- 3. Warm Engram pass (snapshot now present) ---
            assert self._snap_exists(item), "Snapshot must exist after cold pass"
            w = self._call(warm_prompt)
            w_score = judge_scorer.score(w.text, ref)

            result = LMEResult(
                item_id=item_id,
                restore_mode="warm",
                baseline_ttft_s=b.ttft_s,
                baseline_input_tokens=b.input_tokens,
                baseline_output_tokens=b.output_tokens,
                baseline_answer=b.text,
                baseline_score=b_score,
                engram_ttft_s=w.ttft_s,
                engram_input_tokens=w.input_tokens,
                engram_output_tokens=w.output_tokens,
                engram_answer=w.text,
                engram_score=w_score,
                memory_type=item.memory_type,
            )
            results.append(result)
            logger.info(
                "%s [%s] | baseline_ttft=%.3fs warm_ttft=%.3fs token_reduction=%.2f",
                item_id,
                item.memory_type,
                b.ttft_s,
                w.ttft_s,
                result.token_reduction,
            )
        return results

    def run_baseline_only(self, items: List[Question]) -> List[LMEResult]:  # type: ignore[override]
        """Baseline-only run — no Engram path, all results labelled 'cold'."""
        results: List[LMEResult] = []
        for item in items:
            full_prompt = self._build_full_prompt(item)
            ref = self._reference_answer(item)
            judge_scorer = _JudgeScorer(self._judge, item.question)
            b = self._call(full_prompt)
            b_score = judge_scorer.score(b.text, ref)
            results.append(
                LMEResult(
                    item_id=self._item_id(item),
                    restore_mode="cold",
                    baseline_ttft_s=b.ttft_s,
                    baseline_input_tokens=b.input_tokens,
                    baseline_output_tokens=b.output_tokens,
                    baseline_answer=b.text,
                    baseline_score=b_score,
                    engram_ttft_s=0.0,
                    engram_input_tokens=0,
                    engram_output_tokens=0,
                    engram_answer="",
                    engram_score=0.0,
                    memory_type=item.memory_type,
                )
            )
        return results


# ------------------------------------------------------------------ #
# Dataset helpers                                                      #
# ------------------------------------------------------------------ #

_MEMORY_TYPES = ["episodic", "semantic", "temporal", "spatial", "factual"]


def load_questions(path: Path, limit: Optional[int] = None) -> List[Question]:
    """Load questions from a JSONL file.

    Each line must be a JSON object with keys:
    ``question_id``, ``memory_text``, ``question``, ``answer``, ``memory_type``.

    Parameters
    ----------
    path:
        Path to the JSONL file (e.g. ``data_dir/test.jsonl``).
    limit:
        If provided, return only the first ``limit`` questions.
    """
    questions: List[Question] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            questions.append(
                Question(
                    question_id=d["question_id"],
                    memory_text=d["memory_text"],
                    question=d["question"],
                    answer=d["answer"],
                    memory_type=d.get("memory_type", "unknown"),
                )
            )
            if limit is not None and len(questions) >= limit:
                break
    return questions


def generate_dry_run_questions(n: int = 5) -> List[Question]:
    """Generate ``n`` synthetic questions for CPU dry-run testing.

    No download required.  The memory texts are intentionally short so the
    dry-run runs without GPU or network access.
    """
    memory_types = _MEMORY_TYPES
    questions: List[Question] = []
    for i in range(n):
        mt = memory_types[i % len(memory_types)]
        questions.append(
            Question(
                question_id=f"dry_run_{i:03d}",
                memory_text=(
                    f"Alice moved to Paris in 2019. "
                    f"She works as a data scientist. "
                    f"Her favourite restaurant is Le Comptoir. "
                    f"Memory entry {i}: she visited the Louvre on a Tuesday."
                ),
                question=f"Where did Alice move in 2019? (dry-run question {i})",
                answer="Paris",
                memory_type=mt,
            )
        )
    return questions
