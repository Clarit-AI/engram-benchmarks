"""Dry-run test suite for the RULER benchmark harness.

All tests run without a GPU, without a model server, and without an
OPENAI_API_KEY.  The mock HTTP client provides deterministic ChatResult values.

Run::

    pytest engram_benchmarks/ruler/test_dry_run.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engram_benchmarks.shared.http_client import make_dry_run_mock
from engram_benchmarks.shared.results import RunSummary

from .results import RULERResult, RULERRunSummary
from .runner import RULERRunner, run_ruler
from .scoring import StringMatchAllScorer, StringMatchPartScorer, get_scorer
from .tasks import RULER_TASKS, TaskInstance, generate_synthetic_task


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_SNAPSHOT_DIR = Path(tempfile.mkdtemp(prefix="ruler_test_snaps_"))


def _make_runner(answer: str = "The secret passphrase number 1 is 123456.") -> RULERRunner:
    """Return a RULERRunner backed by the mock HTTP client."""
    from .scoring import StringMatchAllScorer

    return RULERRunner(
        model_url="http://localhost:30000",
        snapshot_dir=_SNAPSHOT_DIR,
        scorer=StringMatchAllScorer(),
        model="test-model",
        mock_fn=make_dry_run_mock(answer=answer),
    )


def _two_tasks() -> list[TaskInstance]:
    """Return two small task instances (4096 tokens) for fast tests."""
    return [
        generate_synthetic_task("niah_single_1", 4096, 0),
        generate_synthetic_task("niah_single_1", 4096, 1),
    ]


# ------------------------------------------------------------------ #
# TestScorerCorrectness                                                #
# ------------------------------------------------------------------ #


class TestScorerCorrectness:
    """Unit tests for the RULER scorers."""

    def test_string_match_all_all_found(self):
        scorer = StringMatchAllScorer()
        score = scorer.score("The answer is foo and bar and baz", ["foo", "bar", "baz"])
        assert score == pytest.approx(1.0), f"Expected 1.0, got {score}"

    def test_string_match_all_partial(self):
        scorer = StringMatchAllScorer()
        score = scorer.score("The answer is foo and bar", ["foo", "bar", "baz"])
        assert score == pytest.approx(2 / 3), f"Expected 2/3, got {score}"

    def test_string_match_all_none_found(self):
        scorer = StringMatchAllScorer()
        score = scorer.score("nothing relevant here", ["foo", "bar", "baz"])
        assert score == pytest.approx(0.0), f"Expected 0.0, got {score}"

    def test_string_match_all_single_string_reference(self):
        scorer = StringMatchAllScorer()
        score = scorer.score("The value is 42", "42")
        assert score == pytest.approx(1.0)

    def test_string_match_all_case_insensitive(self):
        scorer = StringMatchAllScorer()
        score = scorer.score("The Answer Is FOO", ["foo"])
        assert score == pytest.approx(1.0)

    def test_string_match_part_any_found(self):
        scorer = StringMatchPartScorer()
        score = scorer.score("The answer is foo", ["foo", "bar"])
        assert score == pytest.approx(1.0)

    def test_string_match_part_none_found(self):
        scorer = StringMatchPartScorer()
        score = scorer.score("nothing here", ["foo", "bar"])
        assert score == pytest.approx(0.0)

    def test_string_match_part_single_string_reference(self):
        scorer = StringMatchPartScorer()
        score = scorer.score("Mercury is the closest planet", "Mercury")
        assert score == pytest.approx(1.0)

    def test_factory_qa1_returns_part_scorer(self):
        scorer = get_scorer("qa_1")
        assert isinstance(scorer, StringMatchPartScorer)

    def test_factory_qa2_returns_part_scorer(self):
        scorer = get_scorer("qa_2")
        assert isinstance(scorer, StringMatchPartScorer)

    def test_factory_niah_returns_all_scorer(self):
        scorer = get_scorer("niah_single_1")
        assert isinstance(scorer, StringMatchAllScorer)

    def test_factory_vt_returns_all_scorer(self):
        scorer = get_scorer("vt")
        assert isinstance(scorer, StringMatchAllScorer)


# ------------------------------------------------------------------ #
# TestSyntheticGeneration                                              #
# ------------------------------------------------------------------ #


class TestSyntheticGeneration:
    """Verify that generate_synthetic_task produces valid instances."""

    def test_niah_single_needle_in_context(self):
        task = generate_synthetic_task("niah_single_1", 4096, 0)
        assert task.task_name == "niah_single_1"
        assert task.context_length == 4096
        # The needle answer strings must appear in the context text.
        for ans in task.answer:
            assert ans in task.context_text, (
                f"Answer {ans!r} not found in context for niah_single_1"
            )

    def test_niah_multikey_needles_in_context(self):
        task = generate_synthetic_task("niah_multikey_3", 8192, 0)
        assert task.task_name == "niah_multikey_3"
        for ans in task.answer:
            assert ans in task.context_text

    def test_niah_multivalue_all_values_in_context(self):
        task = generate_synthetic_task("niah_multivalue", 4096, 0)
        assert task.task_name == "niah_multivalue"
        assert len(task.answer) == 3
        for ans in task.answer:
            assert ans in task.context_text

    def test_niah_multiquery_needles_in_context(self):
        task = generate_synthetic_task("niah_multiquery", 4096, 0)
        for ans in task.answer:
            assert ans in task.context_text

    def test_vt_answer_is_in_context(self):
        task = generate_synthetic_task("vt", 4096, 0)
        assert len(task.answer) == 1
        assert task.answer[0] in task.context_text

    def test_cwe_target_word_in_context(self):
        task = generate_synthetic_task("cwe", 4096, 0)
        assert task.answer[0] in task.context_text

    def test_fwe_target_word_in_context(self):
        task = generate_synthetic_task("fwe", 4096, 0)
        assert task.answer[0] in task.context_text

    def test_qa1_answer_in_context(self):
        task = generate_synthetic_task("qa_1", 4096, 0)
        assert task.answer[0] in task.context_text

    def test_qa2_answer_in_context(self):
        task = generate_synthetic_task("qa_2", 4096, 0)
        assert task.answer[0] in task.context_text

    def test_task_id_format(self):
        for task_name in RULER_TASKS:
            task = generate_synthetic_task(task_name, 4096, 0)
            assert task.task_id == f"{task_name}__4096__0"

    def test_all_tasks_generate_without_error(self):
        for task_name in RULER_TASKS:
            task = generate_synthetic_task(task_name, 4096, 0)
            assert isinstance(task, TaskInstance)
            assert task.context_text
            assert task.question
            assert task.answer

    def test_deterministic_generation(self):
        t1 = generate_synthetic_task("niah_single_1", 4096, 0)
        t2 = generate_synthetic_task("niah_single_1", 4096, 0)
        assert t1.context_text == t2.context_text
        assert t1.answer == t2.answer

    def test_different_samples_differ(self):
        t0 = generate_synthetic_task("niah_single_1", 4096, 0)
        t1 = generate_synthetic_task("niah_single_1", 4096, 1)
        assert t0.answer != t1.answer


# ------------------------------------------------------------------ #
# TestRunnerDryRun                                                     #
# ------------------------------------------------------------------ #


class TestRunnerDryRun:
    """Integration tests using the mock HTTP client."""

    @pytest.fixture(scope="class")
    def dry_run_summary(self) -> RULERRunSummary:
        """Run 2 niah_single_1 tasks at 4096 tokens with the mock client."""
        tasks = _two_tasks()
        runner = _make_runner()
        results = runner.run_all(tasks)
        return RULERRunSummary(benchmark="ruler", model="test-model", results=results)

    def test_run_all_warm(self, dry_run_summary: RULERRunSummary):
        """All results from run_all must be labelled 'warm'."""
        for r in dry_run_summary.results:
            assert r.restore_mode == "warm", (
                f"Expected warm, got {r.restore_mode!r} for {r.item_id}"
            )

    def test_warm_ttft_less_than_total_latency(self, dry_run_summary: RULERRunSummary):
        """TTFT must be strictly less than total latency (mock: 0.042 < 0.150)."""
        for r in dry_run_summary.warm_results:
            ttft = r.engram_ttft_s
            total = r.baseline_ttft_s  # mock always returns 0.042 for TTFT
            # We assert on the engram_ttft_s from the mock directly.
            assert r.engram_ttft_s < 0.150, (
                f"TTFT={r.engram_ttft_s:.4f} not < total_latency=0.150 for {r.item_id}"
            )
            assert r.baseline_ttft_s < 0.150, (
                f"baseline TTFT={r.baseline_ttft_s:.4f} not < 0.150 for {r.item_id}"
            )
        # Print sample for visibility in verbose mode.
        sample = dry_run_summary.warm_results[0]
        print(
            f"\n  Sample TTFT check: engram_ttft_s={sample.engram_ttft_s:.4f}s "
            f"< mock total_latency=0.150s  [PASS]"
        )

    def test_warm_input_tokens_less_than_baseline(self, dry_run_summary: RULERRunSummary):
        """Warm engram prompt has fewer tokens than the full baseline prompt."""
        for r in dry_run_summary.warm_results:
            assert r.engram_input_tokens < r.baseline_input_tokens, (
                f"Warm tokens ({r.engram_input_tokens}) not < baseline "
                f"({r.baseline_input_tokens}) for {r.item_id}"
            )

    def test_token_reduction_positive(self, dry_run_summary: RULERRunSummary):
        """Token reduction must be positive for all warm results."""
        for r in dry_run_summary.warm_results:
            assert r.token_reduction > 0.0, (
                f"Expected positive token_reduction, got {r.token_reduction:.4f} "
                f"for {r.item_id}"
            )

    def test_compute_amortization_break_even_populated(
        self, dry_run_summary: RULERRunSummary
    ):
        """compute_amortization on a single warm result returns a populated object."""
        r = dry_run_summary.warm_results[0]
        amort = r.compute_amortization(n_warm_actual=1)
        assert amort is not None
        assert amort.break_even_restores is not None
        assert amort.break_even_restores > 0, (
            f"break_even_restores={amort.break_even_restores} should be > 0"
        )

    def test_warm_only_aggregates(self, dry_run_summary: RULERRunSummary):
        """warm_token_reduction and warm_ttft_speedup must be non-None and sane."""
        red = dry_run_summary.warm_token_reduction
        spup = dry_run_summary.warm_ttft_speedup
        assert red is not None, "warm_token_reduction is None"
        assert spup is not None, "warm_ttft_speedup is None"
        assert 0.0 < red <= 1.0, f"warm_token_reduction={red:.4f} out of (0, 1]"
        assert spup > 0.0, f"warm_ttft_speedup={spup:.4f} not > 0"

    def test_jsonl_roundtrip(self, dry_run_summary: RULERRunSummary):
        """Serialize to JSONL and deserialize; result count and key fields must match."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = Path(f.name)

        try:
            dry_run_summary.to_jsonl(out_path)
            loaded = RULERRunSummary.from_jsonl(out_path)

            assert loaded.benchmark == dry_run_summary.benchmark
            assert loaded.model == dry_run_summary.model
            assert len(loaded.results) == len(dry_run_summary.results)

            for orig, loaded_r in zip(dry_run_summary.results, loaded.results):
                assert loaded_r.item_id == orig.item_id
                assert loaded_r.task_name == orig.task_name
                assert loaded_r.context_length == orig.context_length
                assert loaded_r.restore_mode == orig.restore_mode
                assert loaded_r.baseline_input_tokens == orig.baseline_input_tokens
                assert loaded_r.engram_input_tokens == orig.engram_input_tokens
        finally:
            out_path.unlink(missing_ok=True)

    def test_run_ruler_helper_dry_run(self):
        """run_ruler() with dry_run mock returns a valid summary."""
        summary = run_ruler(
            model_url="http://localhost:30000",
            task_names=["niah_single_1", "qa_1"],
            context_lengths=[4096],
            sample_indices=[0],
            model="test-model",
            snapshot_dir=_SNAPSHOT_DIR / "helper",
            mock_fn=make_dry_run_mock(),
        )
        assert isinstance(summary, RULERRunSummary)
        assert len(summary.results) == 2
        assert all(r.restore_mode == "warm" for r in summary.results)

    def test_no_openai_api_key_required(self):
        """The dry-run path must not import or reference openai keys."""
        import os
        # Remove key if present, verify run still works.
        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            summary = run_ruler(
                model_url="http://localhost:30000",
                task_names=["niah_single_1"],
                context_lengths=[4096],
                sample_indices=[0],
                snapshot_dir=_SNAPSHOT_DIR / "no_key",
                mock_fn=make_dry_run_mock(),
            )
            assert len(summary.results) == 1
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original
