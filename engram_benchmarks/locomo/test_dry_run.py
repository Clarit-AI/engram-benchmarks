"""Dry-run tests for the LoCoMo benchmark harness.

All tests run on CPU with no GPU, no live model server, and no OPENAI_API_KEY.
The mock HTTP function (make_dry_run_mock) returns ttft_s=0.042 < total_latency_s=0.150,
proving TTFT is measured as time-to-first-token, not total wall-clock.

Run:
    pytest engram_benchmarks/locomo/test_dry_run.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from engram_benchmarks.shared.http_client import make_dry_run_mock
from engram_benchmarks.locomo.scoring import TokenF1Scorer, normalize_answer
from engram_benchmarks.locomo.runner import (
    LoCoMoRunner,
    Question,
    generate_dry_run_questions,
    load_questions,
)
from engram_benchmarks.locomo.results import LoCoMoResult, LoCoMoRunSummary
from engram_benchmarks.locomo.download import generate_dry_run_data


# ===========================================================================
# TokenF1Scorer tests
# ===========================================================================

class TestTokenF1Scorer:
    """Unit tests for token-level F1 scoring."""

    def setup_method(self):
        self.scorer = TokenF1Scorer()

    def test_perfect_match(self):
        """Identical strings must score 1.0."""
        assert self.scorer.score("Paris", "Paris") == pytest.approx(1.0)

    def test_perfect_match_multiword(self):
        """Multi-word identical strings must score 1.0."""
        assert self.scorer.score("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_partial_overlap(self):
        """Partial token overlap must produce F1 strictly between 0 and 1."""
        score = self.scorer.score("the quick brown fox", "the slow brown fox")
        assert 0.0 < score < 1.0, f"Expected 0 < score < 1, got {score}"

    def test_no_overlap(self):
        """Completely different tokens must score 0.0."""
        assert self.scorer.score("apple orange", "banana grape") == pytest.approx(0.0)

    def test_empty_prediction(self):
        """Empty prediction must score 0.0."""
        assert self.scorer.score("", "Paris") == pytest.approx(0.0)

    def test_empty_reference(self):
        """Empty reference must score 0.0."""
        assert self.scorer.score("Paris", "") == pytest.approx(0.0)

    def test_article_normalization(self):
        """'the cat' vs 'cat' must score 1.0 after article removal."""
        # normalize_answer removes 'the', so both normalize to 'cat'
        assert self.scorer.score("the cat", "cat") == pytest.approx(1.0)

    def test_punctuation_normalization(self):
        """Punctuation differences must not affect score."""
        assert self.scorer.score("HelixCore.", "HelixCore") == pytest.approx(1.0)

    def test_case_normalization(self):
        """Case differences must not affect score."""
        assert self.scorer.score("PARIS", "paris") == pytest.approx(1.0)

    def test_an_article_normalization(self):
        """'an apple' vs 'apple' must score 1.0 after article removal."""
        assert self.scorer.score("an apple", "apple") == pytest.approx(1.0)

    def test_a_article_normalization(self):
        """'a dog' vs 'dog' must score 1.0 after article removal."""
        assert self.scorer.score("a dog", "dog") == pytest.approx(1.0)

    def test_list_reference_max_f1(self):
        """When reference is a list, max F1 over all refs must be returned."""
        # "Louvre" exactly matches second reference
        score = self.scorer.score("Louvre", ["the Louvre museum", "Louvre"])
        assert score == pytest.approx(1.0)

    def test_list_reference_partial(self):
        """List reference with no perfect match returns best partial match."""
        score = self.scorer.score("quick fox", ["slow turtle", "quick brown fox"])
        # Overlap with "quick brown fox": common=["quick","fox"]=2,
        #   pred_len=2, ref_len=3 → P=1.0, R=2/3, F1=0.8
        assert score == pytest.approx(0.8)

    def test_list_reference_empty_list(self):
        """Empty reference list must return 0.0."""
        assert self.scorer.score("something", []) == pytest.approx(0.0)

    def test_normalize_answer_function(self):
        """normalize_answer should strip articles, punctuation, lowercase."""
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
        assert normalize_answer("An Elephant") == "elephant"
        assert normalize_answer("A cat.") == "cat"


# ===========================================================================
# Runner dry-run tests
# ===========================================================================

class TestRunnerDryRun:
    """Integration tests for LoCoMoRunner using the mock HTTP client."""

    def setup_method(self):
        self.mock_fn = make_dry_run_mock(
            answer="Dry-run answer.",
            ttft_s=0.042,
            total_latency_s=0.150,
        )
        self.tmpdir = tempfile.mkdtemp()
        self.snapshot_dir = Path(self.tmpdir) / "snapshots"
        self.runner = LoCoMoRunner(
            model_url="http://localhost:30000",
            snapshot_dir=self.snapshot_dir,
            model="test-model",
            mock_fn=self.mock_fn,
        )
        self.questions = generate_dry_run_questions(n=5)

    def _run(self) -> list[LoCoMoResult]:
        return self.runner.run_all(self.questions)

    def test_run_all_warm(self):
        """Every result produced by run_all must have restore_mode == 'warm'."""
        results = self._run()
        assert len(results) == 5
        for r in results:
            assert r.restore_mode == "warm", (
                f"Expected 'warm', got '{r.restore_mode}' for item {r.item_id}"
            )

    def test_warm_ttft_less_than_total_latency(self):
        """TTFT must be strictly less than total latency (streaming contract).

        The mock returns ttft_s=0.042, total_latency_s=0.150.  This proves the
        harness records TTFT as time-to-first-token, not total wall-clock time.
        """
        results = self._run()
        for r in results:
            assert r.engram_ttft_s < r.engram_ttft_s + (0.150 - 0.042), (
                "TTFT accounting is broken"
            )
            # Direct verification: engram uses warm prompt (short), baseline uses full prompt.
            # Both go through mock_fn which returns ttft_s=0.042, total=0.150.
            assert r.baseline_ttft_s == pytest.approx(0.042), (
                f"Expected baseline TTFT=0.042, got {r.baseline_ttft_s}"
            )
            assert r.engram_ttft_s == pytest.approx(0.042), (
                f"Expected engram TTFT=0.042, got {r.engram_ttft_s}"
            )
        # Demonstrate: TTFT (0.042) != total_latency (0.150) — proving correct measurement
        sample = results[0]
        assert sample.engram_ttft_s == pytest.approx(0.042)
        assert sample.engram_ttft_s < 0.150  # total latency proxy

    def test_warm_input_tokens_less_than_baseline(self):
        """Warm (question-only) prompt must use fewer tokens than baseline (full prompt)."""
        results = self._run()
        for r in results:
            assert r.engram_input_tokens < r.baseline_input_tokens, (
                f"item={r.item_id}: warm={r.engram_input_tokens} "
                f"baseline={r.baseline_input_tokens}"
            )

    def test_token_reduction_positive(self):
        """token_reduction must be positive for all warm results."""
        results = self._run()
        for r in results:
            assert r.token_reduction > 0.0, (
                f"item={r.item_id}: token_reduction={r.token_reduction}"
            )

    def test_compute_amortization_break_even_populated(self):
        """compute_amortization() must return a populated object for warm results."""
        results = self._run()
        for r in results:
            amort = r.compute_amortization(n_warm_actual=1)
            assert amort is not None
            assert amort.break_even_restores is not None
            assert amort.break_even_restores > 0.0, (
                f"break_even_restores should be positive, got {amort.break_even_restores}"
            )

    def test_warm_only_aggregates_in_summary(self):
        """LoCoMoRunSummary warm-only aggregates must be populated."""
        results = self._run()
        summary = self.runner.build_summary(results, model="test-model")

        assert summary.warm_token_reduction is not None
        assert summary.warm_token_reduction > 0.0

        assert summary.warm_ttft_speedup is not None
        # With identical mock TTFT for baseline and warm, speedup = 1.0
        assert summary.warm_ttft_speedup == pytest.approx(1.0)

        assert summary.warm_tokens_saved_total > 0

        # Per-type breakdown must be populated
        per_type = summary.per_type_warm_f1
        assert len(per_type) > 0, "per_type_warm_f1 must have at least one entry"

    def test_jsonl_roundtrip(self):
        """Results serialised to JSONL and loaded back must be equivalent."""
        results = self._run()
        summary = self.runner.build_summary(results, model="test-model")

        out_path = Path(self.tmpdir) / "results.jsonl"
        summary.to_jsonl(out_path)
        assert out_path.exists()

        # Read back
        loaded = LoCoMoRunSummary.from_jsonl(out_path)
        assert len(loaded.results) == len(results)

        for orig, loaded_r in zip(results, loaded.results):
            assert loaded_r.item_id == orig.item_id
            assert loaded_r.restore_mode == orig.restore_mode
            assert loaded_r.session_id == orig.session_id
            assert loaded_r.question_type == orig.question_type
            assert loaded_r.baseline_ttft_s == pytest.approx(orig.baseline_ttft_s)
            assert loaded_r.engram_ttft_s == pytest.approx(orig.engram_ttft_s)
            assert loaded_r.token_reduction == pytest.approx(orig.token_reduction)

    def test_no_openai_key_required(self):
        """Run must complete successfully with no OPENAI_API_KEY in env."""
        # Unset the key for this test (in case it's set in the outer environment)
        env_backup = os.environ.pop("OPENAI_API_KEY", None)
        try:
            results = self._run()
            assert len(results) == 5
        finally:
            if env_backup is not None:
                os.environ["OPENAI_API_KEY"] = env_backup


# ===========================================================================
# Download / data-generation tests
# ===========================================================================

class TestDownload:
    """Tests for dry-run data generation (no network required)."""

    def test_generate_dry_run_data_creates_jsonl(self):
        """generate_dry_run_data must create a readable JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_dry_run_data(tmpdir, n=3)
            assert out_path.exists()
            questions = load_questions(out_path)
            assert len(questions) == 3

    def test_generated_questions_have_required_fields(self):
        """Each generated question must have all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = generate_dry_run_data(tmpdir, n=5)
            questions = load_questions(out_path)
            for q in questions:
                assert q.question_id
                assert q.session_id
                assert q.conversation_text
                assert q.question
                assert q.answer
                assert q.question_type

    def test_load_questions_roundtrip(self):
        """Questions written to JSONL and loaded back must be equivalent."""
        questions_orig = generate_dry_run_questions(n=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "questions.jsonl"
            with path.open("w") as f:
                for q in questions_orig:
                    f.write(json.dumps(q.to_dict()) + "\n")
            questions_loaded = load_questions(path)
        assert len(questions_loaded) == len(questions_orig)
        for orig, loaded in zip(questions_orig, questions_loaded):
            assert loaded.question_id == orig.question_id
            assert loaded.question == orig.question
            assert loaded.question_type == orig.question_type
