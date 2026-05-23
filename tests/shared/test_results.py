"""Tests for the shared results schema and warm/cold aggregates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engram_benchmarks.shared.results import BaseResult, RunSummary


def _make_result(item_id: str, restore_mode: str, baseline_tokens: int, engram_tokens: int,
                 baseline_ttft: float = 1.0, engram_ttft: float = 0.1) -> BaseResult:
    return BaseResult(
        item_id=item_id,
        restore_mode=restore_mode,
        baseline_ttft_s=baseline_ttft,
        baseline_input_tokens=baseline_tokens,
        baseline_output_tokens=20,
        baseline_answer="baseline answer",
        baseline_score=0.8,
        engram_ttft_s=engram_ttft,
        engram_input_tokens=engram_tokens,
        engram_output_tokens=15,
        engram_answer="engram answer",
        engram_score=0.9,
    )


class TestBaseResult:
    def test_token_reduction_warm(self):
        r = _make_result("q1", "warm", baseline_tokens=1000, engram_tokens=50)
        assert r.token_reduction == pytest.approx(0.95)

    def test_token_reduction_zero_when_no_baseline(self):
        r = _make_result("q1", "warm", baseline_tokens=0, engram_tokens=0)
        assert r.token_reduction == 0.0

    def test_ttft_speedup_warm(self):
        r = _make_result("q1", "warm", baseline_tokens=100, engram_tokens=10,
                         baseline_ttft=1.0, engram_ttft=0.1)
        assert r.ttft_speedup == pytest.approx(10.0)

    def test_restore_mode_literal(self):
        r = _make_result("q1", "warm", 100, 10)
        assert r.restore_mode == "warm"
        r2 = _make_result("q2", "cold", 100, 100)
        assert r2.restore_mode == "cold"

    def test_to_dict_includes_derived_metrics(self):
        r = _make_result("q1", "warm", 1000, 100, baseline_ttft=2.0, engram_ttft=0.5)
        d = r.to_dict()
        assert "token_reduction" in d
        assert "ttft_speedup" in d
        assert "tokens_saved" in d
        assert d["tokens_saved"] == 900

    def test_from_dict_roundtrip(self):
        r = _make_result("q1", "warm", 500, 50, baseline_ttft=0.8, engram_ttft=0.08)
        d = r.to_dict()
        r2 = BaseResult.from_dict(d)
        assert r2.item_id == r.item_id
        assert r2.baseline_input_tokens == r.baseline_input_tokens
        assert r2.engram_input_tokens == r.engram_input_tokens


class TestRunSummaryWarmColdSeparation:
    def _make_summary(self) -> RunSummary:
        results = [
            _make_result("w1", "warm", 1000, 50, baseline_ttft=1.0, engram_ttft=0.05),
            _make_result("w2", "warm", 800, 40, baseline_ttft=0.8, engram_ttft=0.04),
            _make_result("c1", "cold", 900, 900, baseline_ttft=0.9, engram_ttft=0.9),
        ]
        return RunSummary(benchmark="test", model="test-model", results=results)

    def test_warm_cold_split(self):
        s = self._make_summary()
        assert len(s.warm_results) == 2
        assert len(s.cold_results) == 1

    def test_warm_token_reduction_excludes_cold(self):
        s = self._make_summary()
        warm_reductions = [r.token_reduction for r in s.warm_results]
        expected = sum(warm_reductions) / len(warm_reductions)
        assert s.warm_token_reduction == pytest.approx(expected)
        # Cold reduction should be 0 (100% of tokens used)
        assert s.cold_token_reduction == pytest.approx(0.0)

    def test_warm_ttft_speedup_excludes_cold(self):
        s = self._make_summary()
        # Cold speedup = 0.9/0.9 = 1.0 (no speedup)
        # Warm speedup should be >> 1.0
        assert s.warm_ttft_speedup > 10.0
        assert s.cold_ttft_speedup == pytest.approx(1.0)

    def test_no_warm_results_returns_none(self):
        s = RunSummary(benchmark="x", model="m", results=[
            _make_result("c1", "cold", 500, 500)
        ])
        assert s.warm_token_reduction is None
        assert s.warm_ttft_speedup is None

    def test_compute_amortization_populated(self):
        s = self._make_summary()
        amort = s.compute_amortization()
        assert amort is not None
        assert amort.break_even_restores is not None
        assert amort.break_even_restores > 0

    def test_to_dict_has_warm_and_cold_aggregates(self):
        s = self._make_summary()
        d = s.to_dict()
        assert "warm_token_reduction" in d
        assert "warm_ttft_speedup" in d
        assert "cold_token_reduction" in d
        assert "cold_ttft_speedup" in d
        assert "compute_amortization" in d

    def test_jsonl_roundtrip(self):
        s = self._make_summary()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            s.to_jsonl(path)
            assert path.exists()
            lines = path.read_text().splitlines()
            assert len(lines) == 4  # 1 header + 3 results
            header = json.loads(lines[0])
            assert header["n_warm"] == 2
            assert header["n_cold"] == 1


class TestComputeAmortization:
    def test_break_even_formula(self):
        r = _make_result("q1", "warm", baseline_tokens=1000, engram_tokens=50)
        # snapshot_cost = 1000 (proxy), warm_savings = 1000 - 50 = 950
        amort = r.compute_amortization(n_warm_actual=1)
        assert amort.break_even_restores == pytest.approx(1000 / 950, rel=0.01)

    def test_cumulative_saved_after_break_even(self):
        r = _make_result("q1", "warm", baseline_tokens=1000, engram_tokens=50)
        # 2 warm restores: 2 × 950 - 1000 = 900 saved
        amort = r.compute_amortization(n_warm_actual=2)
        assert amort.cumulative_saved_tokens == pytest.approx(900, abs=5)
        assert amort.paid_off

    def test_not_paid_off_before_break_even(self):
        r = _make_result("q1", "warm", baseline_tokens=10000, engram_tokens=50)
        # Very expensive snapshot, 1 warm restore barely dents it
        amort = r.compute_amortization(n_warm_actual=1)
        assert not amort.paid_off

    def test_amortization_dict_serialisable(self):
        r = _make_result("q1", "warm", 1000, 50)
        amort = r.compute_amortization(1)
        d = amort.to_dict()
        assert json.dumps(d)  # no TypeError
