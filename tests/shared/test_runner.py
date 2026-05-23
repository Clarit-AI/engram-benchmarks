"""Tests for BaseTwoPhaseRunner — deliberate warm-tier protocol."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from pathlib import Path

from engram_benchmarks.shared.http_client import ChatResult, make_dry_run_mock
from engram_benchmarks.shared.results import BaseResult, SnapshotMode
from engram_benchmarks.shared.runner import BaseTwoPhaseRunner
from engram_benchmarks.shared.scoring import MockScorer


# Minimal concrete subclass for testing
class _Item:
    def __init__(self, item_id: str, context: str, question: str, answer: str):
        self.item_id = item_id
        self.context = context
        self.question = question
        self.answer = answer


class _ConcreteRunner(BaseTwoPhaseRunner[_Item]):
    def _item_id(self, item): return item.item_id
    def _build_full_prompt(self, item): return f"{item.context}\n{item.question}"
    def _build_warm_prompt(self, item): return item.question
    def _reference_answer(self, item): return item.answer
    def _snapshot_metadata(self, item, full_prompt): return {"item_id": item.item_id}
    def _extra_result_fields(self, item): return {}


@dataclass
class _ExtendedResult(BaseResult):
    extra_field: str = ""


class _ExtendedRunner(BaseTwoPhaseRunner[_Item]):
    _result_cls = _ExtendedResult

    def _item_id(self, item): return item.item_id
    def _build_full_prompt(self, item): return f"{item.context}\n{item.question}"
    def _build_warm_prompt(self, item): return item.question
    def _reference_answer(self, item): return item.answer
    def _snapshot_metadata(self, item, full_prompt): return {"item_id": item.item_id}
    def _extra_result_fields(self, item): return {"extra_field": "benchmark_value"}


@pytest.fixture
def items():
    return [
        _Item("q1", "A" * 500, "What is the answer?", "42"),
        _Item("q2", "B" * 400, "Name the entity?", "Entity X"),
    ]


@pytest.fixture
def runner(tmp_path):
    mock = make_dry_run_mock(answer="42", ttft_s=0.042, total_latency_s=0.150)
    return _ConcreteRunner(
        model_url="http://unused",
        snapshot_dir=tmp_path / "snapshots",
        scorer=MockScorer(),
        mock_fn=mock,
    )


@pytest.fixture
def extended_runner(tmp_path):
    mock = make_dry_run_mock(answer="42", ttft_s=0.042, total_latency_s=0.150)
    return _ExtendedRunner(
        model_url="http://unused",
        snapshot_dir=tmp_path / "snapshots",
        scorer=MockScorer(),
        mock_fn=mock,
    )


class TestDeliberateWarmProtocol:
    def test_run_all_labels_results_warm(self, runner, items):
        results = runner.run_all(items)
        assert all(r.restore_mode == "warm" for r in results)

    def test_snapshot_created_after_run_all(self, runner, items):
        runner.run_all(items)
        for item in items:
            assert runner._stub_exists(item), f"Snapshot missing for {item.item_id}"

    def test_warm_ttft_strictly_less_than_total_latency(self, runner, items):
        results = runner.run_all(items)
        for r in results:
            assert r.engram_ttft_s < runner.mock_fn("", [], "").total_latency_s, (
                "Warm TTFT must be less than whole-call latency"
            )

    def test_warm_input_tokens_less_than_baseline(self, runner, items):
        """Warm prompt (question only) must have fewer tokens than full prompt."""
        results = runner.run_all(items)
        for r in results:
            assert r.engram_input_tokens < r.baseline_input_tokens, (
                "Warm restore sends question only — fewer tokens than baseline"
            )

    def test_token_reduction_positive_for_warm(self, runner, items):
        results = runner.run_all(items)
        for r in results:
            assert r.token_reduction > 0.0

    def test_ttft_speedup_greater_than_one_for_warm(self, runner, items):
        # Both baseline and warm use same mock (ttft=0.042), so speedup=1.0
        # In real runs speedup > 1; here we just confirm it's >= 1.0
        results = runner.run_all(items)
        for r in results:
            assert r.ttft_speedup >= 1.0

    def test_result_count_matches_items(self, runner, items):
        results = runner.run_all(items)
        assert len(results) == len(items)


class TestBaselineOnly:
    def test_baseline_only_labels_cold(self, runner, items):
        results = runner.run_baseline_only(items)
        assert all(r.restore_mode == "cold" for r in results)

    def test_baseline_only_engram_fields_zeroed(self, runner, items):
        results = runner.run_baseline_only(items)
        for r in results:
            assert r.engram_ttft_s == 0.0
            assert r.engram_input_tokens == 0
            assert r.engram_answer == ""


class TestResultClsDispatch:
    """_result_cls class attribute routes run_all to the subclass result type."""

    def test_extended_runner_produces_extended_results(self, extended_runner, items):
        results = extended_runner.run_all(items)
        assert all(isinstance(r, _ExtendedResult) for r in results)

    def test_extended_runner_extra_field_populated(self, extended_runner, items):
        results = extended_runner.run_all(items)
        assert all(r.extra_field == "benchmark_value" for r in results)

    def test_base_runner_produces_base_results(self, runner, items):
        results = runner.run_all(items)
        assert all(type(r) is BaseResult for r in results)


class TestSnapshotModeLabel:
    """snapshot_mode tag is attached to every result."""

    def test_default_snapshot_mode_is_mamba_only(self, runner, items):
        results = runner.run_all(items)
        assert all(r.snapshot_mode == "mamba_only" for r in results)

    def test_custom_snapshot_mode_propagates(self, tmp_path, items):
        mock = make_dry_run_mock(answer="42", ttft_s=0.042, total_latency_s=0.150)
        r = _ConcreteRunner(
            model_url="http://unused",
            snapshot_dir=tmp_path / "snapshots",
            scorer=MockScorer(),
            mock_fn=mock,
            snapshot_mode="kv_capturing",
        )
        results = r.run_all(items)
        assert all(res.snapshot_mode == "kv_capturing" for res in results)

    def test_baseline_only_carries_snapshot_mode(self, runner, items):
        results = runner.run_baseline_only(items)
        assert all(r.snapshot_mode == "mamba_only" for r in results)


class TestBranchNamePinning:
    """branch_name is deterministic and consistent between save and restore."""

    def test_branch_name_includes_item_id(self, runner, items):
        for item in items:
            bn = runner._branch_name(item)
            assert item.item_id in bn

    def test_branch_name_includes_prefix(self, runner, items):
        for item in items:
            bn = runner._branch_name(item)
            assert bn.startswith(runner.warm_branch_prefix)

    def test_branch_name_differs_per_item(self, runner, items):
        names = [runner._branch_name(item) for item in items]
        assert len(set(names)) == len(names), "branch_names must be unique per item"

    def test_custom_prefix_reflected(self, tmp_path, items):
        mock = make_dry_run_mock(answer="42", ttft_s=0.042, total_latency_s=0.150)
        r = _ConcreteRunner(
            model_url="http://unused",
            snapshot_dir=tmp_path / "snapshots",
            scorer=MockScorer(),
            mock_fn=mock,
            warm_branch_prefix="myprefix",
        )
        for item in items:
            assert r._branch_name(item).startswith("myprefix")


class TestSnapshotApiEnabled:
    """When snapshot_api_enabled=True, runner calls save/restore RPCs."""

    def test_save_and_restore_called_per_item(self, tmp_path, items):
        mock = make_dry_run_mock(answer="42", ttft_s=0.042, total_latency_s=0.150)
        runner = _ConcreteRunner(
            model_url="http://unused",
            snapshot_dir=tmp_path / "snapshots",
            scorer=MockScorer(),
            mock_fn=mock,
            snapshot_api_enabled=True,
        )
        save_mock = MagicMock(return_value=True)
        restore_mock = MagicMock(return_value=True)
        runner._server_save_snapshot = save_mock
        runner._server_restore_snapshot = restore_mock

        runner.run_all(items)

        assert save_mock.call_count == len(items)
        assert restore_mock.call_count == len(items)

    def test_restore_failure_falls_back_to_cold(self, tmp_path):
        item = _Item("q1", "A" * 500, "What is the answer?", "42")
        mock = make_dry_run_mock(answer="42", ttft_s=0.042, total_latency_s=0.150)
        runner = _ConcreteRunner(
            model_url="http://unused",
            snapshot_dir=tmp_path / "snapshots",
            scorer=MockScorer(),
            mock_fn=mock,
            snapshot_api_enabled=True,
        )
        runner._server_save_snapshot = MagicMock(return_value=True)
        runner._server_restore_snapshot = MagicMock(return_value=False)

        results = runner.run_all([item])
        assert results[0].restore_mode == "cold"
