"""GraphWalks dry-run test suite.

All tests run without a live model server (mock_fn injected) and without
network access (synthetic data only).

Test classes
------------
- TestSetF1Scorer          — scorer unit tests
- TestParseAnswer          — parse_answer unit tests
- TestSyntheticGeneration  — generate_synthetic_question contract
- TestRunnerDryRun         — end-to-end two-phase runner with mock HTTP
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engram_benchmarks.shared.http_client import ChatResult, make_dry_run_mock
from engram_benchmarks.graphwalks.scoring import SetF1Scorer, parse_answer
from engram_benchmarks.graphwalks.runner import (
    GraphWalksRunner,
    Question,
    generate_synthetic_question,
)
from engram_benchmarks.graphwalks.results import GraphWalksRunSummary
from engram_benchmarks.graphwalks.download import generate_dry_run_data


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_gw_mock(answer_text: str = "Final Answer: [B]") -> object:
    """Return a mock HTTP fn that returns a GraphWalks-format answer.

    Simulates the Engram speedup contract:
    - Short prompt (warm/question-only): ttft_s=0.042, total_latency_s=0.150
    - Long prompt (baseline/full context): ttft_s=0.200, total_latency_s=0.500

    This ensures engram_ttft_s < baseline_ttft_s in all tests, mirroring the
    real-world expectation that snapshot-based warm restores have lower TTFT.
    """
    from engram_benchmarks.shared.http_client import ChatResult, _word_count

    def _mock(model_url: str, messages: list, model: str) -> ChatResult:
        prompt_text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        tokens = _word_count(prompt_text)
        # Full prompts contain the graph adjacency list ("Graph with N nodes").
        # Warm prompts are question-only (no graph text).
        is_full_context = "Graph with" in prompt_text or "->" in prompt_text
        if is_full_context:
            # Long prompt = baseline (full graph context): higher TTFT
            ttft_s, total_latency_s = 0.200, 0.500
        else:
            # Short prompt = warm restore (question only): lower TTFT
            ttft_s, total_latency_s = 0.042, 0.150
        return ChatResult(
            text=answer_text,
            ttft_s=ttft_s,
            total_latency_s=total_latency_s,
            input_tokens=tokens,
            output_tokens=_word_count(answer_text),
        )

    return _mock


def _make_questions(n: int = 3) -> list[Question]:
    """Build n synthetic Question objects."""
    questions = []
    for i in range(n):
        raw = generate_synthetic_question(num_nodes=5 + i, num_hops=2, seed=i)
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
# TestSetF1Scorer                                                      #
# ------------------------------------------------------------------ #

class TestSetF1Scorer:
    scorer = SetF1Scorer()

    def test_perfect_match_returns_1(self):
        prediction = "Some reasoning.\nFinal Answer: [A, B, C]"
        reference = ["A", "B", "C"]
        assert self.scorer.score(prediction, reference) == pytest.approx(1.0)

    def test_partial_match_between_0_and_1(self):
        prediction = "Final Answer: [A, B]"
        reference = ["A", "B", "C"]
        score = self.scorer.score(prediction, reference)
        assert 0.0 < score < 1.0

    def test_no_match_returns_0(self):
        prediction = "Final Answer: [X, Y]"
        reference = ["A", "B", "C"]
        assert self.scorer.score(prediction, reference) == pytest.approx(0.0)

    def test_empty_prediction_returns_0(self):
        prediction = "I don't know."
        reference = ["A", "B"]
        assert self.scorer.score(prediction, reference) == pytest.approx(0.0)

    def test_both_empty_returns_1(self):
        # Empty predicted from bad format; empty truth (unusual but defined)
        prediction = "No answer here."
        reference: list[str] = []
        # parse_answer returns [] for bad format; truth is []; both empty → 1.0
        assert self.scorer.score(prediction, reference) == pytest.approx(1.0)

    def test_single_node_perfect(self):
        prediction = "The answer is node B.\nFinal Answer: [B]"
        reference = ["B"]
        assert self.scorer.score(prediction, reference) == pytest.approx(1.0)

    def test_single_node_wrong(self):
        prediction = "Final Answer: [C]"
        reference = ["B"]
        assert self.scorer.score(prediction, reference) == pytest.approx(0.0)

    def test_superset_prediction_penalty(self):
        """Predicting extra nodes should reduce precision and thus F1."""
        prediction = "Final Answer: [A, B, C, D]"
        reference = ["A", "B"]
        score = self.scorer.score(prediction, reference)
        # F1 = 2 * (2/4) * (2/2) / (2/4 + 2/2) = 2 * 0.5 * 1 / 1.5 ≈ 0.667
        assert 0.0 < score < 1.0

    def test_string_reference_treated_as_single_node(self):
        prediction = "Final Answer: [B]"
        assert self.scorer.score(prediction, "B") == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# TestParseAnswer                                                      #
# ------------------------------------------------------------------ #

class TestParseAnswer:
    def test_correct_format_single_node(self):
        assert parse_answer("Final Answer: [B]") == ["B"]

    def test_correct_format_multiple_nodes(self):
        result = parse_answer("Final Answer: [A, B, C]")
        assert result == ["A", "B", "C"]

    def test_correct_format_on_last_line(self):
        response = "Let me think step by step.\nThe answer is B.\nFinal Answer: [B]"
        assert parse_answer(response) == ["B"]

    def test_missing_footer_returns_empty(self):
        assert parse_answer("The answer is B.") == []

    def test_footer_not_on_last_line_returns_empty(self):
        # "Final Answer:" appears on a middle line, not the last
        response = "Final Answer: [B]\nSome trailing text that breaks the format."
        assert parse_answer(response) == []

    def test_partial_format_missing_brackets_returns_empty(self):
        # Has the prefix but no brackets
        assert parse_answer("Final Answer: B") == []

    def test_empty_brackets_returns_empty_list(self):
        result = parse_answer("Final Answer: []")
        assert result == []

    def test_whitespace_nodes_stripped(self):
        result = parse_answer("Final Answer: [ A ,  B , C ]")
        assert result == ["A", "B", "C"]

    def test_no_space_after_colon(self):
        # Dataset cards show "Final Answer:" with varying spacing
        result = parse_answer("Final Answer:[A, B]")
        assert result == ["A", "B"]


# ------------------------------------------------------------------ #
# TestSyntheticGeneration                                              #
# ------------------------------------------------------------------ #

class TestSyntheticGeneration:
    def test_generates_dict_with_required_keys(self):
        raw = generate_synthetic_question(num_nodes=5, num_hops=2, seed=42)
        for key in ("question_id", "graph_text", "question", "answer", "num_hops", "num_nodes"):
            assert key in raw, f"Missing key: {key}"

    def test_answer_is_list_of_strings(self):
        raw = generate_synthetic_question(num_nodes=5, num_hops=2, seed=0)
        assert isinstance(raw["answer"], list)
        assert all(isinstance(x, str) for x in raw["answer"])

    def test_depth2_answer_is_correct(self):
        # Path A→B→C→D→E; depth 2 from A = C (index 2)
        raw = generate_synthetic_question(num_nodes=5, num_hops=2, seed=0)
        assert raw["answer"] == ["C"]

    def test_depth1_answer_is_correct(self):
        # Path A→B→C; depth 1 from A = B
        raw = generate_synthetic_question(num_nodes=3, num_hops=1, seed=0)
        assert raw["answer"] == ["B"]

    def test_graph_text_contains_edges(self):
        raw = generate_synthetic_question(num_nodes=4, num_hops=1, seed=0)
        assert "->" in raw["graph_text"]

    def test_question_references_correct_depth(self):
        raw = generate_synthetic_question(num_nodes=5, num_hops=2, seed=0)
        assert "depth 2" in raw["question"]

    def test_answer_in_final_answer_format(self):
        """Answer list can be serialised into Final Answer format."""
        raw = generate_synthetic_question(num_nodes=5, num_hops=2, seed=42)
        formatted = "Final Answer: [" + ", ".join(raw["answer"]) + "]"
        parsed = parse_answer(formatted)
        assert parsed == raw["answer"]

    def test_generate_dry_run_data_saves_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            questions = generate_dry_run_data(tmpdir, n=3)
            assert len(questions) == 3
            files = list(Path(tmpdir).glob("synthetic_*.json"))
            assert len(files) == 3

    def test_num_nodes_clamped_to_26(self):
        raw = generate_synthetic_question(num_nodes=30, num_hops=1, seed=0)
        assert raw["num_nodes"] == 26

    def test_num_hops_clamped_to_num_nodes_minus_1(self):
        raw = generate_synthetic_question(num_nodes=3, num_hops=10, seed=0)
        assert raw["num_hops"] <= raw["num_nodes"] - 1


# ------------------------------------------------------------------ #
# TestRunnerDryRun                                                     #
# ------------------------------------------------------------------ #

class TestRunnerDryRun:
    """End-to-end dry-run tests for GraphWalksRunner."""

    @pytest.fixture
    def questions(self):
        return _make_questions(n=3)

    @pytest.fixture
    def runner_and_results(self, tmp_path, questions):
        mock = _make_gw_mock("Final Answer: [B]")
        runner = GraphWalksRunner(
            model_url="http://localhost:30000",
            snapshot_dir=tmp_path / "snapshots",
            model="test-model",
            mock_fn=mock,
        )
        results = runner.run_all(questions)
        return runner, results

    def test_run_all_warm(self, runner_and_results, questions):
        _, results = runner_and_results
        assert len(results) == len(questions)
        for r in results:
            assert r.restore_mode == "warm"

    def test_warm_ttft_less_than_total_latency(self, runner_and_results):
        """Warm (Engram) TTFT must be strictly less than baseline TTFT on every result.

        The mock assigns ttft_s=0.042 for short/warm prompts and ttft_s=0.200 for
        long/baseline prompts, mirroring the real Engram speedup contract.
        """
        _, results = runner_and_results
        for r in results:
            print(
                f"[{r.item_id}] engram ttft_s={r.engram_ttft_s:.3f}s "
                f"baseline ttft_s={r.baseline_ttft_s:.3f}s "
                f"(mock warm=0.042 baseline=0.200)"
            )
            assert r.engram_ttft_s < r.baseline_ttft_s, (
                f"Expected engram_ttft_s ({r.engram_ttft_s}) < "
                f"baseline_ttft_s ({r.baseline_ttft_s}) for {r.item_id}"
            )

    def test_warm_input_tokens_less_than_baseline(self, runner_and_results):
        """Warm prompt (question only) must have fewer tokens than full prompt."""
        _, results = runner_and_results
        for r in results:
            assert r.engram_input_tokens < r.baseline_input_tokens, (
                f"Warm tokens ({r.engram_input_tokens}) should be < "
                f"baseline tokens ({r.baseline_input_tokens}) for {r.item_id}"
            )

    def test_token_reduction_positive(self, runner_and_results):
        """token_reduction must be strictly positive for warm results."""
        _, results = runner_and_results
        for r in results:
            assert r.token_reduction > 0.0, (
                f"Expected positive token_reduction for {r.item_id}, "
                f"got {r.token_reduction}"
            )

    def test_compute_amortization_break_even_populated(self, runner_and_results, questions):
        """compute_amortization() should return a populated object with a break-even value."""
        _, results = runner_and_results
        summary = GraphWalksRunSummary(
            benchmark="graphwalks",
            model="test-model",
            results=results,
        )
        # RunSummary.compute_amortization needs both warm and cold results.
        # run_all only produces warm results. Test per-result amortization instead.
        for r in results:
            amort = r.compute_amortization(n_warm_actual=1)
            assert amort is not None
            assert amort.snapshot_cost_tokens > 0
            # break_even_restores is None only when warm_savings_per_restore == 0
            # For our test, warm prompt is shorter, so savings should be > 0.
            assert amort.break_even_restores is not None
            assert amort.break_even_restores > 0

    def test_warm_only_aggregates_in_summary(self, runner_and_results, questions):
        """RunSummary warm aggregates must be populated."""
        _, results = runner_and_results
        summary = GraphWalksRunSummary(
            benchmark="graphwalks",
            model="test-model",
            results=results,
        )
        assert summary.warm_token_reduction is not None
        assert summary.warm_token_reduction > 0.0
        assert summary.warm_ttft_speedup is not None
        assert summary.warm_ttft_speedup > 0.0
        assert summary.warm_tokens_saved_total > 0
        # per_hop_breakdown should be populated
        breakdown = summary.per_hop_breakdown
        assert len(breakdown) > 0
        for hops, stats in breakdown.items():
            assert stats["n"] > 0
            assert "mean_engram_score" in stats
            assert "mean_token_reduction" in stats

    def test_jsonl_roundtrip(self, runner_and_results, tmp_path, questions):
        """Serialise to JSONL and reload; all key fields must survive."""
        _, results = runner_and_results
        summary = GraphWalksRunSummary(
            benchmark="graphwalks",
            model="test-model",
            results=results,
        )
        out_path = tmp_path / "results.jsonl"
        summary.to_jsonl(out_path)

        loaded = GraphWalksRunSummary.from_jsonl(out_path)
        assert loaded.benchmark == "graphwalks"
        assert loaded.model == "test-model"
        assert len(loaded.results) == len(results)

        for orig, loaded_r in zip(results, loaded.results):
            assert loaded_r.item_id == orig.item_id
            assert loaded_r.restore_mode == orig.restore_mode
            assert loaded_r.graph_size == orig.graph_size
            assert loaded_r.num_hops == orig.num_hops
            assert abs(loaded_r.engram_ttft_s - orig.engram_ttft_s) < 1e-9
            assert loaded_r.engram_input_tokens == orig.engram_input_tokens

    def test_sample_result_record(self, runner_and_results, capsys):
        """Print a sample result record showing key timing fields.

        Demonstrates that engram (warm) ttft_s < baseline ttft_s.
        Mock values: warm ttft_s=0.042s, baseline ttft_s=0.200s.
        """
        _, results = runner_and_results
        r = results[0]
        record = {
            "item_id": r.item_id,
            "restore_mode": r.restore_mode,
            "engram_ttft_s": r.engram_ttft_s,
            "baseline_ttft_s": r.baseline_ttft_s,
            "ttft_speedup": round(r.ttft_speedup, 4),
            "token_reduction": round(r.token_reduction, 4),
            "graph_size": r.graph_size,
            "num_hops": r.num_hops,
            "engram_score": r.engram_score,
            "baseline_score": r.baseline_score,
        }
        print("\nSample result record:")
        print(json.dumps(record, indent=2))
        # Key assertion: warm Engram TTFT is strictly less than baseline TTFT
        assert r.engram_ttft_s < r.baseline_ttft_s, (
            f"engram_ttft_s={r.engram_ttft_s} must be < baseline_ttft_s={r.baseline_ttft_s}"
        )
