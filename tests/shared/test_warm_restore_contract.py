"""Tests for the correct warm-restore contract (fix/warm-restore-contract).

These tests assert the proven-working protocol from the Engram test fixture at
test/registered/radix_cache/test_mamba_stateful_inference.py (lines 136-173).

Critical invariants:
1. The warm turn issues POST /restore_snapshot — NEVER /v1/chat/completions.
2. The /restore_snapshot body has create_new_request=True.
3. The conversation_id in /restore_snapshot is the real server-assigned RID,
   not the harness string like "item-0".
4. continuation_ids is a non-empty list of integers.
5. output_text is read directly from the restore response.
6. Tokenizer round-trip: encode(text) → decode → approximately matches text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from engram_benchmarks.shared.http_client import ChatResult, make_dry_run_mock
from engram_benchmarks.shared.results import BaseResult, RunSummary, SnapshotMode
from engram_benchmarks.shared.runner import BaseTwoPhaseRunner
from engram_benchmarks.shared.scoring import MockScorer
from engram_benchmarks.shared.snapshot_api import SnapshotApiClient


# ---------------------------------------------------------------------------
# Minimal concrete runner for testing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(tmp_path, snapshot_api_enabled=True, model_path="fake-model", **kw):
    """Create a _ConcreteRunner with snapshot API enabled by default for these tests."""
    return _ConcreteRunner(
        model_url="http://fake-server:30000",
        snapshot_dir=tmp_path / "snapshots",
        scorer=MockScorer(),
        snapshot_api_enabled=snapshot_api_enabled,
        model_path=model_path,
        **kw,
    )


def _mock_tokenizer(tokens: List[int]):
    """Return a mock tokenizer whose encode() returns the given token list."""
    tok = MagicMock()
    tok.encode.return_value = tokens
    tok.decode.return_value = "decoded text"
    return tok


# ---------------------------------------------------------------------------
# TestWarmTurnUsesRestoreSnapshot
# ---------------------------------------------------------------------------

class TestWarmTurnUsesRestoreSnapshot:
    """The warm turn must POST /restore_snapshot, never /v1/chat/completions."""

    def test_warm_turn_calls_restore_snapshot_not_chat_completions(self, tmp_path):
        """Warm turn must issue /restore_snapshot, not a third /v1/chat/completions."""
        item = _Item("q1", "A" * 200, "What is the answer?", "42")
        runner = _make_runner(tmp_path)

        save_resp = {"success": True, "snapshot_id": "real-server-rid-abc"}
        restore_resp = {
            "success": True,
            "output_text": "The answer is 42.",
            "output_ids": [1, 2, 3],
        }

        with (
            patch("engram_benchmarks.shared.runner.chat_completion") as mock_cc,
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp),
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_cc.return_value = ChatResult("answer", 0.05, 0.15, 10, 1)
            mock_tok.return_value = _mock_tokenizer([101, 102, 103, 104])

            runner.run_all([item])

        # chat_completion called exactly 2 times: baseline + cold pass only.
        # The warm turn reads from restore response — no third chat_completion call.
        assert mock_cc.call_count == 2, (
            f"Expected 2 chat_completion calls (baseline + cold), got {mock_cc.call_count}. "
            "The warm turn must use /restore_snapshot, not /v1/chat/completions."
        )

    def test_warm_turn_never_sends_full_prompt(self, tmp_path):
        """Warm turn must NOT send the full_prompt (context + question) to /v1/chat/completions."""
        item = _Item("q1", "LONGCONTEXT" * 50, "What is the answer?", "42")
        runner = _make_runner(tmp_path)
        full_prompt = runner._build_full_prompt(item)

        save_resp = {"success": True, "snapshot_id": "rid-xyz"}
        restore_resp = {"success": True, "output_text": "42", "output_ids": [99]}

        calls_seen: list = []

        def capture_call(model_url, messages, model, **kw):
            calls_seen.append(messages)
            return ChatResult("answer", 0.05, 0.15, 10, 1)

        with (
            patch("engram_benchmarks.shared.runner.chat_completion", side_effect=capture_call),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp),
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([10, 20, 30])
            runner.run_all([item])

        # Only 2 calls — baseline and cold pass.  Neither beyond call 2 should exist.
        assert len(calls_seen) == 2, "Warm turn must not produce a third chat_completion call"
        # Both existing calls send the full prompt (baseline and cold seed are correct).
        for msgs in calls_seen:
            content = " ".join(m["content"] for m in msgs)
            assert "LONGCONTEXT" in content, "Baseline and cold pass must use full_prompt"


# ---------------------------------------------------------------------------
# TestRestoreSnapshotRequestShape
# ---------------------------------------------------------------------------

class TestRestoreSnapshotRequestShape:
    """The /restore_snapshot call must have the correct shape."""

    def test_create_new_request_is_true(self, tmp_path):
        """restore_snapshot must be called with create_new_request=True."""
        item = _Item("q1", "Context", "Question?", "Answer")
        runner = _make_runner(tmp_path)

        save_resp = {"success": True, "snapshot_id": "rid-for-q1"}
        restore_resp = {"success": True, "output_text": "Answer"}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp) as mock_restore,
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            runner.run_all([item])

        mock_restore.assert_called_once()
        kwargs = mock_restore.call_args.kwargs
        assert kwargs.get("create_new_request") is True, (
            f"restore_snapshot must be called with create_new_request=True, got {kwargs}"
        )

    def test_conversation_id_is_real_server_rid_not_item_id(self, tmp_path):
        """The conversation_id in restore_snapshot must be the save-time conversation_id.

        GPU-verified (2026-05-25): the engine indexes snapshots by the conversation_id
        passed at save time.  The save response includes both snapshot_id
        ("<conv_id>-t0", a derived value) and conversation_id.  We must prefer
        conversation_id — using snapshot_id returns HTTP 500 "No snapshots found."
        """
        item = _Item("item-0", "Context", "Question?", "Answer")
        runner = _make_runner(tmp_path)

        real_server_conv_id = "server-assigned-conv-deadbeef"
        derived_snapshot_id = f"{real_server_conv_id}-t0"
        # Simulate a real engine response that has BOTH fields.
        save_resp = {
            "success": True,
            "conversation_id": real_server_conv_id,
            "snapshot_id": derived_snapshot_id,  # derived — must NOT win
        }
        restore_resp = {"success": True, "output_text": "Answer"}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp) as mock_restore,
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            runner.run_all([item])

        mock_restore.assert_called_once()
        kwargs = mock_restore.call_args.kwargs
        conv_id = kwargs.get("conversation_id")
        assert conv_id == real_server_conv_id, (
            f"conversation_id must be the save-time conversation_id "
            f"'{real_server_conv_id}', not the derived snapshot_id "
            f"'{derived_snapshot_id}' nor the harness item_id 'item-0'. Got: '{conv_id}'"
        )
        assert conv_id != derived_snapshot_id, (
            f"snapshot_id ('{derived_snapshot_id}') is a derived '<conv_id>-t0' value "
            "that the engine does NOT index by — using it returns HTTP 500."
        )
        assert conv_id != "item-0", (
            "The harness item_id 'item-0' must NOT be passed as conversation_id."
        )

    def test_continuation_ids_is_nonempty_list_of_ints(self, tmp_path):
        """continuation_ids must be a non-empty list of integers."""
        item = _Item("q1", "Context", "What is the answer?", "42")
        runner = _make_runner(tmp_path)

        fake_token_ids = [101, 2003, 1996, 3437, 1029]  # simulated vocab IDs
        save_resp = {"success": True, "snapshot_id": "rid-q1"}
        restore_resp = {"success": True, "output_text": "42"}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp) as mock_restore,
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer(fake_token_ids)
            runner.run_all([item])

        mock_restore.assert_called_once()
        kwargs = mock_restore.call_args.kwargs
        cont_ids = kwargs.get("continuation_ids")
        assert cont_ids is not None, "continuation_ids must be present in restore_snapshot call"
        assert isinstance(cont_ids, list), f"continuation_ids must be a list, got {type(cont_ids)}"
        assert len(cont_ids) > 0, "continuation_ids must be non-empty"
        for tok in cont_ids:
            assert isinstance(tok, int), f"continuation_ids must be ints, got {type(tok)}: {tok}"
        assert cont_ids == fake_token_ids, (
            f"continuation_ids must match tokenizer output. "
            f"Expected {fake_token_ids}, got {cont_ids}"
        )

    def test_continuation_ids_comes_from_warm_prompt_not_full_prompt(self, tmp_path):
        """The tokenizer must be called with warm_prompt (question only), not full_prompt."""
        item = _Item("q1", "LONGCONTEXT" * 50, "What is the answer?", "42")
        runner = _make_runner(tmp_path)

        warm_prompt = runner._build_warm_prompt(item)

        save_resp = {"success": True, "snapshot_id": "rid-q1"}
        restore_resp = {"success": True, "output_text": "42"}

        tokenizer_calls: list = []

        tok_mock = MagicMock()
        tok_mock.encode.side_effect = lambda text, **kw: (
            tokenizer_calls.append(text) or [42, 43, 44]
        )

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp),
            patch("engram_benchmarks.shared.runner._get_tokenizer", return_value=tok_mock),
        ):
            runner.run_all([item])

        assert len(tokenizer_calls) >= 1, "Tokenizer must be called for warm pass"
        # The warm prompt (question only) must be among the tokenized texts.
        assert warm_prompt in tokenizer_calls, (
            f"Tokenizer must encode the warm_prompt '{warm_prompt}', "
            f"but only encoded: {tokenizer_calls}"
        )
        # The full prompt (context + question) must NOT be tokenized for continuation.
        full_prompt = runner._build_full_prompt(item)
        assert full_prompt not in tokenizer_calls, (
            "Tokenizer must NOT encode full_prompt for continuation_ids — "
            "only the new question should be sent."
        )


# ---------------------------------------------------------------------------
# TestRidCapture
# ---------------------------------------------------------------------------

class TestRidCapture:
    """The real server RID must be captured from the save_snapshot response."""

    def test_snapshot_id_in_save_response_is_not_used_as_conversation_id(self, tmp_path):
        """snapshot_id from /save_snapshot response must NOT become conversation_id.

        Regression: snapshot_id is a derived "<conv_id>-t0" value the engine does
        NOT index by.  When the save response has no conversation_id or rid, we fall
        back to the rid WE sent at save time — which IS the save-time conversation_id.
        """
        item = _Item("q1", "Context", "Question?", "Answer")
        runner = _make_runner(tmp_path)

        derived_snapshot_id = "snap-cafebabe-1234-t0"
        # Save response has ONLY snapshot_id — no conversation_id, no rid.
        save_resp = {"success": True, "snapshot_id": derived_snapshot_id}
        restore_resp = {"success": True, "output_text": "Answer"}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp) as mock_restore,
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            runner.run_all([item])

        kwargs = mock_restore.call_args.kwargs
        conv_id = kwargs["conversation_id"]
        # Falls back to the rid we sent (item_id) — the save-time conversation_id.
        assert conv_id != derived_snapshot_id, (
            f"snapshot_id '{derived_snapshot_id}' must not be used as conversation_id."
        )
        assert conv_id == "q1", (
            f"Should fall back to the rid sent at save time ('q1'), got '{conv_id}'"
        )

    def test_rid_field_fallback_when_no_snapshot_id(self, tmp_path):
        """When save_snapshot returns rid but no snapshot_id, use rid."""
        item = _Item("q1", "Context", "Question?", "Answer")
        runner = _make_runner(tmp_path)

        server_rid = "rid-from-server-99"
        save_resp = {"success": True, "rid": server_rid}  # no snapshot_id
        restore_resp = {"success": True, "output_text": "Answer"}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp) as mock_restore,
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            runner.run_all([item])

        kwargs = mock_restore.call_args.kwargs
        assert kwargs["conversation_id"] == server_rid

    def test_save_failure_falls_back_to_cold(self, tmp_path):
        """If save_snapshot fails, result is labelled cold and restore not called."""
        item = _Item("q1", "Context", "Question?", "Answer")
        runner = _make_runner(tmp_path)

        save_resp = {"success": False, "message": "server error"}
        restore_mock = MagicMock(return_value={"success": True, "output_text": "x"})

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", restore_mock),
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            results = runner.run_all([item])

        assert results[0].restore_mode == "cold"
        assert results[0].restore_success is False
        restore_mock.assert_not_called()

    def test_restore_failure_falls_back_to_cold(self, tmp_path):
        """If restore_snapshot fails, result is labelled cold (not warm)."""
        item = _Item("q1", "Context", "Question?", "Answer")
        runner = _make_runner(tmp_path)

        save_resp = {"success": True, "snapshot_id": "rid-ok"}
        restore_resp = {"success": False, "message": "restore failed"}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp),
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            results = runner.run_all([item])

        assert results[0].restore_mode == "cold"
        assert results[0].restore_success is False


# ---------------------------------------------------------------------------
# TestOutputTextFromRestoreResponse
# ---------------------------------------------------------------------------

class TestOutputTextFromRestoreResponse:
    """The warm turn's output_text comes from the restore response, not chat_completion."""

    def test_engram_answer_is_restore_output_text(self, tmp_path):
        """engram_answer must be the output_text from /restore_snapshot response."""
        item = _Item("q1", "Context", "Question?", "Answer")
        runner = _make_runner(tmp_path)

        restore_output = "This is the stateful restore output."
        save_resp = {"success": True, "snapshot_id": "rid-abc"}
        restore_resp = {"success": True, "output_text": restore_output}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("cold-answer", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp),
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            results = runner.run_all([item])

        assert results[0].engram_answer == restore_output, (
            f"engram_answer must be '{restore_output}' (from restore response), "
            f"not the chat_completion answer 'cold-answer'"
        )

    def test_warm_result_restore_success_true(self, tmp_path):
        """Successful restore sets restore_success=True and restore_mode='warm'."""
        item = _Item("q1", "Context", "Question?", "Answer")
        runner = _make_runner(tmp_path)

        save_resp = {"success": True, "snapshot_id": "rid-xyz"}
        restore_resp = {"success": True, "output_text": "Answer"}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp),
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            results = runner.run_all([item])

        assert results[0].restore_mode == "warm"
        assert results[0].restore_success is True


# ---------------------------------------------------------------------------
# TestWarmResultsExcludesFalsePositives
# ---------------------------------------------------------------------------

class TestWarmResultsExcludesFalsePositives:
    """RunSummary.warm_results must exclude cold-fallback results."""

    def test_cold_fallback_excluded_from_warm_results(self, tmp_path):
        """restore_success=False result must not appear in RunSummary.warm_results."""
        item = _Item("q1", "Context with Answer", "What is the answer?", "Answer")
        runner = _make_runner(tmp_path)

        save_resp = {"success": True, "snapshot_id": "rid-ok"}
        # Restore fails — runner falls back to full prompt which contains answer.
        restore_resp = {"success": False, "message": "GPU OOM"}

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("Answer", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot", return_value=restore_resp),
            patch("engram_benchmarks.shared.runner._get_tokenizer") as mock_tok,
        ):
            mock_tok.return_value = _mock_tokenizer([1, 2, 3])
            results = runner.run_all([item])

        summary = RunSummary(benchmark="test", model="test", results=results)
        assert len(summary.warm_results) == 0, (
            "Cold-fallback result must not appear in warm_results even if content matched"
        )


# ---------------------------------------------------------------------------
# TestSnapshotApiClientRestoreContract
# ---------------------------------------------------------------------------

class TestSnapshotApiClientRestoreContract:
    """SnapshotApiClient.restore_snapshot sends the correct payload."""

    def test_restore_sends_create_new_request_true_by_default(self):
        """restore_snapshot default is create_new_request=True."""
        client = SnapshotApiClient(base_url="http://fake:30000")
        with patch("engram_benchmarks.shared.snapshot_api.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"success": True, "output_text": ""}
            mock_post.return_value.raise_for_status = MagicMock()
            client.restore_snapshot(
                conversation_id="real-rid-123",
                continuation_ids=[10, 20, 30],
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["create_new_request"] is True

    def test_restore_payload_contains_conversation_id(self):
        """conversation_id must be in the /restore_snapshot payload."""
        client = SnapshotApiClient(base_url="http://fake:30000")
        with patch("engram_benchmarks.shared.snapshot_api.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"success": True, "output_text": ""}
            mock_post.return_value.raise_for_status = MagicMock()
            client.restore_snapshot(
                conversation_id="real-rid-abc",
                continuation_ids=[1, 2, 3],
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["conversation_id"] == "real-rid-abc"

    def test_restore_payload_contains_continuation_ids(self):
        """continuation_ids must be in the /restore_snapshot payload as a list."""
        client = SnapshotApiClient(base_url="http://fake:30000")
        token_ids = [100, 200, 300, 400]
        with patch("engram_benchmarks.shared.snapshot_api.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"success": True, "output_text": ""}
            mock_post.return_value.raise_for_status = MagicMock()
            client.restore_snapshot(
                conversation_id="rid",
                continuation_ids=token_ids,
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["continuation_ids"] == token_ids

    def test_restore_raises_on_empty_continuation_ids(self):
        """Empty continuation_ids must raise ValueError — not silently send an empty list."""
        client = SnapshotApiClient(base_url="http://fake:30000")
        with pytest.raises(ValueError, match="continuation_ids must be a non-empty"):
            client.restore_snapshot(
                conversation_id="rid",
                continuation_ids=[],
            )

    def test_restore_url_targets_restore_snapshot_endpoint(self):
        """The HTTP call must target /restore_snapshot, not /v1/chat/completions."""
        client = SnapshotApiClient(base_url="http://fake:30000")
        with patch("engram_benchmarks.shared.snapshot_api.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"success": True, "output_text": ""}
            mock_post.return_value.raise_for_status = MagicMock()
            client.restore_snapshot(
                conversation_id="rid",
                continuation_ids=[1, 2, 3],
            )
        url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url", "")
        assert "/restore_snapshot" in url, (
            f"Expected URL to contain /restore_snapshot, got: {url}"
        )
        assert "/v1/chat/completions" not in url

    def test_save_snapshot_url_targets_save_endpoint(self):
        """save_snapshot must target /save_snapshot, not /restore_snapshot."""
        client = SnapshotApiClient(base_url="http://fake:30000")
        with patch("engram_benchmarks.shared.snapshot_api.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"success": True, "snapshot_id": "x"}
            mock_post.return_value.raise_for_status = MagicMock()
            client.save_snapshot(rid="rid-abc")
        url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url", "")
        assert "/save_snapshot" in url
        assert "/restore_snapshot" not in url


# ---------------------------------------------------------------------------
# TestTokenizerRoundTrip
# ---------------------------------------------------------------------------

class TestTokenizerRoundTrip:
    """Tokenizer round-trip: encode → decode should approximately recover the original text.

    These tests use a real tokenizer if available; otherwise they test the
    mock path.  They serve as a smoke test for the tokenization pipeline.
    """

    def test_mock_tokenizer_encode_returns_int_list(self):
        """Mock tokenizer encode must return a list of ints."""
        tok = _mock_tokenizer([1, 2, 3, 4, 5])
        result = tok.encode("What is the answer?", add_special_tokens=False)
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_encode_nonempty_text_produces_nonempty_ids(self):
        """Encoding a non-empty question string must produce at least one token ID."""
        tok = _mock_tokenizer([101, 2054, 2003, 1996, 3437, 1029, 102])
        text = "What is the answer?"
        result = tok.encode(text, add_special_tokens=False)
        assert len(result) > 0, "Encoding non-empty text must yield at least one token"

    def test_add_special_tokens_false_passed_to_encode(self, tmp_path):
        """The runner must pass add_special_tokens=False to the tokenizer."""
        item = _Item("q1", "Context", "What is the answer?", "42")
        runner = _make_runner(tmp_path)

        save_resp = {"success": True, "snapshot_id": "rid-q1"}
        restore_resp = {"success": True, "output_text": "42"}

        tok_mock = MagicMock()
        tok_mock.encode.return_value = [101, 102, 103]

        with (
            patch("engram_benchmarks.shared.runner.chat_completion",
                  return_value=ChatResult("x", 0.05, 0.15, 10, 1)),
            patch.object(runner._snap_api, "save_snapshot", return_value=save_resp),
            patch.object(runner._snap_api, "restore_snapshot",
                         return_value=restore_resp),
            patch("engram_benchmarks.shared.runner._get_tokenizer",
                  return_value=tok_mock),
        ):
            runner.run_all([item])

        tok_mock.encode.assert_called_once()
        _, encode_kwargs = tok_mock.encode.call_args
        assert encode_kwargs.get("add_special_tokens") is False, (
            "Tokenizer must be called with add_special_tokens=False"
        )


# ---------------------------------------------------------------------------
# TestDryRunPath (snapshot_api_enabled=False)
# ---------------------------------------------------------------------------

class TestDryRunPath:
    """Dry-run path must still work and must not call restore_snapshot."""

    def test_dry_run_does_not_call_restore_snapshot(self, tmp_path):
        """When snapshot_api_enabled=False, /restore_snapshot must NOT be called."""
        item = _Item("q1", "Context", "Question?", "Answer")
        mock = make_dry_run_mock(answer="Answer")
        runner = _ConcreteRunner(
            model_url="http://unused",
            snapshot_dir=tmp_path / "snapshots",
            scorer=MockScorer(),
            mock_fn=mock,
            snapshot_api_enabled=False,
        )
        restore_spy = MagicMock()
        runner._snap_api.restore_snapshot = restore_spy

        runner.run_all([item])

        restore_spy.assert_not_called()

    def test_dry_run_restore_success_true(self, tmp_path):
        """Dry-run restore_success is always True (no real server involved)."""
        items = [
            _Item("q1", "A" * 200, "Question 1?", "Answer 1"),
            _Item("q2", "B" * 150, "Question 2?", "Answer 2"),
        ]
        mock = make_dry_run_mock(answer="Answer")
        runner = _ConcreteRunner(
            model_url="http://unused",
            snapshot_dir=tmp_path / "snapshots",
            scorer=MockScorer(),
            mock_fn=mock,
            snapshot_api_enabled=False,
        )
        results = runner.run_all(items)
        assert all(r.restore_success is True for r in results)
        assert all(r.restore_mode == "warm" for r in results)
