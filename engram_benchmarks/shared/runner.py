"""Base two-phase runner with correct warm-tier measurement protocol.

Warm-tier measurement protocol (retrofit)
------------------------------------------
The original stub implementation used ``_write_snapshot`` to create a local
JSON file as a "snapshot".  That file was a harness-side artefact with no
connection to the server's snapshot system, which means the mock dry-run and a
real GPU run used fundamentally different code paths.

This module implements the *real* Engram snapshot protocol, with the local
stub kept only as a fallback when ``snapshot_api_enabled=False`` (CPU dry-run).

Real-server sequence for each item
-----------------------------------
1. **Baseline pass** — streaming chat completion with full prompt, measure
   first-delta-token TTFT.  No snapshot involvement.

2. **Cold pass (pre-warm)** — streaming chat completion with full prompt.
   Latency NOT reported; this pass exists only to establish server-side state.
   After the response is received, call ``POST /save_snapshot`` with a pinned
   ``branch_name`` (``bm-{item_id}``).  Using a branch_name ensures the
   snapshot is retrievable from disk with a deterministic key regardless of
   how many subsequent auto-saves overwrite the warm tier.

3. **Warm pass (measured)** — call ``POST /restore_snapshot`` with the SAME
   pinned ``branch_name``.  Providing ``branch_name`` bypasses the warm-tier
   short-circuit in ``_load_snapshot_for_pending_restore`` (lines 107-115 of
   scheduler_snapshot_handlers.py), which only activates when BOTH
   ``turn_number`` and ``branch_name`` are None.  Then send streaming chat
   completion with the question-only prompt and record TTFT.

   The warm pass's own auto-saves (under ``every_turn`` server policy) write
   to the warm tier keyed by ``conversation_id``, but the cold-tier snapshot
   (keyed by ``branch_name``) remains on disk untouched.  Subsequent warm
   restores always pin ``branch_name``, so warm-tier auto-saves cannot
   corrupt the measurement.

Req-lifecycle safety
--------------------
``/save_snapshot`` is called after the cold-pass response is complete.  At
that point the server Req may already be freed.  This is safe: under
``every_turn`` policy, the scheduler auto-saves to the warm tier on Req
completion.  ``handle_save_snapshot`` falls through to the warm-tier path when
the Req is not found in any queue, reads state from host RAM, and persists it
to cold tier.  The call is therefore safe to issue after the HTTP response
arrives.

CPU dry-run mode
----------------
When ``snapshot_api_enabled=False``, no snapshot RPCs are issued.  The runner
writes a local JSON stub file (as before) and the warm pass reads from it to
simulate warm vs. cold labels.  Mock tests use this path.

result_cls consolidation
------------------------
``run_all`` creates results via ``self._result_cls(...)`` so every benchmark
subclass gets its own result type without overriding the full method.  Set the
class attribute ``_result_cls`` in the subclass.  RULER and GraphWalks no
longer need their own ``run_all`` overrides.

snapshot_mode label
-------------------
All results carry ``snapshot_mode`` (``"mamba_only"`` or ``"kv_capturing"``),
set at runner construction.  This tags baseline vs. future advanced runs so
their result records remain comparable.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, List, Literal, Optional, Type, TypeVar

from .http_client import ChatResult, MockHttpFn, _word_count, chat_completion
from .results import BaseResult, SnapshotMode
from .scoring import BaseScorer
from .snapshot_api import SnapshotApiClient

logger = logging.getLogger(__name__)

ItemT = TypeVar("ItemT")


class BaseTwoPhaseRunner(ABC, Generic[ItemT]):
    """Abstract two-phase runner.  Subclass once per benchmark.

    Parameters
    ----------
    model_url:
        Base URL of the OpenAI-compatible serving endpoint.
    snapshot_dir:
        Local directory for snapshot stubs (dry-run mode only).
    scorer:
        Scoring object.
    model:
        Model name sent in request payloads.
    max_tokens:
        Max tokens to generate per call.
    mock_fn:
        If provided, replaces HTTP calls (dry-run / unit-test mode).
    snapshot_api_enabled:
        If True (default), issue real ``/save_snapshot`` and
        ``/restore_snapshot`` RPCs to the Engram server.
        If False, use local JSON stubs (CPU dry-run / CI).
    warm_branch_prefix:
        Prefix for the branch_name used to pin cold-pass snapshots.
        Full branch_name = ``{prefix}-{item_id}``.
    snapshot_mode:
        Tag attached to every result.  Use ``"mamba_only"`` for the current
        Mamba-state-only baseline; ``"kv_capturing"`` for future KV variants.
    admin_api_key:
        Optional admin key for the snapshot RPCs.
    """

    _result_cls: ClassVar[Type[BaseResult]] = BaseResult

    def __init__(
        self,
        model_url: str,
        snapshot_dir: Path,
        scorer: BaseScorer,
        model: str = "default",
        max_tokens: int = 256,
        mock_fn: Optional[MockHttpFn] = None,
        snapshot_api_enabled: bool = False,
        warm_branch_prefix: str = "bm-warm",
        snapshot_mode: SnapshotMode = "mamba_only",
        admin_api_key: Optional[str] = None,
    ) -> None:
        self.model_url = model_url
        self.snapshot_dir = Path(snapshot_dir)
        self.scorer = scorer
        self.model = model
        self.max_tokens = max_tokens
        self.mock_fn = mock_fn
        self.snapshot_api_enabled = snapshot_api_enabled
        self.warm_branch_prefix = warm_branch_prefix
        self.snapshot_mode = snapshot_mode
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snap_api = SnapshotApiClient(
            base_url=model_url, api_key=admin_api_key
        )

    # ------------------------------------------------------------------ #
    # Subclass interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _item_id(self, item: ItemT) -> str: ...

    @abstractmethod
    def _build_full_prompt(self, item: ItemT) -> str: ...

    @abstractmethod
    def _build_warm_prompt(self, item: ItemT) -> str: ...

    @abstractmethod
    def _reference_answer(self, item: ItemT) -> Any: ...

    @abstractmethod
    def _snapshot_metadata(self, item: ItemT, full_prompt: str) -> dict: ...

    @abstractmethod
    def _extra_result_fields(self, item: ItemT) -> dict: ...

    def _scorer_for(self, item: ItemT) -> "BaseScorer":
        """Return the scorer to use for this item.

        Override this when scoring varies per item (e.g. RULER uses different
        match strategies per task_name).  Default returns ``self.scorer``.
        """
        return self.scorer

    # ------------------------------------------------------------------ #
    # Snapshot helpers                                                     #
    # ------------------------------------------------------------------ #

    def _branch_name(self, item: ItemT) -> str:
        """Deterministic branch_name key for this item's cold-pass snapshot."""
        return f"{self.warm_branch_prefix}-{self._item_id(item)}"

    # -- real server path --

    def _server_save_snapshot(self, item: ItemT, rid: str) -> bool:
        """Issue POST /save_snapshot to persist the cold-pass state.

        Safe to call after the HTTP response arrives even if the Req has been
        freed: under every_turn policy the state is in warm tier and
        handle_save_snapshot falls through to it.

        Returns True on success, False on failure (logged).
        """
        branch = self._branch_name(item)
        result = self._snap_api.save_snapshot(
            rid=rid,
            conversation_id=self._item_id(item),
            branch_name=branch,
        )
        if not result.get("success"):
            logger.warning(
                "save_snapshot failed for %s (branch=%s): %s",
                self._item_id(item), branch, result.get("message"),
            )
            return False
        logger.info("Snapshot saved: item=%s branch=%s", self._item_id(item), branch)
        return True

    def _server_restore_snapshot(self, item: ItemT) -> bool:
        """Issue POST /restore_snapshot with pinned branch_name.

        Passing branch_name bypasses the warm-tier short-circuit in
        _load_snapshot_for_pending_restore (scheduler_snapshot_handlers.py
        lines 107-115) which only activates when BOTH turn_number and
        branch_name are None.

        Returns True on success.
        """
        branch = self._branch_name(item)
        result = self._snap_api.restore_snapshot(
            conversation_id=self._item_id(item),
            branch_name=branch,
        )
        if not result.get("success"):
            logger.warning(
                "restore_snapshot failed for %s (branch=%s): %s",
                self._item_id(item), branch, result.get("message"),
            )
            return False
        logger.info("Snapshot staged for warm restore: item=%s branch=%s",
                    self._item_id(item), branch)
        return True

    # -- local stub path (dry-run / CI) --

    def _stub_path(self, item: ItemT) -> Path:
        return self.snapshot_dir / self._item_id(item) / "snapshot.json"

    def _stub_exists(self, item: ItemT) -> bool:
        return self._stub_path(item).exists()

    def _write_stub(self, item: ItemT, full_prompt: str) -> None:
        p = self._stub_path(item)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = self._snapshot_metadata(item, full_prompt)
        meta.update({"_stub": True, "_saved_at": time.time(),
                      "_branch": self._branch_name(item)})
        p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # HTTP helper                                                          #
    # ------------------------------------------------------------------ #

    def _call(self, prompt: str) -> ChatResult:
        messages = [{"role": "user", "content": prompt}]
        input_tokens = _word_count(prompt)
        return chat_completion(
            model_url=self.model_url,
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            input_token_count=input_tokens,
            mock_fn=self.mock_fn,
        )

    # ------------------------------------------------------------------ #
    # Three-pass run: baseline → cold Engram → warm Engram                #
    # ------------------------------------------------------------------ #

    def run_all(self, items: List[ItemT]) -> List[BaseResult]:
        """Run baseline, cold Engram, and warm Engram for each item.

        Protocol for each item:
        1. Baseline — full context, no snapshot, measure TTFT.
        2. Cold pass — full context, save snapshot with pinned branch_name
           (NOT counted as a warm metric).
        3. Warm pass — restore snapshot by pinned branch_name (bypasses
           warm-tier short-circuit), send question-only, measure TTFT.

        All results are labelled ``restore_mode="warm"`` because the warm
        measurement was deliberately established.
        """
        results: List[BaseResult] = []
        for item in items:
            item_id = self._item_id(item)
            full_prompt = self._build_full_prompt(item)
            warm_prompt = self._build_warm_prompt(item)
            ref = self._reference_answer(item)

            # 1. Baseline
            scorer = self._scorer_for(item)
            b = self._call(full_prompt)
            b_score = scorer.score(b.text, ref)

            # 2. Cold pass — establish snapshot
            cold_result = self._call(full_prompt)
            cold_rid = getattr(cold_result, "rid", item_id)  # rid if server sends it

            if self.snapshot_api_enabled:
                self._server_save_snapshot(item, rid=cold_rid)
                restored = self._server_restore_snapshot(item)
                if not restored:
                    logger.warning("Warm restore failed for %s; labelling cold", item_id)
                    restore_mode: Literal["warm", "cold"] = "cold"
                    w = self._call(full_prompt)  # fall back to full prompt
                else:
                    restore_mode = "warm"
                    w = self._call(warm_prompt)
            else:
                # Dry-run: local stub simulates warm/cold
                self._write_stub(item, full_prompt)
                assert self._stub_exists(item), "Stub must exist after cold pass"
                restore_mode = "warm"
                w = self._call(warm_prompt)

            w_score = scorer.score(w.text, ref)
            extra = self._extra_result_fields(item)

            results.append(
                self._result_cls(
                    item_id=item_id,
                    restore_mode=restore_mode,
                    snapshot_mode=self.snapshot_mode,
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
                    **extra,
                )
            )
            logger.info(
                "%s | baseline_ttft=%.3fs warm_ttft=%.3fs "
                "token_reduction=%.2f restore=%s mode=%s",
                item_id, b.ttft_s, w.ttft_s,
                results[-1].token_reduction, restore_mode, self.snapshot_mode,
            )
        return results

    def run_baseline_only(self, items: List[ItemT]) -> List[BaseResult]:
        """Baseline-only run — no Engram path, all results labelled 'cold'."""
        results = []
        for item in items:
            full_prompt = self._build_full_prompt(item)
            ref = self._reference_answer(item)
            b = self._call(full_prompt)
            b_score = self._scorer_for(item).score(b.text, ref)
            extra = self._extra_result_fields(item)
            results.append(
                self._result_cls(
                    item_id=self._item_id(item),
                    restore_mode="cold",
                    snapshot_mode=self.snapshot_mode,
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
                    **extra,
                )
            )
        return results
