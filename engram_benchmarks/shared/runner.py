"""Base two-phase runner implementing the proven Engram warm-restore contract.

Warm-restore protocol (correct)
--------------------------------
The canonical reference is the Engram test fixture at:
  test/registered/radix_cache/test_mamba_stateful_inference.py  (lines 136-173)

For each benchmark item the runner executes three passes:

1. Baseline pass
   POST /v1/chat/completions with the full prompt.  No snapshot involvement.
   Measures TTFT for the stateless baseline.

2. Cold pass (seed)
   POST /v1/chat/completions with the full prompt AND ``rid=item_id`` so the
   server tracks this request under a key we control.
   After the response arrives, POST /save_snapshot with the same rid.
   The server echoes back a ``snapshot_id`` (or the rid itself) confirming
   the SSM state is persisted.  The cold-pass latency is NOT reported.

3. Warm pass (measured)
   POST /restore_snapshot with:
     - ``conversation_id``: the real RID/snapshot_id from the save response
     - ``create_new_request``: True
     - ``continuation_ids``: client-side-tokenized new question (ints only)
     - ``max_new_tokens``: generation limit
   Read ``output_text`` directly from the restore response.
   DO NOT call /v1/chat/completions on the warm turn — doing so sends the
   full prompt and defeats the snapshot.

Why this is different from the defective pattern
-------------------------------------------------
The previous stub implementation (origin/main) wrote a local JSON file and
then called /v1/chat/completions with only the question — measuring
prompt-shortening, not SSM restore.  The fix/shared-warm-protocol branch
added SnapshotApiClient but still routed the warm turn through
/v1/chat/completions (defective) and passed item_id as conversation_id
instead of the real server-assigned RID.

This implementation routes the warm turn through /restore_snapshot only.

CPU dry-run mode
-----------------
When ``snapshot_api_enabled=False`` (default — safe for CI without a GPU
server), no snapshot RPCs are issued.  The runner writes a local JSON stub
and calls /v1/chat/completions for the warm pass.  This path measures
prompt-shortening only and is clearly labelled as such in logs.

snapshot_api_enabled=True is the correct path for real GPU benchmarking.

Tokenization
------------
``continuation_ids`` are produced by:
  AutoTokenizer.from_pretrained(model_path).encode(text, add_special_tokens=False)

Set ``model_path`` at runner construction.  The tokenizer is loaded lazily
and cached per-process.  If ``transformers`` is not installed, the live path
raises an ImportError; the dry-run path never imports it.

result_cls dispatch
-------------------
``run_all`` uses ``self._result_cls(...)`` so every benchmark subclass can
extend BaseResult without overriding the full method.  Set ``_result_cls``
as a class attribute in the subclass.

snapshot_mode label
-------------------
All results carry ``snapshot_mode`` (``"mamba_only"`` or ``"kv_capturing"``),
set at runner construction.  Use this to tag baseline vs. future KV variants.
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

# Module-level tokenizer cache — loaded once per process, never per-item.
_tokenizer_cache: dict[str, Any] = {}


def _get_tokenizer(model_path: str) -> Any:
    """Return a cached HuggingFace tokenizer for ``model_path``.

    Raises ImportError if ``transformers`` is not installed.  This is
    intentional — callers on the live path must have transformers available;
    callers on the dry-run path never reach this function.
    """
    if model_path not in _tokenizer_cache:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "'transformers' is required for client-side tokenization "
                "(snapshot_api_enabled=True path). "
                "Install it or set snapshot_api_enabled=False for dry-run mode."
            ) from exc
        _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
    return _tokenizer_cache[model_path]


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
        If True, issue real ``/save_snapshot`` and ``/restore_snapshot`` RPCs
        to the Engram server.  The warm turn reads output_text from the restore
        response and NEVER calls /v1/chat/completions.
        If False (default), use local JSON stubs (CPU dry-run / CI).
    model_path:
        HuggingFace model path or name used for client-side tokenization of
        ``continuation_ids``.  Required when ``snapshot_api_enabled=True``.
        Ignored in dry-run mode.
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
        model_path: str = "",
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
        self.model_path = model_path
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

    def _server_save_snapshot(self, item: ItemT, rid: str) -> str:
        """POST /save_snapshot and return the real server-assigned RID.

        The server response may echo back the rid, a snapshot_id, or a
        conversation_id.  We prefer snapshot_id, then conversation_id, then
        fall back to the rid we sent.  This returned value is what must be
        passed to /restore_snapshot as conversation_id.

        Returns
        -------
        str
            The server-confirmed RID / snapshot_id to use in restore_snapshot.
            Returns the empty string on failure (caller should treat as failure).
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
            return ""

        # Prefer snapshot_id from the server response; fall back to the rid we sent.
        server_rid = (
            result.get("snapshot_id")
            or result.get("rid")
            or result.get("conversation_id")
            or rid
        )
        logger.info(
            "Snapshot saved: item=%s branch=%s server_rid=%s",
            self._item_id(item), branch, server_rid,
        )
        return str(server_rid)

    def _server_restore_snapshot(
        self,
        item: ItemT,
        server_rid: str,
        warm_prompt: str,
    ) -> ChatResult:
        """POST /restore_snapshot and return a ChatResult from the response.

        This is the CORRECT warm turn.  It:
        - Tokenizes ``warm_prompt`` client-side using AutoTokenizer
        - Posts to /restore_snapshot with create_new_request=True and the real RID
        - Reads output_text directly from the restore response
        - NEVER calls /v1/chat/completions

        Returns None (logged as failure) if:
        - The response has success=False
        - output_text is absent from the response
        - continuation_ids is empty after tokenization

        Raises
        ------
        ValueError
            If model_path is empty (required for tokenization on live path).
        """
        if not self.model_path:
            raise ValueError(
                "model_path must be set when snapshot_api_enabled=True. "
                "It is used to tokenize continuation_ids client-side."
            )

        # Client-side tokenization — mirrors the canonical fixture exactly.
        tok = _get_tokenizer(self.model_path)
        continuation_ids: List[int] = tok.encode(
            warm_prompt, add_special_tokens=False
        )
        if not continuation_ids:
            logger.warning(
                "Tokenization of warm_prompt produced empty continuation_ids "
                "for item=%s; warm pass cannot proceed.",
                self._item_id(item),
            )
            return None  # type: ignore[return-value]

        branch = self._branch_name(item)
        t_start = time.perf_counter()
        result = self._snap_api.restore_snapshot(
            conversation_id=server_rid,
            continuation_ids=continuation_ids,
            max_new_tokens=self.max_tokens,
            create_new_request=True,
            branch_name=branch,
        )
        t_end = time.perf_counter()

        if not result.get("success"):
            logger.warning(
                "restore_snapshot failed for %s (server_rid=%s branch=%s): %s",
                self._item_id(item), server_rid, branch, result.get("message"),
            )
            return None  # type: ignore[return-value]

        output_text = result.get("output_text", "")
        # TTFT is not measurable from a synchronous restore response; we use
        # the total round-trip latency as a proxy.  The benchmark harness must
        # note this limitation in reports.
        ttft = t_end - t_start
        return ChatResult(
            text=output_text,
            ttft_s=ttft,
            total_latency_s=ttft,
            input_tokens=len(continuation_ids),
            output_tokens=_word_count(output_text),
        )

    # -- local stub path (dry-run / CI) --

    def _snap_path(self, item: ItemT) -> Path:
        return self.snapshot_dir / self._item_id(item) / "snapshot.json"

    def _snap_exists(self, item: ItemT) -> bool:
        return self._snap_path(item).exists()

    def _write_stub(self, item: ItemT, full_prompt: str) -> None:
        p = self._snap_path(item)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = self._snapshot_metadata(item, full_prompt)
        meta.update({
            "_stub": True,
            "_saved_at": time.time(),
            "_branch": self._branch_name(item),
        })
        p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.debug("Stub snapshot written: %s", p)

    # ------------------------------------------------------------------ #
    # HTTP helper (chat completions — NOT used for warm turn)             #
    # ------------------------------------------------------------------ #

    def _call(self, prompt: str, rid: Optional[str] = None) -> ChatResult:
        """Call /v1/chat/completions.

        IMPORTANT: Do not use this for the warm turn when snapshot_api_enabled=True.
        The warm turn must use _server_restore_snapshot() instead.
        """
        messages = [{"role": "user", "content": prompt}]
        input_tokens = _word_count(prompt)
        return chat_completion(
            model_url=self.model_url,
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            input_token_count=input_tokens,
            rid=rid,
            mock_fn=self.mock_fn,
        )

    # ------------------------------------------------------------------ #
    # Three-pass run: baseline → cold (seed) → warm (restore)             #
    # ------------------------------------------------------------------ #

    def run_all(self, items: List[ItemT]) -> List[BaseResult]:
        """Run baseline, cold seed, and warm restore for each item.

        Real-server protocol (snapshot_api_enabled=True):
        1. Baseline — full context via /v1/chat/completions, measure TTFT.
        2. Cold pass — full context + rid, save snapshot, capture real RID.
        3. Warm pass — /restore_snapshot with real RID + tokenized question,
                       read output_text from response, measure round-trip.
           Warm pass NEVER calls /v1/chat/completions.

        Dry-run protocol (snapshot_api_enabled=False, default):
        1. Baseline — full context mock call.
        2. Cold pass — full context mock call, write local stub.
        3. Warm pass — question-only mock call (prompt-shortening simulation).
           Clearly labelled as stub mode in logs.
        """
        results: List[BaseResult] = []
        for item in items:
            item_id = self._item_id(item)
            full_prompt = self._build_full_prompt(item)
            warm_prompt = self._build_warm_prompt(item)
            ref = self._reference_answer(item)
            scorer = self._scorer_for(item)

            # 1. Baseline
            b = self._call(full_prompt)
            b_score = scorer.score(b.text, ref)

            restore_success: bool
            restore_mode: Literal["warm", "cold"]
            w: ChatResult

            if self.snapshot_api_enabled:
                # 2. Cold pass: send rid so server tracks state under item_id.
                self._call(full_prompt, rid=item_id)

                # Save snapshot; get back the real server-assigned RID.
                server_rid = self._server_save_snapshot(item, rid=item_id)

                if not server_rid:
                    logger.warning(
                        "save_snapshot returned no RID for %s; "
                        "falling back to cold re-run. "
                        "Content match on this result is a false positive.",
                        item_id,
                    )
                    restore_mode = "cold"
                    restore_success = False
                    w = self._call(full_prompt)
                else:
                    # 3. Warm pass: restore via /restore_snapshot only.
                    w_restore = self._server_restore_snapshot(
                        item, server_rid=server_rid, warm_prompt=warm_prompt
                    )
                    if w_restore is None:
                        logger.warning(
                            "restore_snapshot failed for %s; "
                            "falling back to cold re-run. "
                            "Content match on this result is a false positive.",
                            item_id,
                        )
                        restore_mode = "cold"
                        restore_success = False
                        w = self._call(full_prompt)
                    else:
                        restore_mode = "warm"
                        restore_success = True
                        w = w_restore

            else:
                # Dry-run: local stub path (measures prompt-shortening, not SSM restore).
                logger.debug(
                    "item=%s snapshot_api_enabled=False; using dry-run stub path",
                    item_id,
                )
                self._call(full_prompt)  # cold pass — discard result
                self._write_stub(item, full_prompt)
                assert self._snap_exists(item), "Stub must exist after cold pass"
                restore_mode = "warm"
                restore_success = True
                w = self._call(warm_prompt)  # question only in dry-run

            w_score = scorer.score(w.text, ref)
            extra = self._extra_result_fields(item)

            results.append(
                self._result_cls(
                    item_id=item_id,
                    restore_mode=restore_mode,
                    snapshot_mode=self.snapshot_mode,
                    restore_success=restore_success,
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
                "token_reduction=%.2f restore=%s restore_success=%s mode=%s",
                item_id, b.ttft_s, w.ttft_s,
                results[-1].token_reduction, restore_mode,
                restore_success, self.snapshot_mode,
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
                    restore_success=False,
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
