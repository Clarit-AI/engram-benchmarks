"""Base two-phase runner with deliberate warm-tier establishment.

Warm-tier measurement protocol
-------------------------------
A "warm" measurement is only meaningful when the snapshot was already present
BEFORE the timed call.  This module enforces the protocol:

1. ``pre_warm(item)``  — sends the full context to the model, saves a snapshot
   stub.  This is the cold pass; its latency is NOT reported as a warm metric.
2. ``measure_warm(item)`` — sends only the question/query with the snapshot
   assumed present, records the warm TTFT and token counts.
3. If ``measure_warm`` finds no snapshot, it labels the result "cold" honestly
   rather than silently falling back.

Subclasses implement ``_build_full_prompt``, ``_build_warm_prompt``,
``_snapshot_path``, ``_save_snapshot``, and ``_score``.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar

from .http_client import ChatResult, MockHttpFn, chat_completion, _word_count
from .results import BaseResult
from .scoring import BaseScorer

logger = logging.getLogger(__name__)

ItemT = TypeVar("ItemT")


class BaseTwoPhaseRunner(ABC, Generic[ItemT]):
    """Abstract two-phase runner.  Subclass once per benchmark."""

    def __init__(
        self,
        model_url: str,
        snapshot_dir: Path,
        scorer: BaseScorer,
        model: str = "default",
        max_tokens: int = 256,
        mock_fn: Optional[MockHttpFn] = None,
    ) -> None:
        self.model_url = model_url
        self.snapshot_dir = Path(snapshot_dir)
        self.scorer = scorer
        self.model = model
        self.max_tokens = max_tokens
        self.mock_fn = mock_fn  # None = live HTTP; non-None = dry-run/test
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Subclass interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _item_id(self, item: ItemT) -> str:
        """Unique identifier for this item (used as snapshot directory key)."""
        ...

    @abstractmethod
    def _build_full_prompt(self, item: ItemT) -> str:
        """Prompt including the full long context."""
        ...

    @abstractmethod
    def _build_warm_prompt(self, item: ItemT) -> str:
        """Prompt for a warm restore (question/query only, no context)."""
        ...

    @abstractmethod
    def _reference_answer(self, item: ItemT) -> Any:
        """Ground-truth answer for scoring."""
        ...

    @abstractmethod
    def _snapshot_metadata(self, item: ItemT, full_prompt: str) -> dict:
        """Metadata dict to store in the snapshot stub."""
        ...

    @abstractmethod
    def _extra_result_fields(self, item: ItemT) -> dict:
        """Extra fields merged into BaseResult.__init__ kwargs for subclasses
        that extend BaseResult with benchmark-specific fields."""
        ...

    # ------------------------------------------------------------------ #
    # Snapshot helpers                                                     #
    # ------------------------------------------------------------------ #

    def _snap_path(self, item: ItemT) -> Path:
        return self.snapshot_dir / self._item_id(item) / "snapshot.json"

    def _snap_exists(self, item: ItemT) -> bool:
        return self._snap_path(item).exists()

    def _write_snapshot(self, item: ItemT, full_prompt: str) -> None:
        p = self._snap_path(item)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = self._snapshot_metadata(item, full_prompt)
        meta["_stub"] = True
        meta["_saved_at"] = time.time()
        p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.debug("Snapshot saved: %s", p)

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

        For each item:
          1. Baseline  — full context, no snapshot.
          2. Cold pass — full context, saves snapshot (NOT reported as warm).
          3. Warm pass — question only, snapshot present (reported as warm).

        This guarantees that every warm measurement was preceded by a
        deliberate pre-warm, not a lucky leftover from a prior run.
        """
        results: List[BaseResult] = []
        for item in items:
            item_id = self._item_id(item)
            full_prompt = self._build_full_prompt(item)
            warm_prompt = self._build_warm_prompt(item)
            ref = self._reference_answer(item)

            # --- 1. Baseline ---
            b = self._call(full_prompt)
            b_score = self.scorer.score(b.text, ref)

            # --- 2. Cold Engram pass (establishes snapshot; metrics not reported) ---
            self._call(full_prompt)          # discard cold latency
            self._write_snapshot(item, full_prompt)

            # --- 3. Warm Engram pass (snapshot now present) ---
            assert self._snap_exists(item), "Snapshot must exist after cold pass"
            w = self._call(warm_prompt)
            w_score = self.scorer.score(w.text, ref)

            extra = self._extra_result_fields(item)
            results.append(
                BaseResult(
                    item_id=item_id,
                    restore_mode="warm",  # deliberate warm — always warm here
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
                "%s | baseline_ttft=%.3fs warm_ttft=%.3fs token_reduction=%.2f",
                item_id,
                b.ttft_s,
                w.ttft_s,
                results[-1].token_reduction,
            )
        return results

    def run_baseline_only(self, items: List[ItemT]) -> List[BaseResult]:
        """Baseline-only run — no Engram path, all results labelled 'cold'."""
        results = []
        for item in items:
            full_prompt = self._build_full_prompt(item)
            ref = self._reference_answer(item)
            b = self._call(full_prompt)
            b_score = self.scorer.score(b.text, ref)
            extra = self._extra_result_fields(item)
            results.append(
                BaseResult(
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
                    **extra,
                )
            )
        return results
