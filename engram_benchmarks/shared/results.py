"""Shared results schema for all Engram benchmark harnesses.

Warm/cold contract
------------------
Cold numbers are zero-by-construction for the metric we care about (tokens
avoided, TTFT saved).  Averaging cold results into headline warm-tier metrics
would produce meaningless numbers.  This schema enforces the separation:

- ``BaseResult.restore_mode`` labels every question/task "warm" or "cold".
- ``RunSummary`` exposes WARM-ONLY aggregate properties for the three KHA-394
  metrics.  Cold aggregates are provided separately.
- The ``mean_*`` headline properties on ``RunSummary`` explicitly operate over
  warm results only.

Compute-amortization
--------------------
See ``compute_amort.ComputeAmortization`` for the break-even definition.
Each ``BaseResult`` carries a ``compute_amort`` field populated from the three
token counts recorded during the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from .compute_amort import ComputeAmortization

RestoreMode = Literal["warm", "cold"]
SnapshotMode = Literal["mamba_only", "kv_capturing"]


@dataclass
class BaseResult:
    """Per-question/task result.  Subclass this for benchmark-specific fields."""

    item_id: str
    restore_mode: RestoreMode

    # Baseline (stateless, full context every time)
    baseline_ttft_s: float
    baseline_input_tokens: int
    baseline_output_tokens: int
    baseline_answer: str
    baseline_score: float

    # Engram path
    engram_ttft_s: float
    engram_input_tokens: int
    engram_output_tokens: int
    engram_answer: str
    engram_score: float

    # Snapshot metadata
    snapshot_mode: SnapshotMode = "mamba_only"
    # Tracks whether the /restore_snapshot RPC actually succeeded.
    # False means the runner fell back to the full prompt; any content match
    # on that result is a false positive, not proof of warm restore.
    # Always True in dry-run mode (stub path never fails).
    restore_success: bool = True

    # ------------------------------------------------------------------ #
    # Derived metrics                                                      #
    # ------------------------------------------------------------------ #

    @property
    def token_reduction(self) -> float:
        """Fraction of input tokens saved by Engram vs baseline.

        Only meaningful when restore_mode == "warm" (cold saves nothing by
        definition).  Callers should check restore_mode before using this.
        """
        if self.baseline_input_tokens == 0:
            return 0.0
        return (
            self.baseline_input_tokens - self.engram_input_tokens
        ) / self.baseline_input_tokens

    @property
    def ttft_speedup(self) -> float:
        """TTFT speedup ratio (baseline / engram).

        Only meaningful for warm results.  Returns 1.0 when engram TTFT is 0.
        """
        if self.engram_ttft_s == 0.0:
            return 1.0
        return self.baseline_ttft_s / self.engram_ttft_s

    @property
    def tokens_saved(self) -> int:
        return max(0, self.baseline_input_tokens - self.engram_input_tokens)

    def compute_amortization(self, n_warm_actual: int = 1) -> ComputeAmortization:
        """Return the break-even amortization model for this result.

        Parameters
        ----------
        n_warm_actual:
            Number of warm restores observed (defaults to 1 for per-question
            analysis; pass the actual warm count from RunSummary for aggregate).
        """
        # Cold engram input ≈ baseline (full context); warm engram = question only.
        # When restore_mode is "cold", engram_input_tokens IS the snapshot cost.
        # When restore_mode is "warm", we can't recover cold cost from this record
        # alone — callers should aggregate via RunSummary.compute_amortization().
        cold_input = (
            self.engram_input_tokens
            if self.restore_mode == "cold"
            else self.baseline_input_tokens  # proxy when cold record unavailable
        )
        return ComputeAmortization.from_result_pair(
            baseline_input_tokens=self.baseline_input_tokens,
            cold_engram_input_tokens=cold_input,
            warm_engram_input_tokens=self.engram_input_tokens
            if self.restore_mode == "warm"
            else 0,
            n_warm_actual=n_warm_actual,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["token_reduction"] = self.token_reduction
        d["ttft_speedup"] = self.ttft_speedup
        d["tokens_saved"] = self.tokens_saved
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaseResult":
        for k in ("token_reduction", "ttft_speedup", "tokens_saved"):
            d.pop(k, None)
        # snapshot_mode and restore_success were added later; tolerate old records.
        d.setdefault("snapshot_mode", "mamba_only")
        d.setdefault("restore_success", True)
        return cls(**d)


@dataclass
class RunSummary:
    """Aggregated run summary.

    Headline KHA-394 metrics operate over WARM results only.  Cold aggregates
    are provided under ``cold_*`` properties.  Never mix the two in a headline
    number.
    """

    benchmark: str
    model: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results: List[BaseResult] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Warm/cold split                                                      #
    # ------------------------------------------------------------------ #

    @property
    def warm_results(self) -> List[BaseResult]:
        """Warm results where the restore actually succeeded.

        Results where restore failed (restore_success=False) are excluded even
        if restore_mode was labelled "warm" — they fell back to the full prompt
        and their content match would be a false positive.
        """
        return [
            r for r in self.results
            if r.restore_mode == "warm" and r.restore_success
        ]

    @property
    def cold_results(self) -> List[BaseResult]:
        return [r for r in self.results if r.restore_mode == "cold"]

    # ------------------------------------------------------------------ #
    # KHA-394 — WARM-ONLY headline aggregates                             #
    # ------------------------------------------------------------------ #

    @property
    def warm_token_reduction(self) -> Optional[float]:
        """Mean token reduction across warm results only."""
        warm = self.warm_results
        if not warm:
            return None
        return sum(r.token_reduction for r in warm) / len(warm)

    @property
    def warm_ttft_speedup(self) -> Optional[float]:
        """Mean TTFT speedup across warm results only."""
        warm = self.warm_results
        if not warm:
            return None
        return sum(r.ttft_speedup for r in warm) / len(warm)

    @property
    def warm_tokens_saved_total(self) -> int:
        return sum(r.tokens_saved for r in self.warm_results)

    # ------------------------------------------------------------------ #
    # Compute amortization (aggregate)                                     #
    # ------------------------------------------------------------------ #

    def compute_amortization(self) -> Optional[ComputeAmortization]:
        """Break-even amortization across the whole run.

        Uses mean baseline and warm-engram token counts.  Returns None when
        there are no warm results.
        """
        warm = self.warm_results
        cold = self.cold_results
        if not warm or not cold:
            return None
        mean_baseline = sum(r.baseline_input_tokens for r in self.results) / len(
            self.results
        )
        mean_cold_engram = sum(r.engram_input_tokens for r in cold) / len(cold)
        mean_warm_engram = sum(r.engram_input_tokens for r in warm) / len(warm)
        return ComputeAmortization.from_result_pair(
            baseline_input_tokens=int(mean_baseline),
            cold_engram_input_tokens=int(mean_cold_engram),
            warm_engram_input_tokens=int(mean_warm_engram),
            n_warm_actual=len(warm),
        )

    # ------------------------------------------------------------------ #
    # Cold aggregates (separate from headline)                            #
    # ------------------------------------------------------------------ #

    @property
    def cold_token_reduction(self) -> Optional[float]:
        cold = self.cold_results
        if not cold:
            return None
        return sum(r.token_reduction for r in cold) / len(cold)

    @property
    def cold_ttft_speedup(self) -> Optional[float]:
        cold = self.cold_results
        if not cold:
            return None
        return sum(r.ttft_speedup for r in cold) / len(cold)

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        amort = self.compute_amortization()
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "timestamp": self.timestamp,
            "n_total": len(self.results),
            "n_warm": len(self.warm_results),
            "n_cold": len(self.cold_results),
            "warm_token_reduction": self.warm_token_reduction,
            "warm_ttft_speedup": self.warm_ttft_speedup,
            "warm_tokens_saved_total": self.warm_tokens_saved_total,
            "cold_token_reduction": self.cold_token_reduction,
            "cold_ttft_speedup": self.cold_ttft_speedup,
            "compute_amortization": amort.to_dict() if amort else None,
            "results": [r.to_dict() for r in self.results],
        }

    def to_jsonl(self, path: "str | Path") -> None:
        """Write summary header + one result per line."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            header = {k: v for k, v in self.to_dict().items() if k != "results"}
            f.write(json.dumps(header) + "\n")
            for r in self.results:
                f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: "str | Path") -> "RunSummary":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            raise ValueError(f"Empty JSONL: {path}")
        header = json.loads(lines[0])
        results = [BaseResult.from_dict(json.loads(l)) for l in lines[1:]]
        return cls(
            benchmark=header["benchmark"],
            model=header["model"],
            timestamp=header.get("timestamp", ""),
            results=results,
        )
