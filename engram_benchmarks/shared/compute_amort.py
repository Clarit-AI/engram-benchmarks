"""Compute-amortization model for Engram snapshot economics.

Break-even definition
---------------------
Creating a snapshot costs compute proportional to the tokens processed in the
cold pass (≈ ``baseline_input_tokens``).  Every subsequent warm restore avoids
re-processing that same prefix, saving ``warm_savings_tokens`` of prefill work.

    break_even_restores = snapshot_cost_tokens / warm_savings_per_restore

After ``n_warm`` warm restores the cumulative compute saved (net of the one-time
snapshot cost) is:

    cumulative_saved_tokens = n_warm × warm_savings_per_restore - snapshot_cost_tokens

The summary reports both the break-even count and the realised net saving.

Token counts serve as a linear proxy for prefill FLOPs (quadratic attention is
accounted for separately by the scaling factor, but for headline metrics the
proxy is standard in the literature).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ComputeAmortization:
    """Amortization economics for a single question / task."""

    snapshot_cost_tokens: int
    """One-time cost: tokens processed to create the snapshot (cold-path input)."""

    warm_savings_per_restore: int
    """Tokens avoided on each subsequent warm restore (baseline_input - warm_engram_input)."""

    n_warm_actual: int
    """Number of warm restores actually observed in this run."""

    @property
    def break_even_restores(self) -> Optional[float]:
        """How many warm restores pay off the snapshot creation cost.

        Returns None when warm_savings_per_restore is zero (nothing to amortise).
        """
        if self.warm_savings_per_restore <= 0:
            return None
        return self.snapshot_cost_tokens / self.warm_savings_per_restore

    @property
    def cumulative_saved_tokens(self) -> int:
        """Net tokens saved across all warm restores, minus the snapshot cost.

        Negative means the snapshot has not yet paid off.
        """
        return (
            self.n_warm_actual * self.warm_savings_per_restore
            - self.snapshot_cost_tokens
        )

    @property
    def paid_off(self) -> bool:
        """True when the cumulative saving has exceeded the snapshot cost."""
        return self.cumulative_saved_tokens >= 0

    def to_dict(self) -> dict:
        return {
            "snapshot_cost_tokens": self.snapshot_cost_tokens,
            "warm_savings_per_restore": self.warm_savings_per_restore,
            "n_warm_actual": self.n_warm_actual,
            "break_even_restores": self.break_even_restores,
            "cumulative_saved_tokens": self.cumulative_saved_tokens,
            "paid_off": self.paid_off,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ComputeAmortization":
        return cls(
            snapshot_cost_tokens=d["snapshot_cost_tokens"],
            warm_savings_per_restore=d["warm_savings_per_restore"],
            n_warm_actual=d["n_warm_actual"],
        )

    @classmethod
    def from_result_pair(
        cls,
        baseline_input_tokens: int,
        cold_engram_input_tokens: int,
        warm_engram_input_tokens: int,
        n_warm_actual: int,
    ) -> "ComputeAmortization":
        """Construct from the token counts recorded in a benchmark result.

        Parameters
        ----------
        baseline_input_tokens:
            Full-context token count (stateless baseline).
        cold_engram_input_tokens:
            Token count on the cold Engram pass (creates the snapshot;
            ≈ baseline for prefix-identical questions, slightly higher for
            Engram overhead).
        warm_engram_input_tokens:
            Token count on a warm restore (question-only prompt).
        n_warm_actual:
            Number of warm restores measured in this run.
        """
        return cls(
            snapshot_cost_tokens=cold_engram_input_tokens,
            warm_savings_per_restore=max(
                0, baseline_input_tokens - warm_engram_input_tokens
            ),
            n_warm_actual=n_warm_actual,
        )
