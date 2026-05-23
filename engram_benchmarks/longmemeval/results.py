"""LongMemEval-specific result types.

Extends the shared BaseResult and RunSummary with a ``memory_type`` field
and a per-memory-type breakdown property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from engram_benchmarks.shared.results import BaseResult, RunSummary


@dataclass
class LMEResult(BaseResult):
    """Per-question result for LongMemEval.

    Adds ``memory_type`` to the shared schema.  Valid values per the dataset:
    ``episodic``, ``semantic``, ``temporal``, ``spatial``, ``factual``.
    """

    memory_type: str = "unknown"


class LMERunSummary(RunSummary):
    """Aggregated LongMemEval run summary with per-memory-type breakdown.

    All headline KHA-394 metrics (warm_token_reduction, warm_ttft_speedup, …)
    are inherited from RunSummary and operate over warm results only.
    The additional ``breakdown_by_memory_type`` property slices those same
    metrics per LongMemEval memory category.
    """

    @property
    def breakdown_by_memory_type(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Per-memory-type warm-metric breakdown.

        Returns a dict keyed by memory_type.  Each value is:
        ``{"warm_token_reduction": float|None, "warm_ttft_speedup": float|None,
           "n_warm": int}``.

        Only warm results contribute to these numbers (cold results are
        excluded, consistent with RunSummary's headline metrics).
        """
        warm: List[LMEResult] = [
            r for r in self.results
            if r.restore_mode == "warm" and isinstance(r, LMEResult)
        ]

        # Group by memory_type
        groups: Dict[str, List[LMEResult]] = {}
        for r in warm:
            groups.setdefault(r.memory_type, []).append(r)

        breakdown: Dict[str, Dict[str, Optional[float]]] = {}
        for mt, group in sorted(groups.items()):
            token_reduction = (
                sum(r.token_reduction for r in group) / len(group) if group else None
            )
            ttft_speedup = (
                sum(r.ttft_speedup for r in group) / len(group) if group else None
            )
            breakdown[mt] = {
                "n_warm": len(group),
                "warm_token_reduction": token_reduction,
                "warm_ttft_speedup": ttft_speedup,
            }
        return breakdown

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["breakdown_by_memory_type"] = self.breakdown_by_memory_type
        return d
