"""GraphWalks result schema.

Extends the shared ``BaseResult`` / ``RunSummary`` with graph-specific fields
and per-hop breakdown aggregates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from engram_benchmarks.shared.results import BaseResult, RunSummary


@dataclass
class GraphWalksResult(BaseResult):
    """Per-question result for the GraphWalks benchmark.

    Extra fields
    ------------
    graph_size:
        Number of nodes in the graph for this question.
    num_hops:
        BFS depth / hop count for this question.
    """

    graph_size: int = 0
    num_hops: int = 0

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["graph_size"] = self.graph_size
        d["num_hops"] = self.num_hops
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GraphWalksResult":
        for k in ("token_reduction", "ttft_speedup", "tokens_saved"):
            d.pop(k, None)
        return cls(**d)


@dataclass
class GraphWalksRunSummary(RunSummary):
    """Extended RunSummary with per-hop breakdown aggregates.

    ``per_hop`` is a dict mapping hop count (int) → dict of aggregated metrics
    for warm results at that hop depth.
    """

    results: List[GraphWalksResult] = field(default_factory=list)  # type: ignore[assignment]

    @property
    def per_hop_breakdown(self) -> Dict[int, dict]:
        """Warm results grouped by ``num_hops``, each with mean F1 and token reduction."""
        warm = [r for r in self.results if r.restore_mode == "warm"]
        hop_groups: Dict[int, List[GraphWalksResult]] = {}
        for r in warm:
            hop_groups.setdefault(r.num_hops, []).append(r)

        breakdown: Dict[int, dict] = {}
        for hops, group in sorted(hop_groups.items()):
            breakdown[hops] = {
                "n": len(group),
                "mean_engram_score": sum(r.engram_score for r in group) / len(group),
                "mean_baseline_score": sum(r.baseline_score for r in group) / len(group),
                "mean_token_reduction": sum(r.token_reduction for r in group) / len(group),
                "mean_ttft_speedup": sum(r.ttft_speedup for r in group) / len(group),
            }
        return breakdown

    @property
    def mean_graph_size(self) -> Optional[float]:
        warm = [r for r in self.results if r.restore_mode == "warm"]
        if not warm:
            return None
        return sum(r.graph_size for r in warm) / len(warm)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["per_hop_breakdown"] = self.per_hop_breakdown
        d["mean_graph_size"] = self.mean_graph_size
        return d

    def to_jsonl(self, path: "str | Path") -> None:
        """Write summary header + one result per line (GraphWalks schema)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            header = {k: v for k, v in self.to_dict().items() if k != "results"}
            f.write(json.dumps(header) + "\n")
            for r in self.results:
                f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: "str | Path") -> "GraphWalksRunSummary":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Empty JSONL: {path}")
        header = json.loads(lines[0])
        results = [GraphWalksResult.from_dict(json.loads(line)) for line in lines[1:]]
        return cls(
            benchmark=header["benchmark"],
            model=header["model"],
            timestamp=header.get("timestamp", ""),
            results=results,
        )
