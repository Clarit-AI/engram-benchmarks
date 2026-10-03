"""LoCoMo-specific result schema.

Extends the shared BaseResult with fields that matter for LoCoMo:
- ``session_id``    — identifies which conversation the question is from
- ``question_type`` — one of "single_hop", "multi_hop", "temporal",
                      "open_domain", "adversarial" (mirrors LoCoMo categories)

Also provides ``LoCoMoRunSummary``, which extends ``RunSummary`` with a
per-question-type breakdown of warm token F1 scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from engram_benchmarks.shared.results import BaseResult, RunSummary


@dataclass
class LoCoMoResult(BaseResult):
    """Per-question result for the LoCoMo benchmark."""

    session_id: str = ""
    question_type: str = ""  # e.g. "temporal", "single_hop", "multi_hop", "open_domain"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["session_id"] = self.session_id
        d["question_type"] = self.question_type
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LoCoMoResult":
        # Pull derived fields that BaseResult.from_dict strips
        for k in ("token_reduction", "ttft_speedup", "tokens_saved"):
            d.pop(k, None)
        return cls(**d)


@dataclass
class LoCoMoRunSummary(RunSummary):
    """Run summary with per-question-type warm F1 breakdown.

    All KHA-394 headline metrics are inherited from RunSummary (warm-only).
    The ``per_type_warm_f1`` property adds a per-category view.
    """

    # Override results to accept LoCoMoResult objects
    results: List[LoCoMoResult] = field(default_factory=list)

    @property
    def per_type_warm_f1(self) -> Dict[str, Optional[float]]:
        """Mean warm engram F1 grouped by question_type.

        Returns a dict mapping question_type -> mean engram_score for warm
        results only.  Types with no warm results map to None.
        """
        warm = [r for r in self.results if r.restore_mode == "warm"]
        by_type: Dict[str, List[float]] = {}
        for r in warm:
            qt = r.question_type or "unknown"
            by_type.setdefault(qt, []).append(r.engram_score)
        return {
            qt: sum(scores) / len(scores) if scores else None
            for qt, scores in by_type.items()
        }

    @property
    def per_type_baseline_f1(self) -> Dict[str, Optional[float]]:
        """Mean baseline F1 grouped by question_type (all results)."""
        by_type: Dict[str, List[float]] = {}
        for r in self.results:
            qt = r.question_type or "unknown"
            by_type.setdefault(qt, []).append(r.baseline_score)
        return {
            qt: sum(scores) / len(scores) if scores else None
            for qt, scores in by_type.items()
        }

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["per_type_warm_f1"] = self.per_type_warm_f1
        d["per_type_baseline_f1"] = self.per_type_baseline_f1
        return d

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
    def from_jsonl(cls, path: "str | Path") -> "LoCoMoRunSummary":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Empty JSONL: {path}")
        header = json.loads(lines[0])
        results = [LoCoMoResult.from_dict(json.loads(line)) for line in lines[1:]]
        return cls(
            benchmark=header["benchmark"],
            model=header["model"],
            timestamp=header.get("timestamp", ""),
            results=results,
        )
