"""RULER-specific result types extending the shared schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from engram_benchmarks.shared.results import BaseResult, RunSummary


@dataclass
class RULERResult(BaseResult):
    """Per-task RULER result with task identity fields.

    Inherits all BaseResult fields (TTFT, token counts, scores, …) and
    adds the two RULER-specific dimensions: task name and context length.
    """

    task_name: str = ""
    context_length: int = 0

    def to_dict(self) -> dict:
        d = super().to_dict()
        # Ensure RULER-specific fields are present (asdict in parent covers them,
        # but explicit inclusion makes schema obvious).
        d["task_name"] = self.task_name
        d["context_length"] = self.context_length
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RULERResult":
        for k in ("token_reduction", "ttft_speedup", "tokens_saved"):
            d.pop(k, None)
        return cls(**d)


class RULERRunSummary(RunSummary):
    """RunSummary extended with per-task and per-context-length breakdowns.

    All ``results`` stored here must be ``RULERResult`` instances (or dicts
    that can be loaded as such).
    """

    # ------------------------------------------------------------------ #
    # Per-task breakdown                                                   #
    # ------------------------------------------------------------------ #

    @property
    def ruler_results(self) -> List[RULERResult]:
        return [r for r in self.results if isinstance(r, RULERResult)]

    def results_by_task(self) -> Dict[str, List[RULERResult]]:
        """Group warm RULER results by task name."""
        out: Dict[str, List[RULERResult]] = {}
        for r in self.ruler_results:
            if r.restore_mode == "warm":
                out.setdefault(r.task_name, []).append(r)
        return out

    def warm_token_reduction_by_task(self) -> Dict[str, Optional[float]]:
        """Mean warm token reduction per task."""
        by_task = self.results_by_task()
        return {
            task: sum(r.token_reduction for r in rs) / len(rs)
            for task, rs in by_task.items()
            if rs
        }

    def warm_ttft_speedup_by_task(self) -> Dict[str, Optional[float]]:
        """Mean warm TTFT speedup per task."""
        by_task = self.results_by_task()
        return {
            task: sum(r.ttft_speedup for r in rs) / len(rs)
            for task, rs in by_task.items()
            if rs
        }

    # ------------------------------------------------------------------ #
    # Per-context-length breakdown                                         #
    # ------------------------------------------------------------------ #

    def results_by_context_length(self) -> Dict[int, List[RULERResult]]:
        """Group warm RULER results by context length."""
        out: Dict[int, List[RULERResult]] = {}
        for r in self.ruler_results:
            if r.restore_mode == "warm":
                out.setdefault(r.context_length, []).append(r)
        return out

    def warm_token_reduction_by_context_length(self) -> Dict[int, Optional[float]]:
        """Mean warm token reduction per context length."""
        by_ctx = self.results_by_context_length()
        return {
            ctx: sum(r.token_reduction for r in rs) / len(rs)
            for ctx, rs in by_ctx.items()
            if rs
        }

    def warm_ttft_speedup_by_context_length(self) -> Dict[int, Optional[float]]:
        """Mean warm TTFT speedup per context length."""
        by_ctx = self.results_by_context_length()
        return {
            ctx: sum(r.ttft_speedup for r in rs) / len(rs)
            for ctx, rs in by_ctx.items()
            if rs
        }

    # ------------------------------------------------------------------ #
    # Serialisation override                                               #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["warm_token_reduction_by_task"] = self.warm_token_reduction_by_task()
        d["warm_ttft_speedup_by_task"] = self.warm_ttft_speedup_by_task()
        d["warm_token_reduction_by_context_length"] = {
            str(k): v for k, v in self.warm_token_reduction_by_context_length().items()
        }
        d["warm_ttft_speedup_by_context_length"] = {
            str(k): v for k, v in self.warm_ttft_speedup_by_context_length().items()
        }
        return d

    def to_jsonl(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            header = {k: v for k, v in self.to_dict().items() if k != "results"}
            f.write(json.dumps(header) + "\n")
            for r in self.results:
                f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: "str | Path") -> "RULERRunSummary":
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Empty JSONL: {path}")
        header = json.loads(lines[0])
        results = [RULERResult.from_dict(json.loads(line)) for line in lines[1:]]
        return cls(
            benchmark=header["benchmark"],
            model=header["model"],
            timestamp=header.get("timestamp", ""),
            results=results,
        )
