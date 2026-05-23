"""Scoring interface for Engram benchmark harnesses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union


class BaseScorer(ABC):
    """Abstract scorer — implement ``score`` in each benchmark."""

    @abstractmethod
    def score(self, prediction: str, reference: Union[str, list]) -> float:
        """Return a score in [0.0, 1.0]."""
        ...


class MockScorer(BaseScorer):
    """Dry-run scorer: 1.0 for non-empty predictions, 0.0 for empty."""

    def score(self, prediction: str, reference: Union[str, list]) -> float:
        return 1.0 if prediction.strip() else 0.0
