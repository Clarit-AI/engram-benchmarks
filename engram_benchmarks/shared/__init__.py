"""Shared infrastructure for Engram benchmark harnesses."""
from .http_client import chat_completion, ChatResult
from .results import BaseResult, RunSummary, RestoreMode, SnapshotMode
from .runner import BaseTwoPhaseRunner
from .scoring import BaseScorer
from .compute_amort import ComputeAmortization
from .s3_writer import write_results_to_s3
from .snapshot_api import SnapshotApiClient

__all__ = [
    "chat_completion",
    "ChatResult",
    "BaseResult",
    "RunSummary",
    "RestoreMode",
    "SnapshotMode",
    "BaseTwoPhaseRunner",
    "BaseScorer",
    "ComputeAmortization",
    "write_results_to_s3",
    "SnapshotApiClient",
]
