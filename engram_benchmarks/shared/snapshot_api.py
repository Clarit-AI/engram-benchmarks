"""HTTP client for Engram snapshot RPCs (/save_snapshot, /restore_snapshot).

Used by BaseTwoPhaseRunner when snapshot_api_enabled=True (real GPU server).
In CPU dry-run mode (snapshot_api_enabled=False), this module is never called.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds


class SnapshotApiClient:
    """Thin wrapper around Engram's admin snapshot endpoints.

    Parameters
    ----------
    base_url:
        Base URL of the Engram serving endpoint, e.g. ``http://localhost:30000``.
    api_key:
        Optional admin API key sent as ``Authorization: Bearer <key>``.
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def save_snapshot(
        self,
        rid: str,
        conversation_id: str,
        branch_name: str,
        turn_number: Optional[int] = None,
    ) -> dict:
        """POST /save_snapshot — persist the cold-pass state to cold tier.

        Parameters
        ----------
        rid:
            Request ID from the cold-pass response.  Under ``every_turn``
            policy the Req may already be freed; ``handle_save_snapshot``
            falls through to the warm-tier state, so the call is still safe.
        conversation_id:
            Conversation identifier used as the snapshot key.
        branch_name:
            Pinned branch key; enables deterministic cold-tier retrieval on
            subsequent restores regardless of how many auto-saves occur.
        turn_number:
            Optional turn number; omit to let the server use the current turn.
        """
        payload: dict = {
            "rid": rid,
            "conversation_id": conversation_id,
            "branch_name": branch_name,
        }
        if turn_number is not None:
            payload["turn_number"] = turn_number

        url = f"{self.base_url}/save_snapshot"
        try:
            resp = requests.post(
                url, json=payload, headers=self._headers, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("save_snapshot request failed: %s", exc)
            return {"success": False, "message": str(exc)}

    def restore_snapshot(
        self,
        conversation_id: str,
        branch_name: Optional[str] = None,
        turn_number: Optional[int] = None,
        create_new_request: bool = False,
    ) -> dict:
        """POST /restore_snapshot — stage a snapshot for the next inference call.

        Passing ``branch_name`` bypasses the warm-tier short-circuit in
        ``_load_snapshot_for_pending_restore`` (scheduler_snapshot_handlers.py
        lines 107-115), which only activates when BOTH ``turn_number`` and
        ``branch_name`` are None.  Always pass ``branch_name`` for deliberate
        warm-tier measurement.

        Parameters
        ----------
        conversation_id:
            Conversation identifier.
        branch_name:
            Pinned branch key matching the one used in ``save_snapshot``.
            Pass this to force a cold-tier disk read instead of the warm-tier
            short-circuit.
        turn_number:
            Optional turn number; omit when using branch_name.
        create_new_request:
            If True, ask the server to create a new Req for this conversation.
        """
        payload: dict = {"conversation_id": conversation_id}
        if branch_name is not None:
            payload["branch_name"] = branch_name
        if turn_number is not None:
            payload["turn_number"] = turn_number
        if create_new_request:
            payload["create_new_request"] = True

        url = f"{self.base_url}/restore_snapshot"
        try:
            resp = requests.post(
                url, json=payload, headers=self._headers, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("restore_snapshot request failed: %s", exc)
            return {"success": False, "message": str(exc)}
