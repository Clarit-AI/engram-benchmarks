"""HTTP client for Engram snapshot RPCs (/save_snapshot, /restore_snapshot).

Used by BaseTwoPhaseRunner on the real server path (snapshot_api_enabled=True).
In CPU dry-run mode (snapshot_api_enabled=False) this module is never called.

Proven working contract
-----------------------
The protocol here mirrors the canonical reference in the Engram test fixture at
``test/registered/radix_cache/test_mamba_stateful_inference.py`` (lines 136-173):

Turn 1 (seed / cold pass):
  POST /v1/chat/completions with rid=<harness-chosen id>; capture response.
  POST /save_snapshot with that same rid.  The server echoes back a
  ``snapshot_id`` (or the rid) confirming the state was persisted.

Warm turn (restore + continue):
  POST /restore_snapshot with:
    - ``conversation_id``: the real RID captured at turn 1
    - ``create_new_request``: True  (default in this client)
    - ``continuation_ids``: client-side-tokenized new question tokens
    - ``max_new_tokens``: generation limit
  Read ``output_text`` / ``output_ids`` directly from the restore response.
  Do NOT call /v1/chat/completions on the warm turn.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60  # seconds — restore may take a full prefill pass


class SnapshotApiClient:
    """Thin wrapper around Engram's snapshot endpoints.

    Parameters
    ----------
    base_url:
        Base URL of the Engram serving endpoint (e.g. ``http://localhost:30000``).
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

    # ------------------------------------------------------------------ #
    # save_snapshot                                                        #
    # ------------------------------------------------------------------ #

    def save_snapshot(
        self,
        rid: str,
        conversation_id: Optional[str] = None,
        branch_name: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> dict:
        """POST /save_snapshot — persist the cold-pass state.

        Parameters
        ----------
        rid:
            The real request ID — must be the value embedded in the cold-pass
            /v1/chat/completions request via the ``rid`` field.  The server
            tracks state under this key.  Under ``every_turn`` auto-save policy
            the Req may already be freed by the time this call arrives; the
            server falls through to the warm-tier state, so the call is safe.
        conversation_id:
            Optional conversation identifier.  If omitted, the server uses the
            rid as the key.
        branch_name:
            Optional pinned branch key for deterministic cold-tier retrieval.
            When provided, bypasses the warm-tier short-circuit in
            ``_load_snapshot_for_pending_restore``.
        turn_number:
            Optional turn number; omit to let the server use the current turn.

        Returns
        -------
        dict
            Server response dict.  On success, typically contains
            ``{"success": True, "snapshot_id": <id>, ...}``.  On failure,
            ``{"success": False, "message": <error>}``.
        """
        payload: dict = {"rid": rid}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        if branch_name is not None:
            payload["branch_name"] = branch_name
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

    # ------------------------------------------------------------------ #
    # restore_snapshot                                                     #
    # ------------------------------------------------------------------ #

    def restore_snapshot(
        self,
        conversation_id: str,
        continuation_ids: List[int],
        max_new_tokens: int = 256,
        create_new_request: bool = True,
        branch_name: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> dict:
        """POST /restore_snapshot — restore state and generate a continuation.

        This is the ONLY entry point for the warm turn.  Do not call
        /v1/chat/completions on the warm turn — doing so sends the full prompt
        and defeats the purpose of the snapshot.

        The server reconstructs SSM context from the saved snapshot, then
        processes ``continuation_ids`` (the new question tokens only) and
        generates up to ``max_new_tokens`` output tokens.  Results are returned
        synchronously in the response body as ``output_text`` and ``output_ids``.

        Parameters
        ----------
        conversation_id:
            The real server-assigned RID from the turn-1 cold-pass response.
            This is NOT the harness string like "item-0" — it must be the
            value the server echoed back after /save_snapshot.
        continuation_ids:
            Client-side-tokenized list of integer token IDs for the new
            question.  Use ``AutoTokenizer.from_pretrained(model_path).encode(
            text, add_special_tokens=False)`` to produce this.  Must be a
            non-empty list of ints.
        max_new_tokens:
            Maximum tokens to generate.
        create_new_request:
            Ask the server to create a new Req object for this conversation
            before processing.  Default True — matching the canonical fixture.
        branch_name:
            Optional pinned branch key.  When provided, forces a cold-tier
            disk read instead of the warm-tier short-circuit.
        turn_number:
            Optional turn number; omit when using branch_name.

        Returns
        -------
        dict
            Server response dict.  On success, contains at minimum:
            ``{"success": True, "output_text": "...", "output_ids": [...]}``.
            On failure, ``{"success": False, "message": <error>}``.
        """
        if not continuation_ids:
            raise ValueError(
                "continuation_ids must be a non-empty list of ints; "
                "tokenize the new question before calling restore_snapshot()"
            )

        payload: dict = {
            "conversation_id": conversation_id,
            "create_new_request": create_new_request,
            "continuation_ids": continuation_ids,
            "max_new_tokens": max_new_tokens,
        }
        if branch_name is not None:
            payload["branch_name"] = branch_name
        if turn_number is not None:
            payload["turn_number"] = turn_number

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
