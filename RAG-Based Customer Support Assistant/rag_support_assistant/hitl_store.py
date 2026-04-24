from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4


DEFAULT_TICKET_STORE_PATH = Path("data/hitl_tickets.json")
_STORE_LOCK = Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_tickets(store_path: Path) -> List[Dict[str, Any]]:
    if not store_path.exists():
        return []

    raw_data = json.loads(store_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError("Ticket store is corrupted. Expected a JSON list.")
    return raw_data


def _write_tickets(store_path: Path, tickets: List[Dict[str, Any]]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(tickets, indent=2), encoding="utf-8")


def create_ticket(
    *,
    session_id: str,
    user_query: str,
    escalation_reason: str,
    confidence: float,
    retrieved_context: List[Dict[str, Any]],
    store_path: Path = DEFAULT_TICKET_STORE_PATH,
) -> Dict[str, Any]:
    with _STORE_LOCK:
        tickets = _read_tickets(store_path)
        ticket = {
            "ticket_id": f"TCK-{uuid4().hex[:8].upper()}",
            "session_id": session_id,
            "user_query": user_query,
            "escalation_reason": escalation_reason,
            "confidence": confidence,
            "retrieved_context": retrieved_context,
            "status": "pending",
            "human_response": "",
            "resolved_by": "",
            "created_at": _utc_now_iso(),
            "resolved_at": "",
        }
        tickets.append(ticket)
        _write_tickets(store_path, tickets)
        return ticket


def list_tickets(
    *,
    store_path: Path = DEFAULT_TICKET_STORE_PATH,
    status: str | None = None,
    session_id: str | None = None,
) -> List[Dict[str, Any]]:
    with _STORE_LOCK:
        tickets = _read_tickets(store_path)

    if status is not None:
        tickets = [ticket for ticket in tickets if ticket.get("status") == status]
    if session_id is not None:
        tickets = [ticket for ticket in tickets if ticket.get("session_id") == session_id]

    return sorted(tickets, key=lambda ticket: ticket.get("created_at", ""))


def resolve_ticket(
    *,
    ticket_id: str,
    human_response: str,
    resolved_by: str,
    store_path: Path = DEFAULT_TICKET_STORE_PATH,
) -> Dict[str, Any]:
    with _STORE_LOCK:
        tickets = _read_tickets(store_path)
        for ticket in tickets:
            if ticket.get("ticket_id") != ticket_id:
                continue

            ticket["status"] = "resolved"
            ticket["human_response"] = human_response
            ticket["resolved_by"] = resolved_by
            ticket["resolved_at"] = _utc_now_iso()
            _write_tickets(store_path, tickets)
            return ticket

    raise KeyError(f"Ticket not found: {ticket_id}")
