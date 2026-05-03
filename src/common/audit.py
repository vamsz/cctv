"""Audit log helper. Every privileged action is recorded immutably."""
from __future__ import annotations

from typing import Any, Optional

from src.common.db import session_scope
from src.evidence.models import AuditLog


def record(
    action: str,
    *,
    actor_id: Optional[int] = None,
    actor_label: Optional[str] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    detail: Optional[dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    with session_scope() as s:
        s.add(AuditLog(
            action=action,
            actor_id=actor_id,
            actor_label=actor_label,
            target_type=target_type,
            target_id=target_id,
            detail=detail,
            ip_address=ip_address,
            user_agent=user_agent,
        ))
