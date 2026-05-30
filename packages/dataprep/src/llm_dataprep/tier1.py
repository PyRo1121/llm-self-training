"""Train tier assignment per PLAN.md (tier-1 gate)."""

from __future__ import annotations

from typing import Any

ACCEPTED_LABELS = frozenset({"accepted", "edited_heavily"})
EXEC_PASS = frozenset({"pass"})
VERIFY_OK = frozenset({"cursor_ok", "pass"})


def label_ok(label: str | None) -> bool:
    return (label or "accepted") in ACCEPTED_LABELS


def exec_verify_ok(meta: dict[str, Any], *, bootstrap: bool) -> bool:
    """PLAN: exec==pass OR verify==cursor_ok; bootstrap allows unknown until link_logs_to_diffs."""
    exec_status = (meta.get("exec") or "unknown").lower()
    verify = (meta.get("verify") or "unknown").lower()
    if exec_status in EXEC_PASS or verify in VERIFY_OK:
        return True
    if bootstrap and exec_status == "unknown" and verify == "unknown":
        return True
    return False


def safety_ok(meta: dict[str, Any]) -> bool:
    safety = meta.get("safety")
    if isinstance(safety, dict):
        return bool(safety.get("ok", True))
    return meta.get("safety_ok", True) is not False


def assign_train_tier(
    meta: dict[str, Any],
    *,
    bootstrap: bool,
    quality_ok: bool,
) -> int:
    """
    Returns train_tier: 1 = SFT tier-1, 2 = replay-only, 0 = drop/quarantine.
    """
    if not safety_ok(meta):
        return 0
    if (meta.get("label") or "").lower() == "rejected":
        return 0
    if not quality_ok:
        return 0
    if not label_ok(meta.get("label")):
        return 0
    if exec_verify_ok(meta, bootstrap=bootstrap):
        return 1
    return 2
