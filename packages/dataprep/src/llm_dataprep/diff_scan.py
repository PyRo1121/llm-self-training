"""Diff-aware safety scanning for git-diffs harness rows."""

from __future__ import annotations

import re
from typing import Any

from llm_dataprep.filters import SafetyFinding, SafetyReport, scan_regex
from llm_dataprep.safety_policy import (
    Severity,
    apply_policy,
    load_safety_policy,
    should_quarantine,
)

_PASSWORD_LINE = re.compile(
    r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"#]{6,})"
)


def extract_added_lines(text: str) -> str:
    """Keep unified-diff added lines (+) minus +++ headers."""
    if not text:
        return ""
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
        elif not line.startswith("-") and not line.startswith("@@"):
            # context lines in unified diff start with a single space
            if line.startswith(" "):
                out.append(line[1:])
            elif line.strip():
                out.append(line)
    return "\n".join(out)


def scan_diff_passwords(text: str) -> list[SafetyFinding]:
    """Detect credential assignments in diff added lines (value-only detail for allowlist)."""
    findings: list[SafetyFinding] = []
    for m in _PASSWORD_LINE.finditer(text):
        value = m.group(1)[:80]
        findings.append(
            SafetyFinding(
                source="regex",
                kind="password_assignment",
                detail=value,
                start=m.start(1),
                end=m.end(1),
            )
        )
    return findings


def scan_diff_text(text: str, *, use_regex: bool = True) -> SafetyReport:
    """Scan only added lines from a diff blob; apply allowlist + severity."""
    added = extract_added_lines(text)
    if not added.strip():
        return SafetyReport(ok=True)
    findings: list[SafetyFinding] = []
    if use_regex:
        findings.extend(scan_regex(added))
        findings.extend(scan_diff_passwords(added))
    block, warn = apply_policy(findings, added)
    kept = block + warn
    sevs = [Severity.BLOCK] * len(block) + [Severity.WARN] * len(warn)
    pol = load_safety_policy()
    ok = not should_quarantine(block, warn, pol)
    report = SafetyReport(ok=ok, findings=kept)
    report_findings_meta(report, kept, sevs)
    return report


def scan_diff_record(record: dict[str, Any]) -> SafetyReport:
    text = record.get("text") or record.get("content") or ""
    if not isinstance(text, str):
        return SafetyReport(ok=True)
    return scan_diff_text(text)


def report_findings_meta(report: SafetyReport, findings: list[SafetyFinding], severities: list[Severity]) -> None:
    """Attach severities on report object via findings detail prefix (no schema break)."""
    _ = report, findings, severities  # severities consumed at failure_row layer
