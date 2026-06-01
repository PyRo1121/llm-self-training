"""Configurable safety policy: allowlist, Presidio thresholds, block vs warn."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from llm_core import config_dir
from llm_dataprep.filters import SafetyFinding

__all__ = [
    "Severity",
    "SafetyPolicy",
    "load_safety_policy",
    "safety_policy_version",
    "is_allowlisted_snippet",
    "is_allowlisted_in_text",
    "classify_finding",
    "apply_policy",
    "filter_presidio_results",
    "should_quarantine",
    "is_diff_harness",
    "findings_to_dicts",
]


class Severity(str, Enum):
    BLOCK = "block"
    WARN = "warn"


_DEFAULT_PRESIDIO_MIN: dict[str, float] = {
    "EMAIL_ADDRESS": 0.55,
    "PHONE_NUMBER": 0.65,
    "US_SSN": 0.75,
    "CREDIT_CARD": 0.75,
    "US_BANK_NUMBER": 0.75,
    "IBAN_CODE": 0.75,
    "IP_ADDRESS": 0.85,
    "PERSON": 0.92,
    "LOCATION": 0.95,
    "DATE_TIME": 0.95,
    "NRP": 0.95,
    "HF_TOKEN": 0.85,
    "CURSOR_TOKEN": 0.85,
    "TURSO_TOKEN": 0.85,
}

_DEFAULT_PRESIDIO_ENTITIES: frozenset[str] = frozenset(
    {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "US_BANK_NUMBER",
        "IBAN_CODE",
        "IP_ADDRESS",
        "US_DRIVER_LICENSE",
        "US_PASSPORT",
        "CRYPTO",
        "MEDICAL_LICENSE",
        "HF_TOKEN",
        "CURSOR_TOKEN",
        "TURSO_TOKEN",
    }
)

_BLOCK_REGEX_KINDS: frozenset[str] = frozenset(
    {
        "aws_access_key",
        "github_pat",
        "github_fine_pat",
        "openai_key",
        "slack_token",
        "private_key_block",
        "hf_token",
        "anthropic_key",
        "stripe_key",
        "azure_key",
        "npm_token",
        "cursor_token",
        "turso_token",
        "password_assignment",
    }
)

_WARN_REGEX_KINDS: frozenset[str] = frozenset({"generic_api_key"})


@dataclass(frozen=True)
class SafetyPolicy:
    presidio_min_score: dict[str, float]
    presidio_entities: frozenset[str]
    quarantine_severity: Severity
    diff_harnesses: frozenset[str]
    gitleaks_severity: Severity
    presidio_block_entities: frozenset[str]
    exact_allowlist: frozenset[str]
    allowlist_regex: tuple[re.Pattern[str], ...]


@lru_cache(maxsize=1)
def safety_policy_version() -> str:
    doc = _load_merged_safety_yaml()
    raw = doc.get("version")
    return str(raw) if raw is not None else "1"


@lru_cache(maxsize=1)
def load_safety_policy() -> SafetyPolicy:
    doc = _load_merged_safety_yaml()
    presidio = doc.get("presidio") or {}
    min_score = {**_DEFAULT_PRESIDIO_MIN, **(presidio.get("min_score") or {})}
    entities_raw = presidio.get("entities")
    entities = (
        frozenset(str(e) for e in entities_raw)
        if entities_raw
        else _DEFAULT_PRESIDIO_ENTITIES
    )
    block_entities_raw = presidio.get("block_entities")
    block_entities = (
        frozenset(str(e) for e in block_entities_raw)
        if block_entities_raw
        else frozenset(
            {
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "US_SSN",
                "CREDIT_CARD",
                "US_BANK_NUMBER",
                "IBAN_CODE",
                "US_DRIVER_LICENSE",
                "US_PASSPORT",
                "CRYPTO",
                "HF_TOKEN",
                "CURSOR_TOKEN",
                "TURSO_TOKEN",
            }
        )
    )
    diff_h = doc.get("diff_harnesses") or ["git", "git-diffs"]
    allow = _load_allowlist_patterns(doc)
    sev = str(doc.get("quarantine_severity", "block")).lower()
    quarantine = Severity.WARN if sev == "warn" else Severity.BLOCK
    g_sev = str(doc.get("gitleaks_severity", "block")).lower()
    gitleaks_sev = Severity.WARN if g_sev == "warn" else Severity.BLOCK
    return SafetyPolicy(
        presidio_min_score=min_score,
        presidio_entities=entities,
        quarantine_severity=quarantine,
        diff_harnesses=frozenset(str(h).lower() for h in diff_h),
        gitleaks_severity=gitleaks_sev,
        presidio_block_entities=block_entities,
        exact_allowlist=allow[0],
        allowlist_regex=allow[1],
    )


def _load_merged_safety_yaml() -> dict[str, Any]:
    cfg_path = config_dir() / "default.yaml"
    doc: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as fh:
            root = yaml.safe_load(fh) or {}
        doc = root.get("safety") or {}
    allow_path = config_dir() / "safety-allowlist.yaml"
    if allow_path.is_file():
        with allow_path.open(encoding="utf-8") as fh:
            allow_doc = yaml.safe_load(fh) or {}
        doc = {**doc, "allowlist": allow_doc}
    return doc


def _load_allowlist_patterns(doc: dict[str, Any]) -> tuple[frozenset[str], tuple[re.Pattern[str], ...]]:
    allow = doc.get("allowlist") or {}
    exact = frozenset(str(x) for x in (allow.get("exact") or []) if x)
    patterns: list[re.Pattern[str]] = []
    for raw in allow.get("regex") or []:
        if not raw:
            continue
        try:
            patterns.append(re.compile(str(raw)))
        except re.error:
            continue
    return exact, tuple(patterns)


def is_allowlisted_snippet(snippet: str, policy: SafetyPolicy | None = None) -> bool:
    pol = policy or load_safety_policy()
    if snippet in pol.exact_allowlist:
        return True
    for pat in pol.allowlist_regex:
        if pat.fullmatch(snippet) or pat.search(snippet):
            return True
    return False


def is_allowlisted_in_text(finding: SafetyFinding, text: str, policy: SafetyPolicy | None = None) -> bool:
    pol = policy or load_safety_policy()
    detail = finding.detail or ""
    if is_allowlisted_snippet(detail, pol):
        return True
    if finding.start is not None and finding.end is not None and text:
        span = text[finding.start : finding.end]
        if is_allowlisted_snippet(span, pol):
            return True
    # gitleaks detail often embeds match after colon
    for part in detail.split(":"):
        part = part.strip()
        if len(part) >= 8 and is_allowlisted_snippet(part, pol):
            return True
    return False


def classify_finding(finding: SafetyFinding, policy: SafetyPolicy | None = None) -> Severity | None:
    """Return severity or None if finding should be dropped (allowlisted / below threshold)."""
    pol = policy or load_safety_policy()
    src = finding.source
    kind = finding.kind

    if src == "gitleaks":
        if finding.kind == "scan_error":
            return Severity.WARN
        return pol.gitleaks_severity

    if src == "regex":
        if kind in _BLOCK_REGEX_KINDS:
            return Severity.BLOCK
        if kind in _WARN_REGEX_KINDS:
            return Severity.WARN
        return Severity.WARN

    if src == "presidio":
        if kind not in pol.presidio_entities:
            return None
        # detail like score=0.85
        score = _presidio_score_from_detail(finding.detail)
        min_score = pol.presidio_min_score.get(kind, pol.presidio_min_score.get("PERSON", 0.9))
        if score is None or score < min_score:
            return None
        if kind in pol.presidio_block_entities:
            return Severity.BLOCK
        return Severity.WARN

    if src == "json":
        return Severity.BLOCK
    return Severity.WARN


def _presidio_score_from_detail(detail: str) -> float | None:
    m = re.search(r"score=([0-9.]+)", detail or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def apply_policy(
    findings: list[SafetyFinding],
    text: str,
    *,
    policy: SafetyPolicy | None = None,
) -> tuple[list[SafetyFinding], list[SafetyFinding]]:
    """Filter allowlisted / below-threshold; return (block, warn)."""
    pol = policy or load_safety_policy()
    block: list[SafetyFinding] = []
    warn: list[SafetyFinding] = []
    for f in findings:
        if is_allowlisted_in_text(f, text, pol):
            continue
        sev = classify_finding(f, pol)
        if sev is None:
            continue
        if sev == Severity.BLOCK:
            block.append(f)
        else:
            warn.append(f)
    return block, warn


def filter_presidio_results(
    results: Any,
    *,
    policy: SafetyPolicy | None = None,
) -> list[SafetyFinding]:
    """Map Presidio RecognizerResult objects to SafetyFinding, filtered by policy."""
    pol = policy or load_safety_policy()
    findings: list[SafetyFinding] = []
    for r in results or []:
        entity = str(r.entity_type)
        if entity not in pol.presidio_entities:
            continue
        score = float(r.score)
        min_score = pol.presidio_min_score.get(
            entity, pol.presidio_min_score.get("PERSON", 0.9)
        )
        if score < min_score:
            continue
        findings.append(
            SafetyFinding(
                source="presidio",
                kind=entity,
                detail=f"score={score:.2f}",
                start=r.start,
                end=r.end,
            )
        )
    return findings


def should_quarantine(block: list[SafetyFinding], warn: list[SafetyFinding], policy: SafetyPolicy | None = None) -> bool:
    pol = policy or load_safety_policy()
    if pol.quarantine_severity == Severity.BLOCK:
        return len(block) > 0
    return len(block) > 0 or len(warn) > 0


def is_diff_harness(record: dict[str, Any], policy: SafetyPolicy | None = None) -> bool:
    pol = policy or load_safety_policy()
    harness = str(record.get("harness") or record.get("source") or "").lower()
    if harness in pol.diff_harnesses:
        return True
    sp = str(record.get("source_path") or "")
    return "git-diffs" in sp


def findings_to_dicts(findings: list[SafetyFinding], severities: list[Severity]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f, s in zip(findings, severities, strict=False):
        out.append(
            {
                "source": f.source,
                "kind": f.kind,
                "detail": f.detail,
                "severity": s.value,
                "start": f.start,
                "end": f.end,
            }
        )
    return out
