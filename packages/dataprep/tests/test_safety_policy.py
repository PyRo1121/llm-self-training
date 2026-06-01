"""Safety policy: allowlist, severity classification, apply_policy, Presidio filter."""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm_dataprep.filters import SafetyFinding
from llm_dataprep.safety_policy import (
    SafetyPolicy,
    Severity,
    apply_policy,
    classify_finding,
    filter_presidio_results,
    is_allowlisted_in_text,
    is_diff_harness,
)


def _policy(**overrides: object) -> SafetyPolicy:
    base = dict(
        presidio_min_score={"EMAIL_ADDRESS": 0.55, "PERSON": 0.92, "PHONE_NUMBER": 0.65},
        presidio_entities=frozenset({"EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"}),
        quarantine_severity=Severity.BLOCK,
        diff_harnesses=frozenset({"git", "git-diffs"}),
        gitleaks_severity=Severity.BLOCK,
        presidio_block_entities=frozenset({"EMAIL_ADDRESS", "PHONE_NUMBER"}),
        exact_allowlist=frozenset({"sk-test123456789012345678901234567890", "user@example.com"}),
        allowlist_regex=(re.compile(r"(?i)example"), re.compile(r"^[a-f0-9]{40}$")),
    )
    base.update(overrides)
    return SafetyPolicy(**base)  # type: ignore[arg-type]


@dataclass
class _PresidioResult:
    entity_type: str
    score: float
    start: int
    end: int


def test_exact_allowlist_drops_finding() -> None:
    pol = _policy()
    finding = SafetyFinding(
        source="regex",
        kind="openai_key",
        detail="sk-test123456789012345678901234567890",
        start=0,
        end=40,
    )
    assert is_allowlisted_in_text(finding, finding.detail, pol)
    block, warn = apply_policy([finding], finding.detail, policy=pol)
    assert block == []
    assert warn == []


def test_regex_allowlist_drops_finding() -> None:
    pol = _policy()
    sha = "a" * 40
    finding = SafetyFinding(
        source="regex",
        kind="generic_api_key",
        detail=sha,
        start=0,
        end=40,
    )
    assert is_allowlisted_in_text(finding, sha, pol)
    block, warn = apply_policy([finding], sha, policy=pol)
    assert block == []
    assert warn == []


def test_allowlist_matches_text_span() -> None:
    pol = _policy()
    text = "contact user@example.com for help"
    finding = SafetyFinding(
        source="presidio",
        kind="EMAIL_ADDRESS",
        detail="score=0.99",
        start=8,
        end=24,
    )
    assert is_allowlisted_in_text(finding, text, pol)
    block, warn = apply_policy([finding], text, policy=pol)
    assert block == []
    assert warn == []


def test_block_regex_severity() -> None:
    pol = _policy(exact_allowlist=frozenset())
    finding = SafetyFinding(
        source="regex",
        kind="aws_access_key",
        detail="AKIA1234567890ABCDEF",
    )
    assert classify_finding(finding, pol) == Severity.BLOCK
    block, warn = apply_policy([finding], finding.detail, policy=pol)
    assert len(block) == 1
    assert warn == []


def test_warn_regex_severity() -> None:
    pol = _policy(exact_allowlist=frozenset())
    finding = SafetyFinding(
        source="regex",
        kind="generic_api_key",
        detail="api_key=abcdefghijklmnop",
    )
    assert classify_finding(finding, pol) == Severity.WARN
    block, warn = apply_policy([finding], finding.detail, policy=pol)
    assert block == []
    assert len(warn) == 1


def test_presidio_block_vs_warn_entities() -> None:
    pol = _policy(exact_allowlist=frozenset())
    email = SafetyFinding(source="presidio", kind="EMAIL_ADDRESS", detail="score=0.80")
    person = SafetyFinding(source="presidio", kind="PERSON", detail="score=0.95")
    assert classify_finding(email, pol) == Severity.BLOCK
    assert classify_finding(person, pol) == Severity.WARN
    block, warn = apply_policy([email, person], "Jane Doe jane@test.com", policy=pol)
    assert [f.kind for f in block] == ["EMAIL_ADDRESS"]
    assert [f.kind for f in warn] == ["PERSON"]


def test_presidio_below_min_score_dropped() -> None:
    pol = _policy(exact_allowlist=frozenset())
    finding = SafetyFinding(source="presidio", kind="EMAIL_ADDRESS", detail="score=0.40")
    assert classify_finding(finding, pol) is None
    block, warn = apply_policy([finding], "x@y.com", policy=pol)
    assert block == []
    assert warn == []


def test_presidio_missing_score_dropped() -> None:
    pol = _policy(exact_allowlist=frozenset())
    finding = SafetyFinding(source="presidio", kind="EMAIL_ADDRESS", detail="entity hit")
    assert classify_finding(finding, pol) is None
    block, warn = apply_policy([finding], "x@y.com", policy=pol)
    assert block == []
    assert warn == []


def test_gitleaks_severity_from_policy() -> None:
    pol = _policy(gitleaks_severity=Severity.WARN, exact_allowlist=frozenset())
    finding = SafetyFinding(source="gitleaks", kind="generic-api-key", detail="match: secret")
    assert classify_finding(finding, pol) == Severity.WARN
    block, warn = apply_policy([finding], "secret", policy=pol)
    assert block == []
    assert len(warn) == 1


def test_filter_presidio_results_matches_safety_finding() -> None:
    pol = _policy(exact_allowlist=frozenset())
    results = [
        _PresidioResult("EMAIL_ADDRESS", 0.80, 0, 12),
        _PresidioResult("EMAIL_ADDRESS", 0.40, 20, 32),
        _PresidioResult("NRP", 0.99, 40, 50),
        _PresidioResult("PERSON", 0.95, 60, 64),
    ]
    findings = filter_presidio_results(results, policy=pol)
    assert len(findings) == 2
    assert all(isinstance(f, SafetyFinding) for f in findings)
    assert findings[0] == SafetyFinding(
        source="presidio",
        kind="EMAIL_ADDRESS",
        detail="score=0.80",
        start=0,
        end=12,
    )
    assert findings[1].kind == "PERSON"
    assert findings[1].detail == "score=0.95"


def test_is_diff_harness_exported_and_matches_record() -> None:
    pol = _policy()
    assert is_diff_harness({"harness": "git-diffs"}, pol)
    assert is_diff_harness({"source": "git"}, pol)
    assert is_diff_harness({"source_path": "/data/raw/git-diffs/foo.jsonl"}, pol)
    assert not is_diff_harness({"harness": "cursor"}, pol)


def test_is_diff_harness_github_not_prefix_match() -> None:
    pol = _policy()
    assert not is_diff_harness({"harness": "github"}, pol)
    assert not is_diff_harness({"source": "github_public"}, pol)
    assert not is_diff_harness({"source_path": "github:org/repo/path.jsonl"}, pol)
