"""Safety scan for ingest/curate rows: regex, optional gitleaks CLI, optional Presidio PII.

Install:
  uv sync --package llm-dataprep --extra safety
  python -m spacy download en_core_web_sm   # Presidio default model
  pacman -S gitleaks                         # or brew install gitleaks
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# High-signal patterns (PLAN: regex + gitleaks + Presidio)
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9_]{20,}")),
    ("github_fine_pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("generic_api_key", re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
)


@dataclass(frozen=True)
class SafetyFinding:
    source: str  # regex | gitleaks | presidio
    kind: str
    detail: str
    start: int | None = None
    end: int | None = None


@dataclass
class SafetyReport:
    ok: bool
    findings: list[SafetyFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "findings": [
                {
                    "source": f.source,
                    "kind": f.kind,
                    "detail": f.detail,
                    "start": f.start,
                    "end": f.end,
                }
                for f in self.findings
            ],
        }


def scan_regex(text: str) -> list[SafetyFinding]:
    findings: list[SafetyFinding] = []
    for kind, pattern in _SECRET_PATTERNS:
        for m in pattern.finditer(text):
            snippet = m.group(0)[:80]
            findings.append(
                SafetyFinding(
                    source="regex",
                    kind=kind,
                    detail=snippet,
                    start=m.start(),
                    end=m.end(),
                )
            )
    return findings


def scan_gitleaks_dir(
    directory: Path,
    *,
    timeout_s: float = 300.0,
) -> list[SafetyFinding]:
    """One gitleaks dir pass over a folder (v8: --report-format json)."""
    exe = shutil.which("gitleaks")
    if not exe or not directory.is_dir():
        return []

    report = directory / "gitleaks-report.json"
    cmd = [
        exe,
        "dir",
        str(directory),
        "--report-path",
        str(report),
        "--report-format",
        "json",
        "--exit-code",
        "0",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return [
            SafetyFinding(
                source="gitleaks",
                kind="scan_error",
                detail="gitleaks dir failed or timed out",
            )
        ]
    if not report.is_file():
        return []
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else data.get("findings") or []
    findings: list[SafetyFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rule = item.get("RuleID") or item.get("rule") or "secret"
        match = item.get("Match") or item.get("match") or ""
        fname = item.get("File") or ""
        findings.append(
            SafetyFinding(
                source="gitleaks",
                kind=str(rule),
                detail=f"{fname}: {str(match)[:160]}",
            )
        )
    return findings


def gitleaks_line_flags(
    path: Path,
    records: Iterator[tuple[int, dict[str, Any]]],
    *,
    timeout_s: float = 300.0,
    max_rows: int | None = None,
) -> dict[int, list[SafetyFinding]]:
    """Per-file gitleaks: one dir scan; map findings back to JSONL line_no via row_<n>.txt."""
    exe = shutil.which("gitleaks")
    if not exe:
        return {}

    flags: dict[int, list[SafetyFinding]] = {}
    with tempfile.TemporaryDirectory(prefix="gitleaks-rows-") as tmp:
        root = Path(tmp)
        count = 0
        for line_no, record in records:
            if max_rows is not None and count >= max_rows:
                break
            text = record.get("text") or record.get("content") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            (root / f"row_{line_no}.txt").write_text(text[:500_000], encoding="utf-8")
            count += 1
        if count == 0:
            return flags

        for finding in scan_gitleaks_dir(root, timeout_s=timeout_s):
            detail = finding.detail
            fname = detail.split(":", 1)[0].strip() if ":" in detail else ""
            m = re.search(r"row_(\d+)\.txt", fname)
            if not m:
                continue
            line_no = int(m.group(1))
            flags.setdefault(line_no, []).append(finding)

    return flags


def scan_gitleaks(text: str, *, timeout_s: float = 120.0) -> list[SafetyFinding]:
    exe = shutil.which("gitleaks")
    if not exe:
        return []

    findings: list[SafetyFinding] = []
    with tempfile.TemporaryDirectory(prefix="gitleaks-scan-") as tmp:
        path = Path(tmp) / "content.txt"
        path.write_text(text, encoding="utf-8")
        report = Path(tmp) / "report.json"
        cmd = [
            exe,
            "dir",
            str(path),
            "--report-path",
            str(report),
            "--report-format",
            "json",
            "--exit-code",
            "0",
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return [
                SafetyFinding(
                    source="gitleaks",
                    kind="scan_error",
                    detail="gitleaks subprocess failed or timed out",
                )
            ]

        if not report.is_file():
            return findings

        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return findings

        items = data if isinstance(data, list) else data.get("findings") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            rule = item.get("RuleID") or item.get("rule") or "secret"
            match = item.get("Match") or item.get("match") or ""
            findings.append(
                SafetyFinding(
                    source="gitleaks",
                    kind=str(rule),
                    detail=str(match)[:200],
                )
            )
    return findings


_analyzer: Any | None = None
_analyzer_failed = False


def scan_presidio(text: str, *, language: str = "en") -> list[SafetyFinding]:
    global _analyzer, _analyzer_failed
    if _analyzer_failed:
        return []
    if _analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine

            _analyzer = AnalyzerEngine()
        except Exception:
            _analyzer_failed = True
            return []

    try:
        results = _analyzer.analyze(text=text, language=language)
    except Exception:
        return [
            SafetyFinding(
                source="presidio",
                kind="scan_error",
                detail="Presidio analyze() failed",
            )
        ]

    findings: list[SafetyFinding] = []
    for r in results:
        findings.append(
            SafetyFinding(
                source="presidio",
                kind=str(r.entity_type),
                detail=f"score={r.score:.2f}",
                start=r.start,
                end=r.end,
            )
        )
    return findings


def scan_text(
    text: str,
    *,
    use_regex: bool = True,
    use_gitleaks: bool = True,
    use_presidio: bool = True,
) -> SafetyReport:
    """Scan a single text blob (e.g. one message or concatenated row)."""
    if not text or not text.strip():
        return SafetyReport(ok=True)

    findings: list[SafetyFinding] = []
    if use_regex:
        findings.extend(scan_regex(text))
    if use_gitleaks:
        findings.extend(scan_gitleaks(text))
    if use_presidio:
        findings.extend(scan_presidio(text))

    return SafetyReport(ok=len(findings) == 0, findings=findings)


def scan_record_text_fields(
    record: dict[str, Any],
    *,
    use_regex: bool = True,
    use_gitleaks: bool = True,
    use_presidio: bool = True,
) -> SafetyReport:
    """Scan common ingest record fields (`text`, `content`, nested messages)."""
    parts: list[str] = []
    for key in ("text", "content", "message"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                for key in ("content", "text"):
                    val = msg.get(key)
                    if isinstance(val, str) and val.strip():
                        parts.append(val)
    combined = "\n\n".join(parts)
    return scan_text(
        combined,
        use_regex=use_regex,
        use_gitleaks=use_gitleaks,
        use_presidio=use_presidio,
    )
