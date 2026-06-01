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

# High-signal patterns (regex + gitleaks + Presidio)
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9_]{20,}")),
    ("github_fine_pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("stripe_key", re.compile(r"sk_(live|test)_[A-Za-z0-9]{20,}")),
    ("azure_key", re.compile(r"(?i)(DefaultEndpointsProtocol=|AccountKey=)[A-Za-z0-9+/=]{20,}")),
    ("npm_token", re.compile(r"npm_[A-Za-z0-9]{20,}")),
    ("cursor_token", re.compile(r"cursor_[A-Za-z0-9_\-]{20,}")),
    ("turso_token", re.compile(r"(?i)libsql://[^\s\"']+|turso_[A-Za-z0-9_]{16,}")),
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


def _gitleaks_scratch_dir() -> Path:
    """Disk-backed scratch (avoid 16GB tmpfs exhaustion on large JSONL lakes)."""
    from llm_core import data_dir

    d = data_dir() / ".gitleaks-scratch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_gitleaks_report(report: Path) -> list[SafetyFinding]:
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
        fname = item.get("File") or item.get("file") or ""
        line = item.get("StartLine") or item.get("startLine") or item.get("Line")
        detail = f"{fname}:{line}: {str(match)[:160]}" if line else f"{fname}: {str(match)[:160]}"
        findings.append(
            SafetyFinding(
                source="gitleaks",
                kind=str(rule),
                detail=detail,
            )
        )
    return findings


def gitleaks_sidecar_line_flags(
    rows: list[tuple[int, str]],
    source_path: Path,
    *,
    timeout_s: float = 600.0,
    max_target_mb: int = 20_000,
) -> dict[int, list[SafetyFinding]]:
    """Write one line per JSONL row (denormalized text) for accurate gitleaks line mapping."""
    exe = shutil.which("gitleaks")
    if not exe or not rows:
        return {}

    line_map: dict[int, int] = {}  # sidecar line (1-based) -> jsonl line_no
    flags: dict[int, list[SafetyFinding]] = {}
    with tempfile.TemporaryDirectory(prefix="gitleaks-sidecar-", dir=_gitleaks_scratch_dir()) as tmp:
        sidecar = Path(tmp) / f"{source_path.stem}.txt"
        with sidecar.open("w", encoding="utf-8") as fh:
            for i, (line_no, text) in enumerate(rows, start=1):
                line_map[i] = line_no
                flat = text.replace("\n", " ").replace("\r", " ")[:500_000]
                fh.write(flat + "\n")

        report = Path(tmp) / "gitleaks-report.json"
        cmd = [
            exe,
            "dir",
            str(sidecar.resolve()),
            "--report-path",
            str(report),
            "--report-format",
            "json",
            "--exit-code",
            "0",
            "--max-target-megabytes",
            str(max_target_mb),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        except (subprocess.TimeoutExpired, OSError):
            return flags

        if not report.is_file():
            return flags
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return flags
        items = data if isinstance(data, list) else data.get("findings") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            line = item.get("StartLine") or item.get("startLine") or item.get("Line")
            if line is None:
                continue
            try:
                sidecar_line = int(line)
            except (TypeError, ValueError):
                continue
            jsonl_line = line_map.get(sidecar_line)
            if jsonl_line is None:
                continue
            rule = item.get("RuleID") or item.get("rule") or "secret"
            match = item.get("Match") or item.get("match") or ""
            finding = SafetyFinding(
                source="gitleaks",
                kind=str(rule),
                detail=f"{source_path.name}:{jsonl_line}: {str(match)[:160]}",
            )
            flags.setdefault(jsonl_line, []).append(finding)
    return flags


def gitleaks_jsonl_line_flags(
    path: Path,
    *,
    timeout_s: float = 600.0,
    max_target_mb: int = 20_000,
) -> dict[int, list[SafetyFinding]]:
    """Scan JSONL in place (gitleaks dir supports files); map StartLine → JSONL line_no."""
    exe = shutil.which("gitleaks")
    if not exe or not path.is_file():
        return {}

    flags: dict[int, list[SafetyFinding]] = {}
    with tempfile.TemporaryDirectory(prefix="gitleaks-scan-", dir=_gitleaks_scratch_dir()) as tmp:
        report = Path(tmp) / "gitleaks-report.json"
        cmd = [
            exe,
            "dir",
            str(path.resolve()),
            "--report-path",
            str(report),
            "--report-format",
            "json",
            "--exit-code",
            "0",
            "--max-target-megabytes",
            str(max_target_mb),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        except (subprocess.TimeoutExpired, OSError):
            return {}

        if not report.is_file():
            return flags
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return flags
        items = data if isinstance(data, list) else data.get("findings") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            line = item.get("StartLine") or item.get("startLine") or item.get("Line")
            if line is None:
                continue
            try:
                line_no = int(line)
            except (TypeError, ValueError):
                continue
            rule = item.get("RuleID") or item.get("rule") or "secret"
            match = item.get("Match") or item.get("match") or ""
            fname = item.get("File") or item.get("file") or path.name
            finding = SafetyFinding(
                source="gitleaks",
                kind=str(rule),
                detail=f"{fname}:{line_no}: {str(match)[:160]}",
            )
            flags.setdefault(line_no, []).append(finding)

    return flags


def gitleaks_line_flags(
    path: Path,
    records: Iterator[tuple[int, dict[str, Any]]],
    *,
    timeout_s: float = 300.0,
    max_rows: int | None = None,
) -> dict[int, list[SafetyFinding]]:
    """Legacy row-split scan — only for small extracts; large JSONL uses in-place scan."""
    _ = records, max_rows
    return gitleaks_jsonl_line_flags(path, timeout_s=timeout_s)


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
            from llm_dataprep.presidio_custom import create_analyzer_engine

            _analyzer = create_analyzer_engine()
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

    return _presidio_results_to_findings(results)


def _presidio_results_to_findings(results: Any) -> list[SafetyFinding]:
    from llm_dataprep.safety_policy import load_safety_policy

    pol = load_safety_policy()
    findings: list[SafetyFinding] = []
    for r in results or []:
        entity = str(r.entity_type)
        if entity not in pol.presidio_entities:
            continue
        score = float(r.score)
        min_score = pol.presidio_min_score.get(entity, 0.9)
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


_batch_engine: Any | None = None
_batch_engine_failed = False


def scan_presidio_batch(
    texts: list[str],
    *,
    language: str = "en",
    n_process: int | None = None,
    batch_size: int | None = None,
) -> list[list[SafetyFinding]]:
    """Batch PII scan (Presidio BatchAnalyzerEngine + spaCy n_process)."""
    import os

    from llm_dataprep.perf import presidio_batch_size, presidio_n_process

    if not texts:
        return []
    n_proc = presidio_n_process() if n_process is None else max(1, n_process)
    # File-parallel scan workers must not spawn nested spaCy multiprocessing.
    if os.environ.get("LLM_SCAN_SUBPROCESS") == "1":
        n_proc = 1
    bsize = presidio_batch_size() if batch_size is None else max(1, batch_size)

    global _batch_engine, _batch_engine_failed
    if _batch_engine_failed:
        return [scan_presidio(t, language=language) for t in texts]

    if _batch_engine is None:
        try:
            from presidio_analyzer import BatchAnalyzerEngine

            from llm_dataprep.presidio_custom import create_analyzer_engine

            _batch_engine = BatchAnalyzerEngine(analyzer_engine=create_analyzer_engine())
        except Exception:
            _batch_engine_failed = True
            return [scan_presidio(t, language=language) for t in texts]

    try:
        raw = _batch_engine.analyze_iterator(
            texts,
            language=language,
            n_process=n_proc,
            batch_size=bsize,
        )
        return [_presidio_results_to_findings(r) for r in raw]
    except Exception:
        return [scan_presidio(t, language=language) for t in texts]


def _safety_report_after_policy(findings: list[SafetyFinding], text: str) -> SafetyReport:
    from llm_dataprep.safety_policy import apply_policy, should_quarantine

    block, warn = apply_policy(findings, text)
    kept = block + warn
    ok = not should_quarantine(block, warn)
    return SafetyReport(ok=ok, findings=kept)


def record_combined_text(record: dict[str, Any]) -> str:
    """Flatten common ingest fields for safety scanning."""
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
    return "\n\n".join(parts)


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

    return _safety_report_after_policy(findings, text)


def scan_record_text_fields(
    record: dict[str, Any],
    *,
    use_regex: bool = True,
    use_gitleaks: bool = True,
    use_presidio: bool = True,
    presidio_findings: list[SafetyFinding] | None = None,
) -> SafetyReport:
    """Scan common ingest record fields (`text`, `content`, nested messages)."""
    combined = record_combined_text(record)
    if not combined.strip():
        return SafetyReport(ok=True)

    findings: list[SafetyFinding] = []
    if use_regex:
        findings.extend(scan_regex(combined))
    if use_gitleaks:
        findings.extend(scan_gitleaks(combined))
    if use_presidio:
        if presidio_findings is not None:
            findings.extend(presidio_findings)
        else:
            findings.extend(scan_presidio(combined))
    return _safety_report_after_policy(findings, combined)
