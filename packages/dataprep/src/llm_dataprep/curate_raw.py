"""Build curated JSONL from data/raw ingest rows (session-grouped chat)."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

from llm_core import data_dir
from llm_core.yaml_config import load_yaml_config
from llm_dataprep.filters import SafetyFinding, SafetyReport, scan_presidio_batch, scan_regex
from llm_dataprep.perf import presidio_n_process, presidio_session_batch, worker_count
from llm_dataprep.scan_config import PresidioMode, resolve_curate_presidio_mode
from llm_dataprep.safety_policy import (
    Severity,
    apply_policy,
    findings_to_dicts,
    is_diff_harness,
    load_safety_policy,
    should_quarantine,
)
from llm_dataprep.safety_quarantine import load_safety_failure_keys, session_has_quarantined_row
from llm_dataprep.style_tags import enrich_meta
from llm_dataprep.tier1 import assign_train_tier


def _curation_for_session(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge global curation with personal / per-harness overrides."""
    first = rows[0]
    harness = str(first.get("harness") or first.get("source") or "").lower()
    merged = {k: v for k, v in cfg.items() if k not in ("personal", "by_harness")}
    src = first.get("source")
    is_public = src in ("public", "github_public") or first.get("data_source") == "public"
    if not is_public:
        personal = cfg.get("personal") or {}
        merged.update({k: v for k, v in personal.items() if v is not None})
    by_harness = cfg.get("by_harness") or {}
    if harness in by_harness:
        harness_cfg = by_harness[harness] or {}
        merged.update({k: v for k, v in harness_cfg.items() if v is not None})
    return merged


def _map_role(role: str | None, skip_roles: set[str]) -> str | None:
    if not role:
        return None
    r = role.lower()
    if r in skip_roles:
        return None
    if r in ("user", "human"):
        return "user"
    if r in ("assistant", "gpt", "model"):
        return "assistant"
    if r == "system":
        return "system"
    return None


def _infer_project(source_path: str | None, dataset_id: str | None = None) -> str:
    if dataset_id:
        return f"public:{dataset_id}"
    if not source_path:
        return ""
    if "LLM Self Training" in source_path or "llm-self-training" in source_path.lower():
        return "llm-self-training"
    return ""


def _public_meta_defaults(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    src = first.get("source")
    if src == "public":
        return {
            "label": first.get("label") or "accepted",
            "exec": first.get("exec") or "unknown",
            "verify": first.get("verify") or "unknown",
            "data_source": "public",
            "public_dataset": first.get("dataset_id"),
        }
    if src == "github_public":
        return {
            "label": "accepted",
            "exec": "unknown",
            "verify": "unknown",
            "data_source": "public",
            "public_dataset": "github_public",
        }
    return {}


def iter_raw_records(
    paths: list[Path],
    *,
    stats: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    if stats is not None:
                        stats["parse_errors"] = stats.get("parse_errors", 0) + 1
                    continue
                if isinstance(rec, dict):
                    rec["_source_file"] = str(path)
                    rec["_line_no"] = line_no
                    yield rec


def _row_dedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_path"),
        row.get("line_no"),
        row.get("role"),
        (row.get("text") or "")[:200],
    )


def build_session_messages(
    rows: list[dict[str, Any]],
    *,
    skip_roles: set[str],
    max_chars_per_message: int,
    min_message_chars: int,
) -> list[dict[str, str]]:
    rows = sorted(rows, key=lambda r: (r.get("source_path", ""), r.get("line_no", 0)))
    seen: set[tuple[Any, ...]] = set()
    messages: list[dict[str, str]] = []
    for row in rows:
        key = _row_dedup_key(row)
        if key in seen:
            continue
        seen.add(key)
        mapped = _map_role(row.get("role"), skip_roles)
        if not mapped:
            continue
        text = (row.get("text") or "").strip()
        if len(text) < min_message_chars:
            continue
        if len(text) > max_chars_per_message:
            text = text[:max_chars_per_message] + "\n…[truncated]"
        messages.append({"role": mapped, "content": text})
    return messages


def session_quality_ok(
    messages: list[dict[str, str]],
    *,
    min_messages: int,
    min_total_chars: int,
) -> bool:
    if len(messages) < min_messages:
        return False
    roles = {m["role"] for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return False
    total = sum(len(m["content"]) for m in messages)
    return total >= min_total_chars


def chunk_messages(
    messages: list[dict[str, str]],
    *,
    max_messages: int,
    overlap: int,
    min_messages: int,
    min_total_chars: int,
) -> list[list[dict[str, str]]]:
    """Split long sessions for seq-length limits and more tier-1 rows."""
    if len(messages) <= max_messages:
        return [messages]
    step = max(1, max_messages - max(0, overlap))
    chunks: list[list[dict[str, str]]] = []
    start = 0
    while start < len(messages):
        window = messages[start : start + max_messages]
        if session_quality_ok(
            window, min_messages=min_messages, min_total_chars=min_total_chars
        ):
            chunks.append(window)
        if start + max_messages >= len(messages):
            break
        start += step
    return chunks


def _session_windows(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[list[dict[str, str]]]:
    """Build quality-filtered message windows for one session."""
    cfg = _curation_for_session(rows, cfg)
    skip_roles = set(cfg.get("skip_roles") or ["developer"])
    max_chars = int(cfg.get("max_chars_per_message", 16_000))
    min_msg = int(cfg.get("min_message_chars", 40))
    min_messages = int(cfg.get("min_messages", 2))
    min_total = int(cfg.get("min_total_chars", 200))
    max_per_example = int(cfg.get("max_messages_per_example", 24))
    chunk_overlap = int(cfg.get("chunk_overlap_messages", 4))

    messages = build_session_messages(
        rows,
        skip_roles=skip_roles,
        max_chars_per_message=max_chars,
        min_message_chars=min_msg,
    )
    if not session_quality_ok(
        messages, min_messages=min_messages, min_total_chars=min_total
    ):
        return []

    return chunk_messages(
        messages,
        max_messages=max_per_example,
        overlap=chunk_overlap,
        min_messages=min_messages,
        min_total_chars=min_total,
    )


def _presidio_scan_text(rows: list[dict[str, Any]], combined: str) -> str:
    """Text fed to Presidio batch; diff harness scans added lines only."""
    if is_diff_harness(rows[0]):
        from llm_dataprep.diff_scan import extract_added_lines

        added = extract_added_lines(combined)
        return added if added.strip() else ""
    return combined


def _safety_for_window(
    rows: list[dict[str, Any]],
    combined: str,
    *,
    presidio_findings: list[SafetyFinding] | None = None,
    use_gitleaks: bool = False,
    filter_secrets: bool = True,
) -> tuple[SafetyReport, list[Severity]]:
    if not filter_secrets:
        return SafetyReport(ok=True), []

    policy_text = _presidio_scan_text(rows, combined)
    if is_diff_harness(rows[0]) and not policy_text.strip():
        return SafetyReport(ok=True), []

    regex_text = policy_text if is_diff_harness(rows[0]) else combined
    findings: list[SafetyFinding] = list(scan_regex(regex_text))
    if presidio_findings:
        findings.extend(presidio_findings)
    if use_gitleaks:
        from llm_dataprep.filters import scan_gitleaks

        findings.extend(scan_gitleaks(regex_text))

    block, warn = apply_policy(findings, policy_text or regex_text)
    kept = block + warn
    sevs: list[Severity] = [Severity.BLOCK] * len(block) + [Severity.WARN] * len(warn)
    ok = not should_quarantine(block, warn, load_safety_policy())
    return SafetyReport(ok=ok, findings=kept), sevs


def _finalize_curated_example(
    rows: list[dict[str, Any]],
    window: list[dict[str, str]],
    *,
    chunk_idx: int,
    chunk_count: int,
    cfg: dict[str, Any],
    safety: SafetyReport,
    safety_severities: list[Severity] | None = None,
) -> dict[str, Any]:
    first = rows[0]
    session_id = first.get("session_id")
    bootstrap = bool(cfg.get("bootstrap_mode", True))
    pub = _public_meta_defaults(rows)
    meta: dict[str, Any] = {
        "label": pub.get("label", "accepted"),
        "exec": pub.get("exec", "unknown"),
        "verify": pub.get("verify", "unknown"),
        "project": _infer_project(first.get("source_path"), first.get("dataset_id")),
        "harness": first.get("harness") or first.get("source"),
        "session_id": session_id,
        "chunk_index": chunk_idx,
        "chunk_count": chunk_count,
        "source_path": first.get("source_path"),
        "safety": (
            {"ok": safety.ok, "findings": findings_to_dicts(safety.findings, safety_severities)}
            if safety_severities is not None
            else safety.to_dict()
        ),
        "safety_ok": safety.ok,
    }
    if safety_severities is not None:
        meta["safety_severities"] = [s.value for s in safety_severities]
    if pub:
        meta.update(pub)
    meta["train_tier"] = assign_train_tier(meta, bootstrap=bootstrap, quality_ok=True)
    enrich_meta(meta, window)
    return {"messages": window, "meta": meta}


def curate_session(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    use_gitleaks: bool,
    presidio_mode: PresidioMode,
) -> list[dict[str, Any]]:
    cfg = _curation_for_session(rows, cfg)
    windows = _session_windows(rows, cfg)
    if not windows:
        return []

    filter_secrets = bool(cfg.get("filter_secrets_and_pii", True))
    use_presidio = presidio_mode != "off"
    out: list[dict[str, Any]] = []
    for chunk_idx, window in enumerate(windows):
        combined = "\n\n".join(m["content"] for m in window)
        presidio: list[SafetyFinding] | None = None
        if use_presidio and filter_secrets:
            from llm_dataprep.filters import scan_presidio

            scan_text = _presidio_scan_text(rows, combined)
            presidio = (
                scan_presidio(scan_text, mode=presidio_mode) if scan_text.strip() else []
            )
        safety, sevs = _safety_for_window(
            rows,
            combined,
            presidio_findings=presidio,
            use_gitleaks=use_gitleaks,
            filter_secrets=filter_secrets,
        )
        out.append(
            _finalize_curated_example(
                rows,
                window,
                chunk_idx=chunk_idx,
                chunk_count=len(windows),
                cfg=cfg,
                safety=safety,
                safety_severities=sevs,
            )
        )
    return out


def _curate_one_path(
    path: Path,
    *,
    cfg: dict[str, Any],
    failure_keys: set[tuple[str, int]],
    use_gitleaks: bool,
    presidio_mode: PresidioMode,
    tier: int,
    out_fh: TextIO,
) -> tuple[int, int, int, dict[int, int], int]:
    """Curate a single raw JSONL file into out_fh."""
    use_presidio = presidio_mode != "off"
    total_sessions = 0
    skipped_sessions = 0
    written = 0
    stack_index_rows = 0
    tier_counts: dict[int, int] = defaultdict(int)

    by_session: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    file_rows = 0
    load_stats: dict[str, int] = {}
    for rec in iter_raw_records([path], stats=load_stats):
        file_rows += 1
        if rec.get("stack_index"):
            stack_index_rows += 1
            continue
        sid = str(rec.get("session_id") or "unknown")
        harness = str(rec.get("harness") or rec.get("source") or "unknown")
        by_session[(harness, sid)].append(rec)

    pending: list[
        tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any], int, int]
    ] = []
    batch_size = presidio_session_batch()

    def _flush_pending() -> None:
        nonlocal written
        if not pending:
            return
        texts = [
            _presidio_scan_text(rows, "\n\n".join(m["content"] for m in window))
            for rows, window, _cfg, _ci, _cc in pending
        ]
        presidio_batches: list[list[SafetyFinding]] = []
        if use_presidio:
            batch_inputs = [t for t in texts if t.strip()]
            if batch_inputs:
                raw_batches = scan_presidio_batch(
                    batch_inputs,
                    n_process=presidio_n_process(),
                    batch_size=max(16, presidio_session_batch() // 2),
                    mode=presidio_mode,
                )
                it = iter(raw_batches)
                presidio_batches = [
                    next(it) if t.strip() else [] for t in texts
                ]
        for i, (rows, window, session_cfg, chunk_idx, chunk_count) in enumerate(pending):
            combined = "\n\n".join(m["content"] for m in window)
            presidio: list[SafetyFinding] | None = None
            filter_secrets = bool(session_cfg.get("filter_secrets_and_pii", True))
            if use_presidio and filter_secrets:
                if i < len(presidio_batches):
                    presidio = presidio_batches[i]
                else:
                    from llm_dataprep.filters import scan_presidio

                    scan_text = _presidio_scan_text(rows, combined)
                    presidio = scan_presidio(scan_text) if scan_text.strip() else []
            safety, sevs = _safety_for_window(
                rows,
                combined,
                presidio_findings=presidio,
                use_gitleaks=False,
                filter_secrets=filter_secrets,
            )
            curated = _finalize_curated_example(
                rows,
                window,
                chunk_idx=chunk_idx,
                chunk_count=chunk_count,
                cfg=session_cfg,
                safety=safety,
                safety_severities=sevs,
            )
            t = int(curated["meta"].get("train_tier", 0))
            tier_counts[t] += 1
            if t == tier:
                out_fh.write(json.dumps(curated, ensure_ascii=False) + "\n")
                written += 1
        pending.clear()

    for _key, rows in sorted(by_session.items()):
        total_sessions += 1
        if session_has_quarantined_row(rows, failure_keys):
            skipped_sessions += 1
            continue
        session_cfg = _curation_for_session(rows, cfg)
        if use_gitleaks:
            for curated in curate_session(
                rows,
                cfg,
                use_gitleaks=True,
                presidio_mode=presidio_mode,
            ):
                t = int(curated["meta"].get("train_tier", 0))
                tier_counts[t] += 1
                if t == tier:
                    out_fh.write(json.dumps(curated, ensure_ascii=False) + "\n")
                    written += 1
            continue

        windows = _session_windows(rows, session_cfg)
        if not windows:
            continue
        for chunk_idx, window in enumerate(windows):
            pending.append((rows, window, session_cfg, chunk_idx, len(windows)))
            if len(pending) >= batch_size:
                _flush_pending()
    _flush_pending()

    file_written = written
    parse_errors = load_stats.get("parse_errors", 0)
    if parse_errors:
        print(
            f"curate-raw: {path.name} — skipped {parse_errors} malformed JSONL line(s)",
            flush=True,
        )
    print(
        f"curate-raw: {path.name} — {file_rows} raw rows, "
        f"{len(by_session)} sessions → {file_written} tier-{tier} examples",
        flush=True,
    )
    return written, total_sessions, skipped_sessions, dict(tier_counts), stack_index_rows


def _curate_file_worker(payload: tuple[str, dict, list, bool, str, int, str]) -> dict[str, Any]:
    path_str, cfg, failure_list, use_gitleaks, presidio_mode, tier, out_part = payload
    failure_keys = set((str(a), int(b)) for a, b in failure_list)
    with Path(out_part).open("w", encoding="utf-8") as out_fh:
        written, total_sessions, skipped_sessions, tier_counts, stack_index_rows = _curate_one_path(
            Path(path_str),
            cfg=cfg,
            failure_keys=failure_keys,
            use_gitleaks=use_gitleaks,
            presidio_mode=presidio_mode,  # type: ignore[arg-type]
            tier=tier,
            out_fh=out_fh,
        )
    return {
        "path": path_str,
        "written": written,
        "total_sessions": total_sessions,
        "skipped_sessions": skipped_sessions,
        "tier_counts": tier_counts,
        "stack_index_rows": stack_index_rows,
        "part": out_part,
    }


def _curate_paths(
    paths: list[Path],
    *,
    cfg: dict[str, Any],
    failure_keys: set[tuple[str, int]],
    use_gitleaks: bool,
    presidio_mode: PresidioMode,
    tier: int,
    out_fh: Any,
    workers: int = 1,
) -> tuple[int, int, int, dict[int, int], int]:
    """Curate raw files; parallelize per file when workers > 1."""
    if workers <= 1 or len(paths) <= 1:
        total_written = 0
        total_sessions = 0
        skipped_sessions = 0
        stack_index_rows = 0
        tier_counts: dict[int, int] = defaultdict(int)
        for path in paths:
            w, ts, ss, tc, si = _curate_one_path(
                path,
                cfg=cfg,
                failure_keys=failure_keys,
                use_gitleaks=use_gitleaks,
                presidio_mode=presidio_mode,
                tier=tier,
                out_fh=out_fh,
            )
            total_written += w
            total_sessions += ts
            skipped_sessions += ss
            stack_index_rows += si
            for k, v in tc.items():
                tier_counts[k] += v
        return total_written, total_sessions, skipped_sessions, tier_counts, stack_index_rows

    print(f"curate-raw: {len(paths)} file(s), {workers} worker(s)", flush=True)
    failure_list = list(failure_keys)
    parts: list[Path] = []
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="curate-parts-") as tmp:
        tmp_path = Path(tmp)
        payloads = []
        for i, path in enumerate(paths):
            part = tmp_path / f"part_{i:04d}.jsonl"
            parts.append(part)
            payloads.append(
                (
                    str(path.resolve()),
                    cfg,
                    failure_list,
                    use_gitleaks,
                    presidio_mode,
                    tier,
                    str(part),
                )
            )
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            futures = [pool.submit(_curate_file_worker, p) for p in payloads]
            for fut in as_completed(futures):
                results.append(fut.result())

        order = {str(p.resolve()): i for i, p in enumerate(paths)}
        results.sort(key=lambda r: order.get(r["path"], 0))
        for part in parts:
            if part.is_file():
                out_fh.write(part.read_text(encoding="utf-8"))

    total_written = sum(int(r["written"]) for r in results)
    total_sessions = sum(int(r["total_sessions"]) for r in results)
    skipped_sessions = sum(int(r["skipped_sessions"]) for r in results)
    stack_index_rows = sum(int(r["stack_index_rows"]) for r in results)
    tier_counts: dict[int, int] = defaultdict(int)
    for r in results:
        for k, v in r["tier_counts"].items():
            tier_counts[int(k)] += int(v)
    return total_written, total_sessions, skipped_sessions, tier_counts, stack_index_rows


_DATED_STEM = re.compile(r"^(.+)-(\d{4}-\d{2}-\d{2})$")


def _dated_stem_prefix(path: Path) -> tuple[str, str] | None:
    m = _DATED_STEM.match(path.stem)
    if not m:
        return None
    return m.group(1), m.group(2)


def _latest_paths_only(paths: list[Path]) -> list[Path]:
    """Keep newest YYYY-MM-DD file per prefix (cursor-transcripts, public-swe-chat, …)."""
    latest: dict[str, Path] = {}
    undated: list[Path] = []
    for path in paths:
        parsed = _dated_stem_prefix(path)
        if parsed is None:
            undated.append(path)
            continue
        prefix, date_str = parsed
        prev = latest.get(prefix)
        if prev is None or date_str > _dated_stem_prefix(prev)[1]:  # type: ignore[index]
            latest[prefix] = path
    return sorted(set(latest.values()) | set(undated))


def _path_excluded(path: Path, exclude_globs: tuple[str, ...]) -> bool:
    for pattern in exclude_globs:
        if pattern and path.match(pattern):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate data/raw JSONL into data/curated/")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--glob", default="*.jsonl")
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=None,
        help="Skip matching raw files (repeatable). Default skips public-stack-v2-dedup*.jsonl",
    )
    parser.add_argument("--tier", type=int, default=1, help="Only write this train_tier")
    parser.add_argument(
        "--session-gitleaks",
        action="store_true",
        help="Per-session gitleaks in curate (slow; prefer scan-raw --gitleaks-per-file)",
    )
    parser.add_argument(
        "--no-gitleaks",
        action="store_true",
        help="Deprecated alias — gitleaks off by default in curate",
    )
    parser.add_argument("--no-presidio", action="store_true", help="Same as --presidio-mode off")
    parser.add_argument(
        "--presidio-mode",
        choices=("off", "pattern", "full"),
        default=None,
        help="Presidio in curate (default off when honoring scan-raw quarantine)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel raw files (default: CURATE_WORKERS env or CPU-2)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--out-suffix",
        default="",
        help="Extra tag in filename, e.g. 'public' → curated-public-YYYY-MM-DD.jsonl",
    )
    parser.add_argument(
        "--honor-safety-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip sessions with rows listed in data/raw/safety-failures-*.jsonl (default on)",
    )
    parser.add_argument(
        "--latest-per-prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only curate the newest dated file per prefix (cursor-transcripts, codex-sessions, public-*)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir or (data_dir() / "raw")
    exclude = tuple(
        args.exclude_glob
        if args.exclude_glob is not None
        else ("public-stack-v2-dedup*.jsonl",)
    )
    paths = sorted(
        p
        for p in raw_dir.glob(args.glob)
        if p.is_file()
        and not p.name.startswith("safety-failures")
        and not _path_excluded(p, exclude)
    )
    if not paths:
        print(f"No raw files in {raw_dir} (glob={args.glob!r}, exclude={exclude})")
        return

    if args.latest_per_prefix:
        before = len(paths)
        paths = _latest_paths_only(paths)
        skipped = before - len(paths)
        if skipped:
            print(
                f"Latest-per-prefix: using {len(paths)} file(s), skipped {skipped} older dated raw",
                flush=True,
            )

    cfg = load_yaml_config().get("curation") or {}
    use_gitleaks = bool(args.session_gitleaks) and not args.no_gitleaks
    presidio_mode = resolve_curate_presidio_mode(
        cli_no_presidio=args.no_presidio,
        cli_mode=args.presidio_mode,
        honor_safety_failures=bool(args.honor_safety_failures),
    )
    n_workers = worker_count("CURATE_WORKERS") if args.workers is None else max(1, args.workers)
    failure_keys = (
        load_safety_failure_keys(raw_dir) if args.honor_safety_failures else set()
    )
    if failure_keys:
        print(f"Safety quarantine: {len(failure_keys)} flagged raw line keys loaded")
    if not use_gitleaks:
        print("curate-raw: session gitleaks off (use scan-raw --gitleaks-per-file + quarantine)")
    print(f"curate-raw: presidio_mode={presidio_mode}", flush=True)
    if presidio_mode != "off":
        print(
            f"curate-raw: Presidio batch n_process={presidio_n_process()} "
            f"session_batch={presidio_session_batch()}",
            flush=True,
        )
    if exclude:
        print(f"Excluding raw globs: {', '.join(exclude)}")

    out_dir = args.out_dir or (data_dir() / "curated")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = f"-{args.out_suffix.strip()}" if args.out_suffix.strip() else ""
    out_path = out_dir / f"curated{suffix}-{stamp}.jsonl"

    print(f"Curating {len(paths)} raw file(s) → {out_path}", flush=True)
    with out_path.open("w", encoding="utf-8") as out_fh:
        written, total_sessions, skipped_sessions, tier_counts, stack_index_rows = (
            _curate_paths(
                paths,
                cfg=cfg,
                failure_keys=failure_keys,
                use_gitleaks=use_gitleaks,
                presidio_mode=presidio_mode,
                tier=args.tier,
                out_fh=out_fh,
                workers=n_workers,
            )
        )

    if stack_index_rows:
        print(f"Skipped {stack_index_rows} stack_index rows inside included files")
    print(
        f"Sessions: {total_sessions} grouped, quarantine-skipped {skipped_sessions}, "
        f"tier counts {dict(tier_counts)}"
    )
    print(f"Wrote {written} tier-{args.tier} examples → {out_path}")


if __name__ == "__main__":
    main()
