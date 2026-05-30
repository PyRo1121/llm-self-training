"""Build curated JSONL from data/raw ingest rows (session-grouped chat)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from llm_core import data_dir
from llm_dataprep.filters import scan_text
from llm_dataprep.safety_quarantine import load_safety_failure_keys, session_has_quarantined_row
from llm_dataprep.style_tags import enrich_meta
from llm_dataprep.tier1 import assign_train_tier


def _load_curation_config() -> dict[str, Any]:
    from llm_core import config_dir

    path = config_dir() / "default.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return doc.get("curation") or {}


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
    if first.get("source") != "public":
        return {}
    return {
        "label": first.get("label") or "accepted",
        "exec": first.get("exec") or "unknown",
        "verify": first.get("verify") or "unknown",
        "data_source": "public",
        "public_dataset": first.get("dataset_id"),
    }


def iter_raw_records(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
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
    return chunks or [messages]


def curate_session(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    use_gitleaks: bool,
    use_presidio: bool,
) -> list[dict[str, Any]]:
    skip_roles = set(cfg.get("skip_roles") or ["developer"])
    max_chars = int(cfg.get("max_chars_per_message", 16_000))
    min_msg = int(cfg.get("min_message_chars", 40))
    min_messages = int(cfg.get("min_messages", 2))
    min_total = int(cfg.get("min_total_chars", 200))
    max_per_example = int(cfg.get("max_messages_per_example", 24))
    chunk_overlap = int(cfg.get("chunk_overlap_messages", 4))
    bootstrap = bool(cfg.get("bootstrap_mode", True))

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

    windows = chunk_messages(
        messages,
        max_messages=max_per_example,
        overlap=chunk_overlap,
        min_messages=min_messages,
        min_total_chars=min_total,
    )

    first = rows[0]
    session_id = first.get("session_id")
    out: list[dict[str, Any]] = []
    for chunk_idx, window in enumerate(windows):
        combined = "\n\n".join(m["content"] for m in window)
        safety = scan_text(
            combined,
            use_gitleaks=use_gitleaks and cfg.get("filter_secrets_and_pii", True),
            use_presidio=use_presidio and cfg.get("filter_secrets_and_pii", True),
        )

        pub = _public_meta_defaults(rows)
        meta: dict[str, Any] = {
            "label": pub.get("label", "accepted"),
            "exec": pub.get("exec", "unknown"),
            "verify": pub.get("verify", "unknown"),
            "project": _infer_project(
                first.get("source_path"), first.get("dataset_id")
            ),
            "harness": first.get("harness") or first.get("source"),
            "session_id": session_id,
            "chunk_index": chunk_idx,
            "chunk_count": len(windows),
            "source_path": first.get("source_path"),
            "safety": safety.to_dict(),
            "safety_ok": safety.ok,
        }
        if pub:
            meta.update(pub)
        meta["train_tier"] = assign_train_tier(
            meta, bootstrap=bootstrap, quality_ok=True
        )
        enrich_meta(meta, window)
        out.append({"messages": window, "meta": meta})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate data/raw JSONL into data/curated/")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--glob", default="*.jsonl")
    parser.add_argument("--tier", type=int, default=1, help="Only write this train_tier")
    parser.add_argument("--no-gitleaks", action="store_true")
    parser.add_argument("--no-presidio", action="store_true")
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
    args = parser.parse_args()

    raw_dir = args.raw_dir or (data_dir() / "raw")
    paths = sorted(
        p
        for p in raw_dir.glob(args.glob)
        if p.is_file() and not p.name.startswith("safety-failures")
    )
    if not paths:
        print(f"No raw files in {raw_dir}")
        return

    cfg = _load_curation_config()
    use_gitleaks = not args.no_gitleaks
    use_presidio = not args.no_presidio
    failure_keys = (
        load_safety_failure_keys(raw_dir) if args.honor_safety_failures else set()
    )
    if failure_keys:
        print(f"Safety quarantine: {len(failure_keys)} flagged raw line keys loaded")

    by_session: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in iter_raw_records(paths):
        sid = str(rec.get("session_id") or "unknown")
        harness = str(rec.get("harness") or rec.get("source") or "unknown")
        by_session[(harness, sid)].append(rec)

    out_dir = args.out_dir or (data_dir() / "curated")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = f"-{args.out_suffix.strip()}" if args.out_suffix.strip() else ""
    out_path = out_dir / f"curated{suffix}-{stamp}.jsonl"

    written = 0
    skipped_sessions = 0
    tier_counts: dict[int, int] = defaultdict(int)
    with out_path.open("w", encoding="utf-8") as out_fh:
        for _key, rows in sorted(by_session.items()):
            if session_has_quarantined_row(rows, failure_keys):
                skipped_sessions += 1
                continue
            for curated in curate_session(
                rows, cfg, use_gitleaks=use_gitleaks, use_presidio=use_presidio
            ):
                tier = int(curated["meta"].get("train_tier", 0))
                tier_counts[tier] += 1
                if tier != args.tier:
                    continue
                out_fh.write(json.dumps(curated, ensure_ascii=False) + "\n")
                written += 1

    print(
        f"Sessions: {len(by_session)} grouped, quarantine-skipped {skipped_sessions}, "
        f"tier counts {dict(tier_counts)}"
    )
    print(f"Wrote {written} tier-{args.tier} examples → {out_path}")


if __name__ == "__main__":
    main()
