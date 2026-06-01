"""Unified ingest for local coding-agent harnesses on this machine."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_core import data_dir, repo_root
from llm_dataprep import (
    aider_history,
    amp_threads,
    antigravity_ingest,
    claude_sessions,
    cline_tasks,
    codex_sessions,
    continue_sessions,
    copilot_cli,
    crush_db,
    cursor_transcripts,
    factory_droid,
    gemini_cli,
    git_diffs,
    goose_sessions,
    kimi_sessions,
    kiro_sessions,
    mux_sessions,
    openclaw_sessions,
    openhands_events,
    opencode_db,
    pi_sessions,
    roo_code,
    t3_threads,
    tokscale_cache,
    watchfire_logs,
    windsurf_vscdb,
    zed_threads,
)
from llm_dataprep.discover import probe_all
from llm_dataprep.harnesses import get_harness, harnesses_for_ingest


def _run_harness(
    name: str,
    *,
    out_dir: Path | None,
    repo: Path | None,
    include_subagents: bool,
    limit_files: int | None,
    max_codex_mb: float | None,
    aider_scan: tuple[Path, ...] | None,
    include_aider_tool: bool,
) -> tuple[str, Path | None, int]:
    spec = get_harness(name)
    if spec.ingest_tier == "detect":
        print(f"{name}: skipped — {spec.notes or 'detect-only'}")
        return name, None, 0
    if spec.ingest_tier == "blocked":
        print(f"{name}: skipped — {spec.notes}")
        return name, None, 0

    if name == "cursor":
        return name, *cursor_transcripts.ingest(
            out_dir=out_dir, include_subagents=include_subagents, limit_files=limit_files
        )
    if name == "codex":
        return name, *codex_sessions.ingest(
            out_dir=out_dir, max_file_mb=max_codex_mb, limit_files=limit_files
        )
    if name == "claude_code":
        return name, *claude_sessions.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "pi":
        return name, *pi_sessions.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "opencode":
        return name, *opencode_db.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "t3code":
        return name, *t3_threads.ingest(out_dir=out_dir, limit_rows=limit_files)
    if name == "aider":
        return name, *aider_history.ingest(
            scan_roots=aider_scan,
            out_dir=out_dir,
            limit_files=limit_files,
            include_tool=include_aider_tool,
        )
    if name == "cline":
        return name, *cline_tasks.ingest(out_dir=out_dir, limit_tasks=limit_files)
    if name == "continue":
        return name, *continue_sessions.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "gemini_cli":
        return name, *gemini_cli.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "copilot":
        return name, *copilot_cli.ingest(out_dir=out_dir, limit_sessions=limit_files)
    if name == "amp":
        return name, *amp_threads.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "factory":
        return name, *factory_droid.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "openhands":
        return name, *openhands_events.ingest(out_dir=out_dir, limit_sessions=limit_files)
    if name == "windsurf":
        path, n = windsurf_vscdb.ingest(out_dir=out_dir, limit_dbs=limit_files)
        if n == 0:
            print(
                "windsurf: 0 records — install Windsurf and chat once, or export via UI. "
                ".pb cache is NOT decrypted."
            )
        return name, path, n
    if name == "git":
        target = repo or repo_root()
        if not (target / ".git").exists():
            print(f"git: skipped — not a git repo ({target}); pass --repo /path/to/repo")
            return name, None, 0
        return name, *git_diffs.ingest(target, out_dir=out_dir)
    if name == "kimi":
        return name, *kimi_sessions.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "goose":
        return name, *goose_sessions.ingest(out_dir=out_dir, limit_rows=limit_files)
    if name == "kiro":
        return name, *kiro_sessions.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "openclaw":
        return name, *openclaw_sessions.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "zed_ai":
        path, n = zed_threads.ingest(out_dir=out_dir, limit_threads=limit_files)
        if n == 0:
            print("zed_ai: 0 records — install Zed, chat once, and: uv sync --package llm-dataprep --extra zed")
        return name, path, n
    if name == "roo_code":
        return name, *roo_code.ingest(out_dir=out_dir, limit_tasks=limit_files)
    if name == "mux":
        return name, *mux_sessions.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "antigravity":
        path, n = antigravity_ingest.ingest(out_dir=out_dir, limit_files=limit_files)
        if n == 0:
            print(
                "antigravity: 0 records — run `tokscale antigravity sync` (editor + LS running). "
                "~/.gemini/antigravity/*.pb are encrypted and not parsed."
            )
        return name, path, n
    if name == "trae":
        path, n = tokscale_cache.ingest_trae(out_dir=out_dir, limit_files=limit_files)
        if n == 0:
            print("trae: 0 records — run `tokscale trae sync` first")
        return name, path, n
    if name == "watchfire":
        return name, *watchfire_logs.ingest(out_dir=out_dir, limit_files=limit_files)
    if name == "crush":
        return name, *crush_db.ingest(out_dir=out_dir, limit_dbs=limit_files)

    raise ValueError(name)


def _default_max_codex_mb() -> float | None:
    from llm_core import config_dir

    path = config_dir() / "default.yaml"
    if path.is_file():
        import yaml

        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        raw = (doc.get("paths") or {}).get("codex_max_file_mb")
        if raw is not None:
            return None if float(raw) == 0 else float(raw)
    return 200.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest local agent logs into data/raw (see packages/dataprep/AGENT_HARNESSES.md)"
    )
    parser.add_argument(
        "--harness",
        default="all",
        help="Comma-separated ids or 'all' (runs full+partial tiers only)",
    )
    parser.add_argument("--list-harnesses", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument("--include-aider-tool", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--aider-scan", type=Path, nargs="*", default=None)
    parser.add_argument("--max-codex-mb", type=float, default=None)
    args = parser.parse_args()

    if args.list_harnesses:
        print(f"{'id':14} {'tier':8} {'ok':3}  detail")
        print("-" * 78)
        for probe in probe_all():
            h = probe.spec
            ok = "yes" if probe.present else "no"
            print(f"{h.harness_id:14} {h.ingest_tier:8} {ok:3}  {probe.detail}")
        print("\nCatalog: packages/dataprep/AGENT_HARNESSES.md")
        return

    if args.harness == "all":
        names = [h.harness_id for h in harnesses_for_ingest()]
    else:
        names = [s.strip() for s in args.harness.split(",") if s.strip()]
        for part in names:
            get_harness(part)

    max_mb = _default_max_codex_mb() if args.max_codex_mb is None else (
        None if args.max_codex_mb == 0 else args.max_codex_mb
    )
    total = 0
    for name in names:
        label, path, n = _run_harness(
            name,
            out_dir=args.out_dir,
            repo=args.repo,
            include_subagents=args.include_subagents,
            limit_files=args.limit_files,
            max_codex_mb=max_mb,
            aider_scan=tuple(args.aider_scan) if args.aider_scan else None,
            include_aider_tool=args.include_aider_tool,
        )
        print(f"{label}: {n} records → {path or '(skipped)'}")
        total += n
    print(f"Total: {total} records → {args.out_dir or data_dir() / 'raw'}")


if __name__ == "__main__":
    main()
