"""Run Phase 1 data-lake pipeline end-to-end (ingest → scan → curate → link → replay → audit)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_core import data_dir, repo_root


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n→ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 data lake — full pipeline")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument(
        "--fresh-raw",
        action="store_true",
        help="Archive existing data/raw/*.jsonl before ingest (avoid duplicate append)",
    )
    parser.add_argument("--repo", type=Path, default=None, help="Git repo for git harness")
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument(
        "--gitleaks",
        action="store_true",
        help="Per-file gitleaks on scan-raw (default on when gitleaks is on PATH)",
    )
    parser.add_argument(
        "--no-gitleaks",
        action="store_true",
        help="Disable gitleaks even if installed",
    )
    parser.add_argument("--presidio", action="store_true", help="Presidio on audit-sample")
    parser.add_argument(
        "--no-honor-safety-failures",
        action="store_true",
        help="Do not skip sessions listed in safety-failures-*.jsonl at curate",
    )
    parser.add_argument("--mark-exec", action="store_true", help="link_logs_to_diffs sets exec=pass")
    parser.add_argument("--public", action="store_true", help="Run public-ingest before personal ingest")
    parser.add_argument("--skip-gated", action="store_true", help="Skip gated HF datasets (SWE-chat)")
    args = parser.parse_args()

    root = repo_root()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    curated = data_dir() / "curated" / f"curated-{stamp}.jsonl"

    raw_dir = data_dir() / "raw"
    if args.fresh_raw and raw_dir.is_dir():
        archive = raw_dir / f"archive-{stamp}-phase1"
        archive.mkdir(parents=True, exist_ok=True)
        for p in raw_dir.glob("*.jsonl"):
            if p.name.startswith("safety-failures"):
                continue
            p.rename(archive / p.name)
        print(f"Archived prior raw JSONL → {archive}")

    if args.public:
        pub_cmd = ["uv", "run", "--package", "llm-dataprep", "public-ingest", "--replace"]
        if args.skip_gated:
            pub_cmd.append("--skip-gated")
        _run(pub_cmd, cwd=root)

    if not args.skip_ingest:
        _run(
            ["uv", "run", "--package", "llm-dataprep", "agent-ingest"]
            + (["--include-subagents"] if args.include_subagents else [])
            + (["--repo", str(args.repo)] if args.repo else []),
            cwd=root,
        )

    use_gitleaks = (args.gitleaks or bool(shutil.which("gitleaks"))) and not args.no_gitleaks
    scan_cmd = ["uv", "run", "--package", "llm-dataprep", "scan-raw", "--no-presidio"]
    if use_gitleaks:
        scan_cmd.extend(["--gitleaks", "--gitleaks-per-file"])
    _run(scan_cmd, cwd=root)

    curate_cmd = [
        "uv",
        "run",
        "--package",
        "llm-dataprep",
        "curate-raw",
        "--no-gitleaks",
        "--no-presidio",
    ]
    if args.no_honor_safety_failures:
        curate_cmd.append("--no-honor-safety-failures")
    _run(curate_cmd, cwd=root)

    if not curated.is_file():
        # curate writes dated file — pick latest
        curated_dir = data_dir() / "curated"
        candidates = sorted(curated_dir.glob("curated-*.jsonl"))
        if not candidates:
            print("No curated output found")
            sys.exit(1)
        curated = candidates[-1]

    link_cmd = [
        "uv",
        "run",
        "--package",
        "llm-dataprep",
        "link-logs-to-diffs",
        "--curated",
        str(curated),
        "--out",
        str(curated),
    ]
    if args.mark_exec:
        link_cmd.append("--mark-exec")
    _run(link_cmd, cwd=root)

    _run(
        [
            "uv",
            "run",
            "--package",
            "llm-dataprep",
            "replay-seed",
            "--curated",
            str(curated),
        ],
        cwd=root,
    )

    audit_cmd = [
        "uv",
        "run",
        "--package",
        "llm-dataprep",
        "audit-sample",
        "--curated",
        str(curated),
    ]
    if args.presidio:
        audit_cmd.append("--use-presidio")
    if use_gitleaks:
        audit_cmd.append("--gitleaks")
    _run(audit_cmd, cwd=root)

    _run(
        [
            "uv",
            "run",
            "--package",
            "llm-dataprep",
            "warehouse-sync-registry",
        ],
        cwd=root,
    )

    wh_load = [
        "uv",
        "run",
        "--package",
        "llm-dataprep",
        "warehouse-load",
        "--latest",
        "--tier",
        "1",
    ]
    if args.fresh_raw:
        wh_load.append("--clear")
    _run(wh_load, cwd=root)

    print(f"\nPhase 1 pipeline done. Curated: {curated}")


if __name__ == "__main__":
    main()
