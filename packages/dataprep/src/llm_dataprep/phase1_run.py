"""Run Phase 1 data-lake pipeline end-to-end (ingest → scan → curate → link → replay → audit)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_core import data_dir, repo_root

from llm_dataprep.perf import worker_count
from llm_dataprep.scan_config import resolve_curate_presidio_mode, resolve_scan_presidio_mode
from llm_dataprep.safety_policy import load_safety_policy, safety_policy_version


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n→ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _worker_args() -> tuple[int, int]:
    scan = worker_count("SCAN_WORKERS")
    curate = worker_count("CURATE_WORKERS")
    return scan, curate


def _print_safety_policy(
    *,
    use_gitleaks: bool,
    gitleaks_per_file: bool,
    scan_presidio_mode: str,
    curate_presidio_mode: str,
    honor_safety_failures: bool,
) -> None:
    pol = load_safety_policy()
    print("\nSafety policy (config/default.yaml + safety-allowlist.yaml):")
    print(f"  version={safety_policy_version()}")
    print(f"  quarantine_severity={pol.quarantine_severity.value}")
    print(
        f"  scan: gitleaks={'on' if use_gitleaks else 'off'}"
        f" severity={pol.gitleaks_severity.value} per_file={gitleaks_per_file}"
    )
    print(
        f"  scan presidio_mode={scan_presidio_mode}"
        f" curate presidio_mode={curate_presidio_mode}"
        f" block_entities={len(pol.presidio_block_entities)}"
        f" entities={len(pol.presidio_entities)}"
    )
    print(f"  allowlist={len(pol.exact_allowlist)} exact + {len(pol.allowlist_regex)} regex")
    print(f"  diff_harnesses={','.join(sorted(pol.diff_harnesses))}")
    print(f"  curate honor_safety_failures={honor_safety_failures}")
    print("  scope: secrets + PII only (not topic/refusal filtering)")


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
        help="Force gitleaks on scan-raw (default on when gitleaks is on PATH)",
    )
    parser.add_argument(
        "--no-gitleaks",
        action="store_true",
        help="Disable gitleaks even if installed",
    )
    parser.add_argument(
        "--gitleaks-per-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan-raw: one gitleaks pass per JSONL file (default on; use --no-gitleaks-per-file for per-row)",
    )
    parser.add_argument(
        "--no-presidio",
        action="store_true",
        help="Skip Presidio in scan-raw and curate-raw (same as --presidio-mode off)",
    )
    parser.add_argument(
        "--presidio-mode",
        choices=("off", "pattern", "full"),
        default=None,
        help="Override scan + curate Presidio mode (default from config)",
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
    parser.add_argument("--workers", type=int, default=None, help="Parallel scan/curate file workers")
    args = parser.parse_args()

    root = repo_root()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    curated = data_dir() / "curated" / f"curated-{stamp}.jsonl"
    scan_workers, curate_workers = _worker_args()
    if args.workers is not None:
        scan_workers = curate_workers = max(1, args.workers)

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
    gitleaks_per_file = bool(args.gitleaks_per_file)
    honor_safety_failures = not args.no_honor_safety_failures
    scan_presidio_mode = resolve_scan_presidio_mode(
        cli_no_presidio=args.no_presidio,
        cli_mode=args.presidio_mode,
    )
    curate_presidio_mode = resolve_curate_presidio_mode(
        cli_no_presidio=args.no_presidio,
        cli_mode=args.presidio_mode,
        honor_safety_failures=honor_safety_failures,
    )
    scan_cmd = ["uv", "run", "--package", "llm-dataprep", "scan-raw", "--workers", str(scan_workers)]
    if use_gitleaks:
        scan_cmd.append("--gitleaks")
        if not gitleaks_per_file:
            scan_cmd.append("--no-gitleaks-per-file")
    if scan_presidio_mode == "off":
        scan_cmd.append("--no-presidio")
    else:
        scan_cmd.extend(["--presidio-mode", scan_presidio_mode])
    _run(scan_cmd, cwd=root)

    curate_cmd = [
        "uv",
        "run",
        "--package",
        "llm-dataprep",
        "curate-raw",
        "--workers",
        str(curate_workers),
    ]
    if curate_presidio_mode == "off":
        curate_cmd.append("--no-presidio")
    else:
        curate_cmd.extend(["--presidio-mode", curate_presidio_mode])
    if args.no_honor_safety_failures:
        curate_cmd.append("--no-honor-safety-failures")
    _run(curate_cmd, cwd=root)

    if not curated.is_file():
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
    if args.presidio or scan_presidio_mode == "full":
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
    print(
        f"Workers: scan={scan_workers} curate={curate_workers} "
        f"scan_presidio={scan_presidio_mode} curate_presidio={curate_presidio_mode}"
    )
    _print_safety_policy(
        use_gitleaks=use_gitleaks,
        gitleaks_per_file=gitleaks_per_file,
        scan_presidio_mode=scan_presidio_mode,
        curate_presidio_mode=curate_presidio_mode,
        honor_safety_failures=honor_safety_failures,
    )


if __name__ == "__main__":
    main()
