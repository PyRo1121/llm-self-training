"""Ingest public Hugging Face coding datasets into data/raw."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from llm_core import data_dir
from llm_core.yaml_config import load_yaml_config
from llm_dataprep.public.fast_ingest import ingest_one_fast
from llm_dataprep.public.hf_cache import apply_hf_download_env, hf_cache_root
from llm_dataprep.public.loaders import _hf_auth_label
from llm_dataprep.public.registry import get_spec, list_specs


def _load_config() -> dict[str, Any]:
    doc = load_yaml_config()
    return doc.get("public_datasets") or {}


def ingest_one(
    dataset_id: str,
    *,
    out_dir: Path | None,
    max_rows: int | None,
    skip_gated: bool,
    replace: bool = False,
    mode: str = "fast",
    refresh_download: bool = False,
) -> tuple[str, Path | None, int]:
    cfg_root = _load_config()
    spec = get_spec(dataset_id)
    cfg_ds = (cfg_root.get("datasets") or {}).get(dataset_id) or {}
    if cfg_ds.get("enabled") is False:
        print(f"{dataset_id}: disabled in config")
        return dataset_id, None, 0

    cap = max_rows
    if cap is None:
        cap = cfg_ds.get("max_rows", spec.default_max_rows)

    return ingest_one_fast(
        spec,
        spec.loader,
        cfg=cfg_ds,
        ingest_cfg=cfg_root,
        out_dir=out_dir,
        max_rows=cap,
        skip_gated=skip_gated,
        replace=replace,
        refresh_download=refresh_download,
        mode=mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest public HF coding datasets (see docs/oss/PUBLIC-DATASETS.md)"
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated ids or 'all' (default: enabled in config)",
    )
    parser.add_argument("--list", action="store_true", help="List registry and exit")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="Override cap per dataset")
    parser.add_argument("--skip-gated", action="store_true", help="Skip gated sets (e.g. SWE-chat)")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite today's raw file for each dataset instead of appending",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "stream"),
        default=None,
        help="fast=download parquet cache + local convert (default); stream=legacy HF streaming",
    )
    parser.add_argument(
        "--refresh-download",
        action="store_true",
        help="Force re-download and re-convert even when Hub revision unchanged",
    )
    parser.add_argument(
        "--remote-stream",
        action="store_true",
        help="Alias for --mode stream (legacy row-by-row Hub streaming)",
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'id':22} {'tier':6} {'released':10} {'gated':5}  repo")
        print("-" * 80)
        for spec in list_specs():
            print(
                f"{spec.dataset_id:22} {spec.tier:6} {spec.released:10} "
                f"{'yes' if spec.gated else 'no':5}  {spec.hf_repo}"
            )
        print("\nDocs: docs/oss/PUBLIC-DATASETS.md")
        return

    cfg = _load_config()
    enabled_default = cfg.get("enabled", True)
    if not enabled_default:
        print("public_datasets.enabled is false in config/default.yaml")
        return

    apply_hf_download_env()
    mode = "stream" if args.remote_stream else (args.mode or (cfg.get("ingest") or {}).get("mode") or "fast")
    cache_root = hf_cache_root(cfg)
    print(f"Hugging Face: {_hf_auth_label()}")
    print(f"Ingest mode: {mode} | HF cache: {cache_root}")

    if args.datasets == "all":
        cfg_sets = cfg.get("datasets") or {}
        ids = [
            spec.dataset_id
            for spec in list_specs()
            if cfg_sets.get(spec.dataset_id, {}).get("enabled", True)
        ]
    else:
        ids = [s.strip() for s in args.datasets.split(",") if s.strip()]
        for did in ids:
            get_spec(did)

    total = 0
    errors = 0
    for did in ids:
        try:
            _id, path, n = ingest_one(
                did,
                out_dir=args.out_dir,
                max_rows=args.max_rows,
                skip_gated=args.skip_gated,
                replace=args.replace,
                mode=mode,
                refresh_download=args.refresh_download,
            )
            print(f"{_id}: {n} records → {path or '(skipped)'}")
            total += n
        except Exception as exc:
            errors += 1
            print(f"{did}: ERROR — {exc}", file=sys.stderr)

    print(f"Total public records: {total} → {args.out_dir or data_dir() / 'raw'}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
