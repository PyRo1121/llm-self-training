"""Ingest public Hugging Face coding datasets into data/raw."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from llm_core import data_dir
from llm_dataprep.public.loaders import _hf_auth_label, _hf_token
from llm_dataprep.public.registry import get_spec, list_specs
from llm_dataprep.raw_io import append_records


def _load_config() -> dict[str, Any]:
    from llm_core import config_dir

    path = config_dir() / "default.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return doc.get("public_datasets") or {}


def ingest_one(
    dataset_id: str,
    *,
    out_dir: Path | None,
    max_rows: int | None,
    skip_gated: bool,
    replace: bool = False,
) -> tuple[str, Path | None, int]:
    spec = get_spec(dataset_id)
    if spec.gated and skip_gated and not _hf_token():
        print(f"{dataset_id}: skipped (gated — run: hf auth login)")
        return dataset_id, None, 0

    cfg_ds = (_load_config().get("datasets") or {}).get(dataset_id) or {}
    if cfg_ds.get("enabled") is False:
        print(f"{dataset_id}: disabled in config")
        return dataset_id, None, 0

    cap = max_rows
    if cap is None:
        cap = cfg_ds.get("max_rows", spec.default_max_rows)

    records = spec.loader(max_rows=cap, hf_repo=spec.hf_repo)
    prefix = f"public-{dataset_id.replace('_', '-')}"
    path, n = append_records(prefix, records, out_dir=out_dir, replace=replace)
    return dataset_id, path, n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest public HF coding datasets (see docs/PUBLIC-DATASETS.md)"
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
    args = parser.parse_args()

    if args.list:
        print(f"{'id':22} {'tier':6} {'released':10} {'gated':5}  repo")
        print("-" * 80)
        for spec in list_specs():
            print(
                f"{spec.dataset_id:22} {spec.tier:6} {spec.released:10} "
                f"{'yes' if spec.gated else 'no':5}  {spec.hf_repo}"
            )
        print("\nDocs: docs/PUBLIC-DATASETS.md")
        return

    cfg = _load_config()
    enabled_default = cfg.get("enabled", True)
    if not enabled_default:
        print("public_datasets.enabled is false in config/default.yaml")
        return

    print(f"Hugging Face: {_hf_auth_label()}")

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
