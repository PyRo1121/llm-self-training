"""Seed data/replay from curated tier-1 sample + tier-2 (PLAN continual FT buffer)."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llm_core import data_dir


def _load_replay_config() -> dict[str, Any]:
    from llm_core import config_dir

    path = config_dir() / "default.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return doc.get("replay") or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed data/replay from curated JSONL")
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = _load_replay_config()
    target = int(cfg.get("target_rows", 300))
    tier1_fraction = float(cfg.get("tier1_sample_fraction", 0.25))
    include_tier2 = bool(cfg.get("include_tier2", True))

    tier1: list[dict[str, Any]] = []
    tier2: list[dict[str, Any]] = []
    with args.curated.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tier = int(row.get("meta", {}).get("train_tier", 0))
            if tier == 1:
                tier1.append(row)
            elif tier == 2:
                tier2.append(row)

    rng = random.Random(args.seed)
    picked: list[dict[str, Any]] = []
    if include_tier2:
        picked.extend(tier2)

    cap = max(0, target - len(picked))
    sample_n = min(cap, int(len(tier1) * tier1_fraction), len(tier1))
    if sample_n:
        picked.extend(rng.sample(tier1, sample_n))

    out_dir = args.out_dir or (data_dir() / "replay")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = out_dir / f"replay-{stamp}.jsonl"

    with out_path.open("w", encoding="utf-8") as out_fh:
        for row in picked:
            meta = row.setdefault("meta", {})
            meta["replay_stratum"] = True
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Replay seed: tier2={len(tier2)} tier1_sample={sample_n} "
        f"→ {len(picked)} rows written to {out_path}"
    )


if __name__ == "__main__":
    main()
