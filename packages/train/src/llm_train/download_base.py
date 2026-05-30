"""Download HF base weights for train (disk-only; safe while GPU train runs)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from llm_train.config import decensor_settings, train_settings

_EXPECTED_SHARDS = 4


def _cache_repo_dir(model_id: str) -> Path:
    slug = model_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{slug}"


def _cache_status(model_id: str) -> tuple[int, int]:
    """Return (bytes on disk, shard symlinks in latest snapshot)."""
    root = _cache_repo_dir(model_id)
    if not root.is_dir():
        return 0, 0
    bytes_on_disk = sum(
        p.stat().st_size for p in root.rglob("*") if p.is_file()
    )
    snapshots = root / "snapshots"
    shards = 0
    if snapshots.is_dir():
        dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime)
        if dirs:
            shards = len(list(dirs[-1].glob("model-*-of-*.safetensors")))
    return bytes_on_disk, shards


def _print_cache_hint(model_id: str) -> None:
    nbytes, shards = _cache_status(model_id)
    gib = nbytes / (1024**3)
    if nbytes == 0:
        print("HF cache: empty (expect ~14 GiB for 7B bf16)", flush=True)
        return
    print(
        f"HF cache: {gib:.1f} GiB on disk, {shards}/{_EXPECTED_SHARDS} weight shards linked",
        flush=True,
    )
    if shards >= _EXPECTED_SHARDS and gib >= 13.0:
        print("Looks complete — snapshot_download will verify quickly.", flush=True)
    elif gib > 1.0:
        print(
            "Large shards download slowly; progress bar may sit at 0% for minutes.",
            flush=True,
        )


def _snapshot_download(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    _print_cache_hint(model_id)
    print(f"Fetching {model_id} → HF cache…", flush=True)
    path = snapshot_download(repo_id=model_id)
    nbytes, shards = _cache_status(model_id)
    print(
        f"Cached at: {path} ({nbytes / (1024**3):.1f} GiB, {shards} shards)",
        flush=True,
    )
    if shards < _EXPECTED_SHARDS:
        print(
            f"Warning: expected {_EXPECTED_SHARDS} shards, got {shards} — re-run to resume.",
            file=sys.stderr,
        )
    return path


def _ollama_pull(tag: str) -> None:
    if not shutil.which("ollama"):
        print("ollama not on PATH — skip pull", file=sys.stderr)
        return
    print(f"ollama pull {tag} …", flush=True)
    subprocess.run(["ollama", "pull", tag], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache HF train bases (abliterated + aligned reference)"
    )
    parser.add_argument(
        "--decensor",
        action="store_true",
        help="Download abliterated base + optional Ollama reference tag",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override HF repo id (default: config train.base_model or decensor.base_model)",
    )
    parser.add_argument(
        "--also-aligned",
        action="store_true",
        help="Also cache upstream aligned Qwen (for A/B)",
    )
    parser.add_argument(
        "--ollama-pull",
        action="store_true",
        help="Pull config decensor ollama_reference after HF download",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Only ollama pull (skip HF snapshot)",
    )
    args = parser.parse_args()

    dec = decensor_settings()
    if args.decensor:
        model_id = args.model or dec["base_model"]
        ollama_tag = dec["ollama_reference"]
    else:
        model_id = args.model or train_settings()["base_model"]
        ollama_tag = None

    if not args.no_hf:
        _snapshot_download(model_id)
        if args.also_aligned or args.decensor:
            aligned = dec["upstream_aligned"]
            if aligned != model_id:
                _snapshot_download(aligned)

    if args.ollama_pull or args.decensor:
        tag = ollama_tag or dec["ollama_reference"]
        if tag:
            _ollama_pull(tag)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
