"""Download Hugging Face dataset repos to local cache (parallel snapshot_download)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_core import data_dir
from llm_dataprep.public.loaders import _hf_token
from llm_dataprep.public.registry import PublicDatasetSpec

INGEST_STATE_FILE = ".ingest_state.json"
DOWNLOAD_MARKER = ".download_complete"


def _ingest_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("ingest") or {}
    return {
        "cache_dir": raw.get("cache_dir"),
        "download_workers": int(raw.get("download_workers", 16)),
        "convert_workers": int(raw.get("convert_workers", 8)),
        "refresh_download": bool(raw.get("refresh_download", False)),
        "write_buffer_rows": int(raw.get("write_buffer_rows", 500)),
    }


def hf_cache_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or {}
    settings = _ingest_settings(cfg)
    env = os.environ.get("LLM_HF_CACHE_DIR")
    if env:
        return Path(env).resolve()
    if settings.get("cache_dir"):
        p = Path(str(settings["cache_dir"]))
        return p.resolve() if p.is_absolute() else (data_dir() / p)
    return data_dir() / "hf_cache"


def dataset_cache_dir(dataset_id: str, cfg: dict[str, Any] | None = None) -> Path:
    return hf_cache_root(cfg) / dataset_id.replace("/", "_")


# Session grouping or multi-file side loads — keep conversion single-threaded.
SEQUENTIAL_CONVERT_IDS = frozenset(
    {
        "swe_chat",
        "cooper_qwen9b_coop_claude",
    }
)


def apply_hf_download_env() -> None:
    """Enable hf_xet high-performance mode when not already set by the operator."""
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "24")


def loader_cfg_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash loader filters (exit_status, require_patch, …) — reconvert if these change."""
    payload = {k: v for k, v in sorted(cfg.items()) if k != "max_rows"}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def fetch_hub_meta(
    spec: PublicDatasetSpec,
    token: str | None = None,
) -> dict[str, Any]:
    """Lightweight Hub check: dataset revision + lastModified (one API call)."""
    from huggingface_hub import HfApi

    tok = token if token is not None else _hf_token()
    info = HfApi(token=tok).dataset_info(spec.hf_repo)
    last_mod = info.last_modified or info.lastModified
    if last_mod is not None and last_mod.tzinfo is None:
        last_mod = last_mod.replace(tzinfo=timezone.utc)
    return {
        "sha": info.sha,
        "last_modified": last_mod,
    }


def read_ingest_state(cache_dir: Path) -> dict[str, Any] | None:
    path = cache_dir / INGEST_STATE_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_ingest_state(cache_dir: Path, state: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / INGEST_STATE_FILE).write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def find_latest_raw_jsonl(prefix: str, out_dir: Path | None) -> Path | None:
    root = out_dir or (data_dir() / "raw")
    matches = sorted(root.glob(f"{prefix}-*.jsonl"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _local_marker_time(cache_dir: Path) -> datetime | None:
    marker = cache_dir / DOWNLOAD_MARKER
    if not marker.is_file():
        return None
    return datetime.fromtimestamp(marker.stat().st_mtime, tz=timezone.utc)


def _hub_not_newer_than_local(
    hub_last_modified: datetime | None,
    local_time: datetime,
) -> bool:
    if hub_last_modified is None:
        return False
    return hub_last_modified <= local_time


def try_skip_ingest(
    cache_dir: Path,
    spec: PublicDatasetSpec,
    hub_meta: dict[str, Any],
    loader_cfg: dict[str, Any],
    *,
    raw_prefix: str,
    out_dir: Path | None,
) -> tuple[bool, Path | None, int, str]:
    """Skip download+convert when Hub revision unchanged and raw JSONL exists."""
    fingerprint = loader_cfg_fingerprint(loader_cfg)
    state = read_ingest_state(cache_dir)

    if state:
        if state.get("hub_sha") != hub_meta.get("sha"):
            return False, None, 0, ""
        if state.get("loader_fingerprint") != fingerprint:
            return False, None, 0, ""
        raw_path = Path(str(state.get("raw_path") or ""))
        if raw_path.is_file():
            hub_lm = hub_meta.get("last_modified")
            ingested = state.get("ingested_at", "?")
            hub_s = hub_lm.isoformat() if hub_lm else state.get("hub_last_modified", "?")
            msg = (
                f"up to date — Hub lastModified {hub_s} ≤ local ingest {ingested} "
                f"(revision {str(hub_meta.get('sha', ''))[:12]}…)"
            )
            return True, raw_path, int(state.get("record_count") or 0), msg

    marker_time = _local_marker_time(cache_dir)
    raw_path = find_latest_raw_jsonl(raw_prefix, out_dir)
    if marker_time is None or raw_path is None:
        return False, None, 0, ""

    raw_time = datetime.fromtimestamp(raw_path.stat().st_mtime, tz=timezone.utc)
    local_time = max(marker_time, raw_time)
    if not _hub_not_newer_than_local(hub_meta.get("last_modified"), local_time):
        return False, None, 0, ""

    record_count = 0
    with raw_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                record_count += 1

    write_ingest_state(
        cache_dir,
        {
            "hf_repo": spec.hf_repo,
            "hub_sha": hub_meta.get("sha"),
            "hub_last_modified": (
                hub_meta["last_modified"].isoformat()
                if hub_meta.get("last_modified")
                else None
            ),
            "ingested_at": raw_time.isoformat(),
            "raw_path": str(raw_path.resolve()),
            "record_count": record_count,
            "loader_fingerprint": fingerprint,
            "bootstrapped_from_legacy": True,
        },
    )
    hub_s = (
        hub_meta["last_modified"].isoformat()
        if hub_meta.get("last_modified")
        else "?"
    )
    msg = (
        f"up to date — Hub lastModified {hub_s} ≤ local files "
        f"({raw_path.name}); skipping download + convert"
    )
    return True, raw_path, record_count, msg


def record_ingest_complete(
    cache_dir: Path,
    spec: PublicDatasetSpec,
    hub_meta: dict[str, Any],
    loader_cfg: dict[str, Any],
    raw_path: Path,
    record_count: int,
) -> None:
    write_ingest_state(
        cache_dir,
        {
            "hf_repo": spec.hf_repo,
            "hub_sha": hub_meta.get("sha"),
            "hub_last_modified": (
                hub_meta["last_modified"].isoformat()
                if hub_meta.get("last_modified")
                else None
            ),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(raw_path.resolve()),
            "record_count": record_count,
            "loader_fingerprint": loader_cfg_fingerprint(loader_cfg),
        },
    )
    (cache_dir / DOWNLOAD_MARKER).write_text(
        datetime.now(timezone.utc).isoformat(),
        encoding="utf-8",
    )


def cache_is_current_for_hub(
    cache_dir: Path,
    hub_meta: dict[str, Any],
    *,
    force: bool,
) -> bool:
    if force:
        return False
    state = read_ingest_state(cache_dir)
    if state and state.get("hub_sha") == hub_meta.get("sha"):
        return (cache_dir / DOWNLOAD_MARKER).is_file()
    marker_time = _local_marker_time(cache_dir)
    if marker_time is None:
        return False
    return _hub_not_newer_than_local(hub_meta.get("last_modified"), marker_time)


def ensure_downloaded(
    spec: PublicDatasetSpec,
    *,
    cfg: dict[str, Any] | None = None,
    refresh: bool = False,
    token: str | None = None,
    hub_meta: dict[str, Any] | None = None,
) -> Path:
    """Snapshot-download full dataset repo; resumable and parallel.

    Skips download when Hub ``lastModified`` / revision is not newer than the
    local cache marker (see ``.ingest_state.json``).
    """
    apply_hf_download_env()
    settings = _ingest_settings(cfg or {})
    cache = dataset_cache_dir(spec.dataset_id, cfg)
    cache.mkdir(parents=True, exist_ok=True)
    force = bool(refresh or settings.get("refresh_download"))

    tok = token if token is not None else _hf_token()
    if hub_meta is None:
        hub_meta = fetch_hub_meta(spec, tok)

    if cache_is_current_for_hub(cache, hub_meta, force=force):
        return cache

    if spec.gated and not tok:
        raise RuntimeError(
            f"{spec.dataset_id} is gated — run `hf auth login` or set HF_TOKEN "
            f"(https://huggingface.co/datasets/{spec.hf_repo})"
        )

    from huggingface_hub import snapshot_download

    workers = int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", settings["download_workers"]))
    hub_s = (
        hub_meta["last_modified"].isoformat()
        if hub_meta.get("last_modified")
        else "unknown"
    )
    print(
        f"{spec.dataset_id}: Hub updated ({hub_s}) — downloading {spec.hf_repo} → {cache} "
        f"(max_workers={workers})",
        flush=True,
    )
    snapshot_download(
        repo_id=spec.hf_repo,
        repo_type="dataset",
        local_dir=str(cache),
        token=tok,
        max_workers=workers,
        force_download=force,
        revision=hub_meta.get("sha"),
    )
    (cache / DOWNLOAD_MARKER).write_text(
        datetime.now(timezone.utc).isoformat(),
        encoding="utf-8",
    )
    return cache


def list_parquet_shards(cache_dir: Path) -> list[Path]:
    skip = {".cache", ".git"}
    shards: list[Path] = []
    for path in sorted(cache_dir.rglob("*.parquet")):
        if not path.is_file():
            continue
        if any(part in skip for part in path.relative_to(cache_dir).parts):
            continue
        if path.name.startswith(".ingest_part_"):
            continue
        shards.append(path)
    return shards


def cache_stats(cache_dir: Path) -> dict[str, Any]:
    shards = list_parquet_shards(cache_dir)
    total_bytes = sum(p.stat().st_size for p in shards)
    return {
        "parquet_files": len(shards),
        "parquet_bytes": total_bytes,
        "parquet_gib": round(total_bytes / (1024**3), 2),
    }
