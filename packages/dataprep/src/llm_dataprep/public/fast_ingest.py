"""Fast public ingest: HF snapshot cache → local parquet → parallel JSONL conversion."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator

from llm_dataprep.public.hf_cache import (
    SEQUENTIAL_CONVERT_IDS,
    _ingest_settings,
    cache_stats,
    dataset_cache_dir,
    ensure_downloaded,
    fetch_hub_meta,
    list_parquet_shards,
    record_ingest_complete,
    try_skip_ingest,
)
from llm_dataprep.public.loaders import set_loader_context
from llm_dataprep.public.registry import PublicDatasetSpec
from llm_dataprep.raw_io import append_records_buffered, merge_jsonl_parts


def _worker_convert_shard(
    payload: tuple[str, str, list[str], dict[str, Any]],
) -> tuple[str, int]:
    """ProcessPool worker: convert assigned parquet shards → temp JSONL."""
    dataset_id, cache_dir, shard_strs, loader_cfg = payload
    from llm_dataprep.public.registry import get_spec

    spec = get_spec(dataset_id)
    loader = spec.loader
    shard_paths = [Path(s) for s in shard_strs]
    part_path = Path(cache_dir) / f".ingest_part_{os.getpid()}_{id(shard_strs)}.jsonl"

    set_loader_context(local_dir=cache_dir, shard_files=shard_paths)
    try:
        count = append_records_buffered(
            part_path,
            loader(hf_repo=spec.hf_repo, **loader_cfg),
            buffer_rows=500,
            replace=True,
        )
    finally:
        set_loader_context(local_dir=None, shard_files=None)

    return str(part_path), count


def ingest_one_fast(
    spec: PublicDatasetSpec,
    loader: Callable[..., Iterator[dict[str, Any]]],
    *,
    cfg: dict[str, Any],
    ingest_cfg: dict[str, Any],
    out_dir: Path | None,
    max_rows: int | None,
    skip_gated: bool,
    replace: bool,
    refresh_download: bool,
    mode: str,
) -> tuple[str, Path | None, int]:
    from llm_dataprep.public.loaders import _hf_token

    if spec.gated and skip_gated and not _hf_token():
        print(f"{spec.dataset_id}: skipped (gated — run: hf auth login)")
        return spec.dataset_id, None, 0

    loader_cfg = {
        k: v
        for k, v in cfg.items()
        if k not in ("enabled", "max_rows")
    }
    if max_rows is not None:
        loader_cfg["max_rows"] = max_rows
    elif cfg.get("max_rows") is not None:
        loader_cfg["max_rows"] = cfg.get("max_rows")
    elif spec.default_max_rows is not None:
        loader_cfg["max_rows"] = spec.default_max_rows

    prefix = f"public-{spec.dataset_id.replace('_', '-')}"
    settings = _ingest_settings(ingest_cfg)
    force = bool(refresh_download or settings.get("refresh_download"))
    capped = loader_cfg.get("max_rows") is not None
    cache = dataset_cache_dir(spec.dataset_id, ingest_cfg)

    if mode == "stream":
        records = loader(hf_repo=spec.hf_repo, **loader_cfg)
        out_path, n = append_records_buffered_to_dated(
            prefix,
            records,
            out_dir=out_dir,
            replace=replace,
            buffer_rows=settings["write_buffer_rows"],
        )
        return spec.dataset_id, out_path, n

    tok = _hf_token()
    hub_meta = fetch_hub_meta(spec, tok)

    if not force and not capped:
        skipped, skip_path, skip_count, skip_msg = try_skip_ingest(
            cache,
            spec,
            hub_meta,
            loader_cfg,
            raw_prefix=prefix,
            out_dir=out_dir,
        )
        if skipped and skip_path is not None:
            print(f"{spec.dataset_id}: {skip_msg}", flush=True)
            return spec.dataset_id, skip_path, skip_count

    cache = ensure_downloaded(
        spec,
        cfg=ingest_cfg,
        refresh=force,
        token=tok,
        hub_meta=hub_meta,
    )
    stats = cache_stats(cache)
    print(
        f"{spec.dataset_id}: cache {stats['parquet_files']} parquet file(s), "
        f"{stats['parquet_gib']} GiB",
        flush=True,
    )

    shards = list_parquet_shards(cache)
    convert_workers = int(
        os.environ.get("PUBLIC_CONVERT_WORKERS", settings["convert_workers"])
    )
    use_parallel = (
        convert_workers > 1
        and len(shards) >= convert_workers
        and spec.dataset_id not in SEQUENTIAL_CONVERT_IDS
        and not capped
    )

    if use_parallel:
        n, out_path = _parallel_convert(
            spec,
            cache,
            loader_cfg,
            convert_workers,
            prefix,
            out_dir,
            replace,
        )
        record_ingest_complete(cache, spec, hub_meta, loader_cfg, out_path, n)
        return spec.dataset_id, out_path, n

    set_loader_context(local_dir=str(cache), shard_files=None)
    try:
        records = loader(hf_repo=spec.hf_repo, **loader_cfg)
        out_path, n = append_records_buffered_to_dated(
            prefix,
            records,
            out_dir=out_dir,
            replace=replace,
            buffer_rows=settings["write_buffer_rows"],
        )
    finally:
        set_loader_context(local_dir=None, shard_files=None)
    record_ingest_complete(cache, spec, hub_meta, loader_cfg, out_path, n)
    return spec.dataset_id, out_path, n


def append_records_buffered_to_dated(
    prefix: str,
    records: Iterator[dict[str, Any]],
    *,
    out_dir: Path | None,
    replace: bool,
    buffer_rows: int,
) -> tuple[Path, int]:
    from llm_dataprep.raw_io import dated_raw_path

    path = dated_raw_path(prefix, out_dir)
    if replace and path.is_file():
        path.unlink()
        mode_append = False
    elif path.is_file() and not replace:
        mode_append = True
    else:
        mode_append = False

    if mode_append:
        n = append_records_buffered(path, records, buffer_rows=buffer_rows, replace=False)
    else:
        n = append_records_buffered(path, records, buffer_rows=buffer_rows, replace=True)
    return path, n


def _parallel_convert(
    spec: PublicDatasetSpec,
    cache: Path,
    loader_cfg: dict[str, Any],
    num_workers: int,
    prefix: str,
    out_dir: Path | None,
    replace: bool,
) -> tuple[int, Path]:
    from llm_dataprep.raw_io import dated_raw_path

    shards = list_parquet_shards(cache)
    buckets: list[list[Path]] = [[] for _ in range(num_workers)]
    for i, shard in enumerate(shards):
        buckets[i % num_workers].append(shard)

    tasks = [
        (
            spec.dataset_id,
            str(cache),
            [str(p) for p in bucket],
            dict(loader_cfg),
        )
        for bucket in buckets
        if bucket
    ]

    print(
        f"{spec.dataset_id}: parallel convert {len(shards)} shard(s) "
        f"with {len(tasks)} worker(s)",
        flush=True,
    )

    part_paths: list[Path] = []
    total = 0
    try:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(_worker_convert_shard, t) for t in tasks]
            for fut in as_completed(futures):
                part_str, count = fut.result()
                part_paths.append(Path(part_str))
                total += count
                print(
                    f"{spec.dataset_id}: shard batch done — {count} records "
                    f"({part_str})",
                    flush=True,
                )

        out_path = dated_raw_path(prefix, out_dir)
        if replace and out_path.is_file():
            out_path.unlink()
        merge_jsonl_parts(part_paths, out_path)
    finally:
        for p in part_paths:
            p.unlink(missing_ok=True)

    print(f"{spec.dataset_id}: merged {total} records → {out_path}", flush=True)
    return total, out_path
