"""Parallel dataprep tuning — env overrides with sane CPU/RAM defaults."""

from __future__ import annotations

import os
from functools import lru_cache


def _cpu_count() -> int:
    try:
        import multiprocessing

        return multiprocessing.cpu_count() or 4
    except OSError:
        return 4


def worker_count(env_key: str, *, default: int | None = None) -> int:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return max(1, int(raw))
    if default is not None:
        return max(1, default)
    # Leave headroom for OS + one heavy Presidio/spaCy worker.
    return max(1, min(12, _cpu_count() - 2))


@lru_cache(maxsize=1)
def presidio_n_process() -> int:
    return worker_count("PRESIDIO_N_PROCESS", default=max(1, min(8, _cpu_count() // 2)))


@lru_cache(maxsize=1)
def presidio_batch_size() -> int:
    raw = os.environ.get("PRESIDIO_BATCH_SIZE", "").strip()
    if raw:
        return max(1, int(raw))
    return 256


@lru_cache(maxsize=1)
def presidio_session_batch() -> int:
    """Curate: combined session texts per Presidio batch."""
    raw = os.environ.get("PRESIDIO_SESSION_BATCH", "").strip()
    if raw:
        return max(1, int(raw))
    return 128
