"""CLI — reclaim GPU VRAM before train (ghost contexts, competitors)."""

from __future__ import annotations

import argparse
import sys

from llm_core.gpu_mutex import (
    attempt_nvidia_gpu_reset,
    ensure_gpu_ready_for_train,
    format_gpu_competitors,
    gpu_ghost_entries,
    gpu_vram_recovery_instructions,
    kill_nvidia_smi_gpu_hogs,
    load_gpu_mutex_settings,
    reclaim_gpu_for_train,
    resolve_gpu_ghost_vram,
    _gpu_free_mib,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Free GPU VRAM for train-qlora (stop services, kill hogs, ghost recovery)"
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Only attempt nvidia-smi --gpu-reset (no service stops)",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Do not try sudo for gpu-reset",
    )
    args = parser.parse_args()
    cfg = load_gpu_mutex_settings()

    if args.reset_only:
        ok = attempt_nvidia_gpu_reset(
            int(cfg.get("gpu_index", 0)),
            use_sudo=not args.no_sudo,
        )
        sys.exit(0 if ok else 1)

    reclaim_gpu_for_train(cfg, reclaim_unknown=False)
    kill_nvidia_smi_gpu_hogs(min_mib=50)
    resolve_gpu_ghost_vram(cfg)

    ghosts = gpu_ghost_entries()
    free = _gpu_free_mib()
    print(f"GPU free: {free} MiB | compute: {format_gpu_competitors()}", file=sys.stderr)
    if ghosts:
        for pid, name, mem in ghosts:
            print(f"  ghost: pid={pid} {name} ({mem} MiB)", file=sys.stderr)
        print(gpu_vram_recovery_instructions(), file=sys.stderr)
        sys.exit(1)

    if free is not None and free >= int(cfg.get("min_free_vram_mib", 8000)):
        print("GPU ready for train.", file=sys.stderr)
        sys.exit(0)

    if ensure_gpu_ready_for_train(cfg, reclaim_unknown=False):
        print("GPU ready for train.", file=sys.stderr)
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()
