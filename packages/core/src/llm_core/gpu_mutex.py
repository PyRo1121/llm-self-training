"""Release GPU VRAM from desktop inference and other projects before train.

Stops configured systemd units (hyprwhspr, etc.), Ollama, then SIGTERM/SIGKILL
GPU compute PIDs that match kill patterns or unknown hogs above a MiB threshold.
Never kills the current train process tree.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import yaml

from llm_core.paths import config_dir

HYPRWSPR_UNIT = "hyprwhspr.service"
HYPRWSPR_GPU_MIN_FREE_MIB = 8_000
HYPRWSPR_BLOCK_DROPIN = (
    Path.home() / ".config/systemd/user/hyprwhspr.service.d/llm-train-gpu-mutex.conf"
)

DEFAULT_KILL_SUBSTRINGS = [
    "hyprwhspr",
    "ollama",
    "llama-server",
    "llama.cpp",
    "vllm",
    "text-generation",
    "oobabooga",
    "comfyui",
    "comfy",
    "invokeai",
    "automatic1111",
    "stable-diffusion",
    "whisper",
    "faster-whisper",
    "train-qlora",
    "accelerate",
    "deepspeed",
    "torchrun",
    "transformers-cli",
    "jupyter",
    "marimo",
    "radar",  # sibling GPU apps on this machine
    "llm-train",
    "llm_rag",
    "chronicals",
]

DEFAULT_PROTECT_SUBSTRINGS = [
    "Xorg",
    "Xwayland",
    "wayland",
    "gnome-shell",
    "kwin",
    "hyprland",
    "sway",
    "nvidia-settings",
    "nvidia-persistenced",
    "picom",
    "gamescope",
    "cursor",  # IDE GPU preview — usually small; protect unless huge hog mode
]

DEFAULT_SYSTEMD_UNITS = [HYPRWSPR_UNIT]


def _log(msg: str, *, stream: TextIO = sys.stderr) -> None:
    print(msg, file=stream, flush=True)


def load_gpu_mutex_settings() -> dict[str, Any]:
    path = config_dir() / "default.yaml"
    doc: dict[str, Any] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    g = doc.get("gpu_mutex") or {}
    return {
        "enabled": bool(g.get("enabled", True)),
        "min_free_vram_mib": int(g.get("min_free_vram_mib", HYPRWSPR_GPU_MIN_FREE_MIB)),
        "min_competitor_mib": int(g.get("min_competitor_mib", 250)),
        "reclaim_unknown_hogs": bool(g.get("reclaim_unknown_hogs", False)),
        "kill_process_substrings": list(
            g.get("kill_process_substrings") or DEFAULT_KILL_SUBSTRINGS
        ),
        "protect_process_substrings": list(
            g.get("protect_process_substrings") or DEFAULT_PROTECT_SUBSTRINGS
        ),
        "stop_systemd_units": list(g.get("stop_systemd_units") or DEFAULT_SYSTEMD_UNITS),
        "stop_ollama": bool(g.get("stop_ollama", True)),
        "restore_hyprwhspr": bool(g.get("restore_hyprwhspr", True)),
        "kill_grace_s": float(g.get("kill_grace_s", 2.0)),
        "ghost_vram_threshold_mib": int(g.get("ghost_vram_threshold_mib", 1000)),
        "ghost_wait_timeout_s": float(g.get("ghost_wait_timeout_s", 12.0)),
        "attempt_gpu_reset": bool(g.get("attempt_gpu_reset", True)),
        "gpu_reset_use_sudo": bool(g.get("gpu_reset_use_sudo", True)),
        "gpu_index": int(g.get("gpu_index", 0)),
    }


def _systemctl_user(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def hyprwhspr_is_active() -> bool:
    if not shutil.which("systemctl"):
        return False
    r = _systemctl_user("is-active", HYPRWSPR_UNIT)
    return r.returncode == 0 and r.stdout.strip() == "active"


def _gpu_free_mib() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        return None
    line = r.stdout.strip().splitlines()
    return int(line[0]) if line else None


def gpu_compute_processes() -> list[tuple[int, str, int]]:
    """Return [(pid, process_name, used_mib), ...] for GPU compute clients."""
    if not shutil.which("nvidia-smi"):
        return []
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return []
    rows: list[tuple[int, str, int]] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[0]), parts[1], int(parts[2])))
        except ValueError:
            continue
    return rows


def _proc_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _pid_alive(pid: int) -> bool:
    if not _proc_exists(pid):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _ppid_map() -> dict[int, int]:
    out: dict[int, int] = {}
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("PPid:"):
                    out[int(entry.name)] = int(line.split()[1])
                    break
        except OSError:
            continue
    return out


def process_tree_pids(root_pid: int) -> set[int]:
    """Root PID plus all descendants (train workers)."""
    ppid_map = _ppid_map()
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in ppid_map.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def _matches_substrings(haystack: str, patterns: list[str]) -> bool:
    low = haystack.lower()
    return any(p.lower() in low for p in patterns if p)


def kill_stale_llm_train_gpu_processes(*, root_pid: int | None = None) -> int:
    """SIGTERM other train-qlora / repo venv PIDs (leftover after SIGTERM runs)."""
    root = root_pid or os.getpid()
    protected = process_tree_pids(root)
    killed = 0
    repo_marker = "LLM Self Training"
    for pid, name, mem in gpu_compute_processes():
        if pid in protected:
            continue
        cmd = _cmdline(pid)
        low = f"{name} {cmd}".lower()
        if "train-qlora" not in low and "llm_train" not in low and repo_marker not in cmd:
            continue
        if mem < 50:
            continue
        _log(f"GPU reclaim: stale train pid={pid} ({mem} MiB) cmd={cmd[:100]!r}")
        if _terminate_pid(pid, grace_s=1.0):
            killed += 1
    return killed


def classify_gpu_competitor(
    *,
    pid: int,
    name: str,
    mem_mib: int,
    cmdline: str,
    protected_pids: set[int],
    settings: dict[str, Any],
) -> str:
    """Return: skip | protect | kill | warn."""
    if pid in protected_pids:
        return "skip"
    if mem_mib < int(settings["min_competitor_mib"]):
        return "skip"

    hay = f"{name} {cmdline}"
    if _matches_substrings(hay, settings["protect_process_substrings"]):
        return "protect"

    if _matches_substrings(hay, settings["kill_process_substrings"]):
        return "kill"

    if settings.get("reclaim_unknown_hogs"):
        return "kill"

    return "warn"


def _terminate_pid(pid: int, *, grace_s: float) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except OSError:
        return False

    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


def gpu_ghost_entries() -> list[tuple[int, str, int]]:
    """Dead PIDs still holding VRAM according to nvidia-smi."""
    ghosts: list[tuple[int, str, int]] = []
    for pid, name, mem in gpu_compute_processes():
        if not _pid_alive(pid):
            ghosts.append((pid, name, mem))
    return ghosts


def kill_nvidia_smi_gpu_hogs(
    *,
    root_pid: int | None = None,
    min_mib: int = 200,
) -> int:
    """SIGKILL any alive GPU compute PID above min_mib (incl. nvidia [Not Found] rows)."""
    root = root_pid or os.getpid()
    protected = process_tree_pids(root)
    killed = 0
    for pid, name, mem in gpu_compute_processes():
        if pid in protected or mem < min_mib:
            continue
        if not _proc_exists(pid):
            continue
        cmd = _cmdline(pid)
        _log(
            f"GPU reclaim: SIGKILL pid={pid} {name} ({mem} MiB) "
            f"cmd={cmd[:120]!r}"
        )
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except OSError as exc:
            _log(f"GPU reclaim: could not kill pid={pid}: {exc}")
    if killed:
        time.sleep(2)
    return killed


def attempt_nvidia_gpu_reset(
    gpu_index: int = 0,
    *,
    use_sudo: bool = True,
) -> bool:
    """Try nvidia-smi --gpu-reset (often blocked on primary/display GPU)."""
    if not shutil.which("nvidia-smi"):
        return False

    def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False)

    for prefix in ([], ["sudo", "-n"] if use_sudo else [], ["sudo"] if use_sudo else []):
        if prefix == ["sudo"] and not shutil.which("sudo"):
            continue
        cmd = [*prefix, "nvidia-smi", "--gpu-reset", "-i", str(gpu_index)]
        r = _run(cmd)
        if r.returncode == 0:
            _log(f"GPU reset OK ({' '.join(cmd)})")
            time.sleep(3)
            return True
        err = (r.stderr or r.stdout or "").strip()
        if "primary GPU" in err or "primary gpu" in err.lower():
            _log(
                "GPU reset blocked: 4070 Ti is primary display GPU. "
                "Log out/in or reboot to clear ghost VRAM, or close all GPU apps and retry."
            )
            return False
        if prefix == ["sudo"] and err:
            _log(f"GPU reset failed ({' '.join(cmd)}): {err[:300]}")
    return False


_last_throttled_log = 0.0


def _log_throttled(msg: str, *, interval_s: float = 5.0) -> None:
    global _last_throttled_log
    now = time.monotonic()
    if now - _last_throttled_log >= interval_s:
        _log(msg)
        _last_throttled_log = now


def resolve_gpu_ghost_vram(settings: dict[str, Any] | None = None) -> bool:
    """Wait briefly, kill alive hogs, optional gpu-reset. True if ghost MiB is gone."""
    cfg = settings or load_gpu_mutex_settings()
    threshold = int(cfg.get("ghost_vram_threshold_mib", 1000))
    wait_s = float(cfg.get("ghost_wait_timeout_s", 12.0))
    gpu_idx = int(cfg.get("gpu_index", 0))

    ghosts = gpu_ghost_entries()
    ghost_mib = sum(m for _, _, m in ghosts)
    if ghost_mib < threshold:
        return True

    for pid, name, mem in ghosts:
        _log(f"GPU ghost: pid={pid} {name} ({mem} MiB) — driver has not released VRAM")

    kill_nvidia_smi_gpu_hogs(min_mib=threshold)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        ghost_mib = sum(m for _, _, m in gpu_ghost_entries())
        if ghost_mib < threshold:
            return True
        _log_throttled(f"GPU ghost VRAM ~{ghost_mib} MiB — waiting for driver…")
        time.sleep(2)

    ghost_mib = sum(m for _, _, m in gpu_ghost_entries())
    if ghost_mib < threshold:
        return True

    if cfg.get("attempt_gpu_reset", True):
        _log("Attempting nvidia-smi --gpu-reset to clear ghost VRAM…")
        if attempt_nvidia_gpu_reset(
            gpu_idx, use_sudo=bool(cfg.get("gpu_reset_use_sudo", True))
        ):
            time.sleep(3)
            ghost_mib = sum(m for _, _, m in gpu_ghost_entries())
            if ghost_mib < threshold:
                return True

    return sum(m for _, _, m in gpu_ghost_entries()) < threshold


def gpu_vram_recovery_instructions() -> str:
    return (
        "Clear ghost VRAM (stuck after crashed train):\n"
        "  1. uv run --package llm-core clear-gpu-vram\n"
        "  2. If still stuck: log out and back in, or reboot\n"
        "  3. Optional (display GPU): close Hyprland session, then:\n"
        "       sudo nvidia-smi --gpu-reset -i 0\n"
        "  Check: nvidia-smi"
    )


def ensure_gpu_ready_for_train(
    settings: dict[str, Any] | None = None,
    *,
    root_pid: int | None = None,
    warn_only: bool = False,
    reclaim_unknown: bool | None = None,
) -> bool:
    """Full reclaim + ghost resolve. False = not enough VRAM to start train."""
    cfg = settings or load_gpu_mutex_settings()
    if not cfg.get("enabled", True):
        return True

    reclaim_gpu_for_train(
        cfg,
        root_pid=root_pid,
        warn_only=warn_only,
        reclaim_unknown=reclaim_unknown,
    )
    if warn_only:
        return True

    kill_nvidia_smi_gpu_hogs(root_pid=root_pid, min_mib=200)
    resolve_gpu_ghost_vram(cfg)

    min_mib = int(cfg.get("min_free_vram_mib", HYPRWSPR_GPU_MIN_FREE_MIB))
    if wait_for_train_vram(min_free_mib=min_mib, timeout_s=20.0):
        return True

    free = _gpu_free_mib()
    _log(
        f"Cannot start train: {free} MiB free, need {min_mib} MiB. "
        f"On GPU: {format_gpu_competitors()}"
    )
    _log(gpu_vram_recovery_instructions())
    return False


def kill_gpu_competitors(
    settings: dict[str, Any],
    *,
    root_pid: int | None = None,
    warn_only: bool = False,
    reclaim_unknown: bool | None = None,
) -> list[tuple[int, str, int, str]]:
    """Stop/kill foreign GPU processes. Returns [(pid, name, mib, action_taken), ...]."""
    root = root_pid or os.getpid()
    protected = process_tree_pids(root)
    effective = {**settings}
    if reclaim_unknown is not None:
        effective["reclaim_unknown_hogs"] = reclaim_unknown

    actions: list[tuple[int, str, int, str]] = []
    grace = float(effective.get("kill_grace_s", 2.0))

    for pid, name, mem in gpu_compute_processes():
        if not _pid_alive(pid):
            continue
        cmd = _cmdline(pid)
        decision = classify_gpu_competitor(
            pid=pid,
            name=name,
            mem_mib=mem,
            cmdline=cmd,
            protected_pids=protected,
            settings=effective,
        )
        if decision == "skip":
            continue
        if decision == "protect":
            _log(f"GPU protect: pid={pid} {name} ({mem} MiB) — left running")
            continue
        if decision == "warn":
            _log(
                f"GPU competitor: pid={pid} {name} ({mem} MiB) "
                f"cmd={cmd[:120]!r} — add to gpu_mutex.kill_process_substrings or enable reclaim_unknown_hogs"
            )
            actions.append((pid, name, mem, "warn"))
            continue

        if warn_only:
            _log(
                f"GPU would kill: pid={pid} {name} ({mem} MiB) cmd={cmd[:120]!r} (warn-only mode)"
            )
            actions.append((pid, name, mem, "would_kill"))
            continue

        _log(f"GPU reclaim: SIGTERM pid={pid} {name} ({mem} MiB) cmd={cmd[:120]!r}")
        if _terminate_pid(pid, grace_s=grace):
            actions.append((pid, name, mem, "killed"))
        else:
            actions.append((pid, name, mem, "failed"))
            _log(f"GPU reclaim: could not terminate pid={pid}")

    return actions


def stop_systemd_units(units: list[str], *, block_hyprwhspr: bool = True) -> list[str]:
    """Best-effort stop user units before killing GPU PIDs."""
    if not shutil.which("systemctl"):
        return []
    stopped: list[str] = []
    for unit in units:
        if not unit:
            continue
        r = _systemctl_user("is-active", unit)
        if r.returncode != 0 or r.stdout.strip() != "active":
            continue
        r = _systemctl_user("stop", unit)
        if r.returncode == 0:
            _log(f"Stopped systemd unit {unit} (VRAM reclaim)")
            stopped.append(unit)
        else:
            _log(
                f"systemctl stop {unit} failed: "
                f"{r.stderr.strip() or r.stdout.strip()}"
            )
    if block_hyprwhspr and HYPRWSPR_UNIT in units:
        block_hyprwhspr_restarts()
    return stopped


def block_hyprwhspr_restarts() -> bool:
    if not shutil.which("systemctl"):
        return False
    HYPRWSPR_BLOCK_DROPIN.parent.mkdir(parents=True, exist_ok=True)
    HYPRWSPR_BLOCK_DROPIN.write_text(
        "# llm_core.gpu_mutex — auto-removed when train ends\n"
        "[Unit]\n"
        "RefuseManualStart=yes\n"
        "[Service]\n"
        "Restart=no\n",
        encoding="utf-8",
    )
    r = _systemctl_user("daemon-reload")
    if r.returncode != 0:
        _log(f"hyprwhspr daemon-reload failed: {r.stderr.strip() or r.stdout.strip()}")
        return False
    return True


def unblock_hyprwhspr() -> None:
    if not shutil.which("systemctl"):
        return
    if HYPRWSPR_BLOCK_DROPIN.is_file():
        HYPRWSPR_BLOCK_DROPIN.unlink()
        _systemctl_user("daemon-reload")


def stop_hyprwhspr(*, block_restarts: bool = True) -> bool:
    """Backward-compatible hyprwhspr stop."""
    was_active = hyprwhspr_is_active()
    stop_systemd_units([HYPRWSPR_UNIT], block_hyprwhspr=block_restarts)
    kill_gpu_competitors(
        load_gpu_mutex_settings(),
        root_pid=os.getpid(),
        reclaim_unknown=False,
    )
    return was_active


def start_hyprwhspr() -> None:
    if not shutil.which("systemctl"):
        return
    unblock_hyprwhspr()
    r = _systemctl_user("start", HYPRWSPR_UNIT)
    if r.returncode != 0:
        _log(f"hyprwhspr start failed: {r.stderr.strip() or r.stdout.strip()}")
    else:
        _log("Restarted hyprwhspr.service")


def stop_ollama() -> None:
    if not shutil.which("ollama"):
        return
    subprocess.run(["ollama", "stop"], capture_output=True, check=False)
    _log("Stopped Ollama models (VRAM reclaim)")


def reclaim_gpu_for_train(
    settings: dict[str, Any] | None = None,
    *,
    root_pid: int | None = None,
    warn_only: bool = False,
    reclaim_unknown: bool | None = None,
) -> None:
    """Full reclaim pipeline: systemd → Ollama → GPU PID kill."""
    cfg = settings or load_gpu_mutex_settings()
    if not cfg.get("enabled", True):
        return

    os.environ["LLM_TRAIN_ACTIVE"] = "1"
    units = list(cfg.get("stop_systemd_units") or [])
    stop_systemd_units(units, block_hyprwhspr=HYPRWSPR_UNIT in units)

    if cfg.get("stop_ollama", True):
        stop_ollama()

    n_stale = kill_stale_llm_train_gpu_processes(root_pid=root_pid)
    if n_stale:
        time.sleep(2)

    kill_gpu_competitors(
        cfg,
        root_pid=root_pid,
        warn_only=warn_only,
        reclaim_unknown=reclaim_unknown,
    )


def _gpu_ghost_mib() -> int:
    """VRAM held by dead PIDs still listed in nvidia-smi."""
    total = 0
    for pid, _name, mem in gpu_compute_processes():
        if not _pid_alive(pid):
            total += mem
    return total


def wait_for_train_vram(
    *,
    min_free_mib: int = HYPRWSPR_GPU_MIN_FREE_MIB,
    timeout_s: float = 30.0,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        free = _gpu_free_mib()
        ghosts = _gpu_ghost_mib()
        if ghosts >= 500:
            _log_throttled(f"GPU ghost VRAM ~{ghosts} MiB — waiting…")
        if free is not None and free >= min_free_mib and ghosts < 500:
            return True
        time.sleep(2)
    return False


def reclaim_gpu_before_load(
    *,
    min_free_mib: int | None = None,
    timeout_s: float = 45.0,
    settings: dict[str, Any] | None = None,
) -> bool:
    """Re-reclaim if competitors respawned during HF download."""
    cfg = settings or load_gpu_mutex_settings()
    min_mib = min_free_mib if min_free_mib is not None else int(
        cfg.get("min_free_vram_mib", HYPRWSPR_GPU_MIN_FREE_MIB)
    )
    reclaim_gpu_for_train(cfg, root_pid=os.getpid())
    resolve_gpu_ghost_vram(cfg)
    return wait_for_train_vram(min_free_mib=min_mib, timeout_s=timeout_s)


def warn_foreign_gpu_processes(*, exclude_pid: int | None = None) -> None:
    """Log-only pass (no kills)."""
    reclaim_gpu_for_train(
        load_gpu_mutex_settings(),
        root_pid=exclude_pid or os.getpid(),
        warn_only=True,
        reclaim_unknown=False,
    )


def format_gpu_competitors() -> str:
    procs = gpu_compute_processes()
    if not procs:
        return "none"
    return "; ".join(f"pid={pid} {name} ({mem} MiB)" for pid, name, mem in procs)


class GpuMutex:
    """Stop services + kill/throttle GPU competitors; restore hyprwhspr on exit."""

    def __init__(
        self,
        *,
        settings: dict[str, Any] | None = None,
        enabled: bool = True,
        stop_hyprwhspr_service: bool = True,
        stop_ollama_models: bool = True,
        restore_hyprwhspr: bool | None = None,
        min_free_vram_mib: int | None = None,
        warn_only: bool = False,
        reclaim_unknown: bool | None = None,
    ) -> None:
        self._cfg = settings or load_gpu_mutex_settings()
        self._enabled = enabled and bool(self._cfg.get("enabled", True))
        self._stop_hyprwhspr = stop_hyprwhspr_service
        self._stop_ollama = stop_ollama_models
        self._restore_hyprwhspr = (
            restore_hyprwhspr
            if restore_hyprwhspr is not None
            else bool(self._cfg.get("restore_hyprwhspr", True))
        )
        self._min_free_vram_mib = min_free_vram_mib or int(
            self._cfg.get("min_free_vram_mib", HYPRWSPR_GPU_MIN_FREE_MIB)
        )
        self._warn_only = warn_only
        self._reclaim_unknown = reclaim_unknown
        self._had_hyprwhspr = False
        self._blocked = False

    def __enter__(self) -> GpuMutex:
        if not self._enabled:
            return self

        cfg = dict(self._cfg)
        if not self._stop_hyprwhspr:
            units = [u for u in cfg.get("stop_systemd_units", []) if u != HYPRWSPR_UNIT]
            cfg["stop_systemd_units"] = units
        if not self._stop_ollama:
            cfg["stop_ollama"] = False

        self._had_hyprwhspr = hyprwhspr_is_active()
        if not self._warn_only and not ensure_gpu_ready_for_train(
            cfg,
            root_pid=os.getpid(),
            warn_only=False,
            reclaim_unknown=self._reclaim_unknown,
        ):
            raise SystemExit(1)
        self._blocked = HYPRWSPR_BLOCK_DROPIN.is_file()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._enabled:
            return
        if self._blocked or HYPRWSPR_BLOCK_DROPIN.is_file():
            unblock_hyprwhspr()
        if self._restore_hyprwhspr and self._had_hyprwhspr:
            if exc_type is not None:
                _log("Train failed — restoring hyprwhspr.service")
            else:
                _log("Train finished — restoring hyprwhspr.service")
            start_hyprwhspr()
        if os.environ.get("LLM_TRAIN_ACTIVE"):
            os.environ.pop("LLM_TRAIN_ACTIVE", None)
