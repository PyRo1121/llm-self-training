"""Unit tests for GPU competitor classification (no GPU required)."""

from __future__ import annotations

import os
import signal
from unittest.mock import patch

from llm_core.gpu_mutex import (
    classify_gpu_competitor,
    kill_nvidia_smi_gpu_hogs,
    process_tree_pids,
)


def test_classify_kill_pattern():
    settings = {
        "min_competitor_mib": 100,
        "kill_process_substrings": ["ollama"],
        "protect_process_substrings": ["Xorg"],
        "reclaim_unknown_hogs": False,
    }
    assert (
        classify_gpu_competitor(
            pid=999,
            name="ollama",
            mem_mib=500,
            cmdline="/usr/bin/ollama serve",
            protected_pids=set(),
            settings=settings,
        )
        == "kill"
    )


def test_classify_protect_desktop():
    settings = {
        "min_competitor_mib": 100,
        "kill_process_substrings": ["python"],
        "protect_process_substrings": ["Xorg"],
        "reclaim_unknown_hogs": True,
    }
    assert (
        classify_gpu_competitor(
            pid=1,
            name="Xorg",
            mem_mib=800,
            cmdline="",
            protected_pids=set(),
            settings=settings,
        )
        == "protect"
    )


def test_classify_unknown_hog():
    settings = {
        "min_competitor_mib": 200,
        "kill_process_substrings": [],
        "protect_process_substrings": [],
        "reclaim_unknown_hogs": True,
    }
    assert (
        classify_gpu_competitor(
            pid=42,
            name="python3",
            mem_mib=6000,
            cmdline="/home/u/other-project/train.py",
            protected_pids=set(),
            settings=settings,
        )
        == "kill"
    )


def test_classify_skips_protected_tree():
    settings = {
        "min_competitor_mib": 100,
        "kill_process_substrings": ["python"],
        "protect_process_substrings": [],
        "reclaim_unknown_hogs": True,
    }
    assert (
        classify_gpu_competitor(
            pid=100,
            name="python3",
            mem_mib=5000,
            cmdline="train-qlora",
            protected_pids={100},
            settings=settings,
        )
        == "skip"
    )


def test_process_tree_includes_children():
    root = os.getpid()
    tree = process_tree_pids(root)
    assert root in tree


def test_kill_nvidia_smi_gpu_hogs_respects_classify():
    settings = {
        "min_competitor_mib": 100,
        "kill_process_substrings": ["ollama"],
        "protect_process_substrings": ["Xorg"],
        "reclaim_unknown_hogs": False,
    }
    processes = [
        (111, "ollama", 500),
        (222, "Xorg", 900),
        (333, "python3", 6000),
    ]

    with (
        patch("llm_core.gpu_mutex.gpu_compute_processes", return_value=processes),
        patch("llm_core.gpu_mutex.process_tree_pids", return_value={999}),
        patch("llm_core.gpu_mutex._proc_exists", return_value=True),
        patch("llm_core.gpu_mutex._cmdline", side_effect=lambda pid: f"cmd-{pid}"),
        patch("llm_core.gpu_mutex.os.kill") as mock_kill,
        patch("llm_core.gpu_mutex.time.sleep"),
    ):
        killed = kill_nvidia_smi_gpu_hogs(
            root_pid=999,
            min_mib=100,
            settings=settings,
        )

    assert killed == 1
    mock_kill.assert_called_once_with(111, signal.SIGKILL)


def test_kill_nvidia_smi_gpu_hogs_skips_protected_tree():
    settings = {
        "min_competitor_mib": 100,
        "kill_process_substrings": ["python"],
        "protect_process_substrings": [],
        "reclaim_unknown_hogs": True,
    }
    processes = [(100, "python3", 5000,)]

    with (
        patch("llm_core.gpu_mutex.gpu_compute_processes", return_value=processes),
        patch("llm_core.gpu_mutex.process_tree_pids", return_value={100}),
        patch("llm_core.gpu_mutex._proc_exists", return_value=True),
        patch("llm_core.gpu_mutex._cmdline", return_value="train-qlora"),
        patch("llm_core.gpu_mutex.os.kill") as mock_kill,
        patch("llm_core.gpu_mutex.time.sleep"),
    ):
        killed = kill_nvidia_smi_gpu_hogs(
            root_pid=100,
            min_mib=100,
            settings=settings,
        )

    assert killed == 0
    mock_kill.assert_not_called()
