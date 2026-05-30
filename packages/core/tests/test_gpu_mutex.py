"""Unit tests for GPU competitor classification (no GPU required)."""

from __future__ import annotations

import os

from llm_core.gpu_mutex import classify_gpu_competitor, process_tree_pids


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
