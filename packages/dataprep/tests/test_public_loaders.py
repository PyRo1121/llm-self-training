"""Smoke tests for public HF loaders (no network)."""

from __future__ import annotations

from llm_dataprep.public.loaders import _cooper_traj_path


def test_cooper_traj_path() -> None:
    assert _cooper_traj_path("coop/anyhow_task/390/f1_f2", "agent1_traj.json") == (
        "coop/anyhow_task/390/f1_f2/agent1_traj.json"
    )
    assert _cooper_traj_path("anyhow_task/390/f1_f2", "agent2_traj.json") == (
        "coop/anyhow_task/390/f1_f2/agent2_traj.json"
    )
