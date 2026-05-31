"""Config profile merge tests."""

from __future__ import annotations

import os

from llm_core.yaml_config import load_yaml_config, profile_path


def test_profile_path_cloud_h100():
    p = profile_path("cloud-h100")
    assert p is not None
    assert p.name == "cloud-h100.yaml"


def test_cloud_h100_merges_training_mix(monkeypatch):
    monkeypatch.setenv("LLM_CONFIG_PROFILE", "cloud-h100")
    doc = load_yaml_config()
    assert doc.get("training_mix", {}).get("personal_ratio") == 0.75
    assert doc.get("gpu_mutex", {}).get("enabled") is False
