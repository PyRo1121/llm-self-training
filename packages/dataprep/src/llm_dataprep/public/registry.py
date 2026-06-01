"""Registry of public Hugging Face datasets (May 2026 — Top 10 + legacy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from llm_dataprep.public import loaders


@dataclass(frozen=True)
class PublicDatasetSpec:
    dataset_id: str
    hf_repo: str
    description: str
    license_note: str
    default_max_rows: int | None  # None = ingest all (small sets only)
    released: str
    gated: bool = False
    loader_name: str = ""
    tier: str = "top10"  # top10 | legacy

    @property
    def loader(self) -> Callable[..., Iterator[dict]]:
        fn = getattr(loaders, self.loader_name or f"load_{self.dataset_id}", None)
        if fn is None:
            raise ValueError(f"No loader for {self.dataset_id}")
        return fn


# Top 10 — Dec 2025–May 2026 agentic / code SFT (see docs/oss/PUBLIC-DATASETS.md)
PUBLIC_DATASETS: tuple[PublicDatasetSpec, ...] = (
    PublicDatasetSpec(
        dataset_id="swe_chat",
        hf_repo="SALT-NLP/SWE-chat",
        description="Real wild agent sessions (Apr 2026). conversations.parquet.",
        license_note="Gated — accept terms on HF + HF_TOKEN",
        default_max_rows=None,
        released="2026-04",
        gated=True,
        loader_name="load_swe_chat",
    ),
    PublicDatasetSpec(
        dataset_id="coderforge_preview",
        hf_repo="togethercomputer/CoderForge-Preview",
        description="826k test-verified agent trajectories (filtered_reward1 split).",
        license_note="See Together dataset card",
        default_max_rows=10_000,
        released="2026-02",
        loader_name="load_coderforge_preview",
    ),
    PublicDatasetSpec(
        dataset_id="zen_agentic",
        hf_repo="zenlm/zen-agentic-dataset",
        description="~12B tokens real Claude Code + git history (zstd JSONL shards).",
        license_note="See Zen dataset card; install llm-dataprep[zed] for zstd",
        default_max_rows=5_000,
        released="2026-05",
        loader_name="load_zen_agentic",
    ),
    PublicDatasetSpec(
        dataset_id="swe_next",
        hf_repo="TIGER-Lab/SWE-Next-SFT-Trajectories",
        description="3.7k execution-grounded SWE agent trajectories (ShareGPT messages).",
        license_note="See dataset card",
        default_max_rows=None,
        released="2026-03",
        loader_name="load_swe_next",
    ),
    PublicDatasetSpec(
        dataset_id="swe_zero_12m",
        hf_repo="AlienKevin/SWE-ZERO-12M-trajectories",
        description="12M SWE-Zero mini-swe-agent trajectories (messages[]; filter exit_status).",
        license_note="See AlienKevin dataset card",
        default_max_rows=None,
        released="2026",
        loader_name="load_swe_zero_12m",
    ),
    PublicDatasetSpec(
        dataset_id="swe_zero_openhands",
        hf_repo="nvidia/SWE-Zero-openhands-trajectories",
        description="NVIDIA SWE-Zero OpenHands trajectories (tool calls + model_patch).",
        license_note="CC-BY-4.0; see NVIDIA dataset card",
        default_max_rows=None,
        released="2026",
        loader_name="load_swe_zero_openhands",
    ),
    PublicDatasetSpec(
        dataset_id="agentic_sft_new",
        hf_repo="WaltonFuture/agentic-sft-new",
        description="Merged agentic SFT — tool calling, code editing, multi-hop.",
        license_note="See dataset card",
        default_max_rows=10_000,
        released="2026-05",
        loader_name="load_agentic_sft_new",
    ),
    PublicDatasetSpec(
        dataset_id="agentic_cot_coding",
        hf_repo="mepartha/Agentic-Chain-of-Thought-Coding-SFT-Dataset-v1.1",
        description="Agentic CoT reasoning traces on coding tasks.",
        license_note="See dataset card",
        default_max_rows=10_000,
        released="2026",
        loader_name="load_agentic_cot_coding",
    ),
    PublicDatasetSpec(
        dataset_id="nemotron_opencode",
        hf_repo="nvidia/Nemotron-SFT-OpenCode-v1",
        description="459k agentic OpenCode-style tool-calling rows (6 JSONL splits).",
        license_note="CC-BY-4.0",
        default_max_rows=10_000,
        released="2026-03",
        loader_name="load_nemotron_opencode",
    ),
    # Legacy — pre-Top-10 bootstrap sets (disabled by default in config)
    PublicDatasetSpec(
        dataset_id="opencode_refined",
        hf_repo="EER6/nvidia-OpenCodeInstruct-refined",
        description="Strict OpenCodeInstruct subset (judge=5, test=1.0). ~445k.",
        license_note="Apache-2.0",
        default_max_rows=5_000,
        released="2026-04",
        loader_name="load_opencode_refined",
        tier="legacy",
    ),
    PublicDatasetSpec(
        dataset_id="ling_coder_sft",
        hf_repo="inclusionAI/Ling-Coder-SFT",
        description="Ling-Coder ShareGPT SFT (messages, multi-language).",
        license_note="See inclusionAI dataset card",
        default_max_rows=None,
        released="2026",
        loader_name="load_ling_coder_sft",
        tier="legacy",
    ),
    PublicDatasetSpec(
        dataset_id="nemotron_swe_v2",
        hf_repo="nvidia/Nemotron-SFT-SWE-v2",
        description="NVIDIA SWE SFT v2 (agentless split; tool→user).",
        license_note="CC-BY-4.0; openhands_swe not streamable on HF",
        default_max_rows=None,
        released="2026",
        loader_name="load_nemotron_swe_v2",
        tier="legacy",
    ),
    PublicDatasetSpec(
        dataset_id="scale_swe",
        hf_repo="AweAI-Team/Scale-SWE",
        description="Scale-SWE problem_statement + patch (SWE-bench style).",
        license_note="See AweAI-Team dataset card",
        default_max_rows=None,
        released="2026",
        loader_name="load_scale_swe",
        tier="legacy",
    ),
    PublicDatasetSpec(
        dataset_id="cooper_qwen9b_coop_claude",
        hf_repo="CooperBench/qwen9b-coop-claude-code",
        description="CooperBench two-agent coop trajectories (Claude Code, Qwen3.5-9B).",
        license_note="See CooperBench dataset card; ~368 pairs",
        default_max_rows=None,
        released="2026",
        loader_name="load_cooper_qwen9b_coop_claude",
        tier="top10",
    ),
)


def get_spec(dataset_id: str) -> PublicDatasetSpec:
    for spec in PUBLIC_DATASETS:
        if spec.dataset_id == dataset_id:
            return spec
    raise KeyError(dataset_id)


def list_specs(*, tier: str | None = None) -> tuple[PublicDatasetSpec, ...]:
    if tier is None:
        return PUBLIC_DATASETS
    return tuple(s for s in PUBLIC_DATASETS if s.tier == tier)
