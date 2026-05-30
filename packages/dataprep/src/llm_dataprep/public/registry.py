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


# Top 10 — Dec 2025–May 2026 agentic / code SFT (see docs/PUBLIC-DATASETS.md)
PUBLIC_DATASETS: tuple[PublicDatasetSpec, ...] = (
    PublicDatasetSpec(
        dataset_id="ultradata_sft_2605",
        hf_repo="openbmb/UltraData-SFT-2605",
        description="15M+ SFT; loads Knowledge + Code + Math configs (no_think split).",
        license_note="Accept terms on HF if prompted; hf auth login",
        default_max_rows=50_000,
        released="2026-05",
        gated=False,
        loader_name="load_ultradata_sft_2605",
    ),
    PublicDatasetSpec(
        dataset_id="swe_chat",
        hf_repo="SALT-NLP/SWE-chat",
        description="Real wild agent sessions (Apr 2026). conversations.parquet.",
        license_note="Gated — accept terms on HF + HF_TOKEN",
        default_max_rows=200_000,
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
        dataset_id="high_coder_sft",
        hf_repo="Crownelius/High-Coder-SFT-Medium",
        description="124k long-form synthetic code (prompt + source file).",
        license_note="MIT",
        default_max_rows=10_000,
        released="2026-05",
        loader_name="load_high_coder_sft",
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
    PublicDatasetSpec(
        dataset_id="agent_trove",
        hf_repo="open-thoughts/AgentTrove",
        description="1.7M agentic traces (OpenThoughts; strong coding component).",
        license_note="See OpenThoughts license",
        default_max_rows=10_000,
        released="2026-04",
        loader_name="load_agent_trove",
    ),
    # Legacy — pre-Top-10 bootstrap sets (disabled by default in config)
    PublicDatasetSpec(
        dataset_id="opencode_broad",
        hf_repo="EER6/nvidia-OpenCodeInstruct-broad",
        description="OpenCodeInstruct filtered (judge≥4, test≥0.8). ~1.7M rows.",
        license_note="Apache-2.0",
        default_max_rows=10_000,
        released="2026-04",
        loader_name="load_opencode_broad",
        tier="legacy",
    ),
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
        dataset_id="nemotron_swe",
        hf_repo="nvidia/Nemotron-Cascade-SFT-SWE",
        description="NVIDIA Nemotron Cascade SWE SFT (Dec 2025).",
        license_note="See NVIDIA dataset license",
        default_max_rows=10_000,
        released="2025-12",
        loader_name="load_nemotron_swe",
        tier="legacy",
    ),
    PublicDatasetSpec(
        dataset_id="self_code_align",
        hf_repo="bigcode/self-oss-instruct-sc2-exec-filter-50k",
        description="SelfCodeAlign exec-filtered (~3k high-signal).",
        license_note="See bigcode card",
        default_max_rows=None,
        released="2024",
        loader_name="load_self_code_align",
        tier="legacy",
    ),
    PublicDatasetSpec(
        dataset_id="magicoder_75k",
        hf_repo="ise-uiuc/Magicoder-OSS-Instruct-75K",
        description="Magicoder OSS-Instruct 75k (classic diversity booster).",
        license_note="See dataset card",
        default_max_rows=10_000,
        released="2023",
        loader_name="load_magicoder",
        tier="legacy",
    ),
    PublicDatasetSpec(
        dataset_id="codex_7m",
        hf_repo="Modotte/CodeX-7M-Non-Thinking",
        description="7.36M curated coding instruction pairs (non-thinking).",
        license_note="See dataset card",
        default_max_rows=10_000,
        released="2026-02",
        loader_name="load_codex_7m",
        tier="legacy",
    ),
    PublicDatasetSpec(
        dataset_id="codex_2m_thinking",
        hf_repo="Modotte/CodeX-2M-Thinking",
        description="2.19M coding pairs with reasoning traces.",
        license_note="See dataset card",
        default_max_rows=5_000,
        released="2026-02",
        loader_name="load_codex_2m_thinking",
        tier="legacy",
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
