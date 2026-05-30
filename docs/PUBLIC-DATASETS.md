# Public Hugging Face datasets (May 2026)

Ingest into the **same** `data/raw` → `curate-raw` pipeline as personal agent logs. Policy unchanged: **secrets/PII only** — no topic/refusal filtering.

## Top 10 (Dec 2025 – May 2026)

Final public-ingest plan for `pyro-coder` bootstrap. Caps in `config/default.yaml` → `public_datasets.datasets.*.max_rows`.

| Rank | ID | Hugging Face | Released | Default cap | Gated | Why ingest |
|------|-----|--------------|----------|-------------|-------|------------|
| 1 | `ultradata_sft_2605` | [openbmb/UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605) | May 2026 | 50k | no | **Knowledge + Code + Math** configs (`no_think` split) |
| 2 | `swe_chat` | [SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) | Apr 2026 | 200k turn rows | **yes** | Real Cursor / Claude Code wild sessions + tool calls |
| 3 | `coderforge_preview` | [togethercomputer/CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) | Feb 2026 | 10k | no | Test-verified long-horizon agent trajectories (`filtered_reward1`) |
| 4 | `zen_agentic` | [zenlm/zen-agentic-dataset](https://huggingface.co/datasets/zenlm/zen-agentic-dataset) | May 2026 | 5k | no* | Hidden gem — real Claude Code + git history (~12B tokens) |
| 5 | `swe_next` | [TIGER-Lab/SWE-Next-SFT-Trajectories](https://huggingface.co/datasets/TIGER-Lab/SWE-Next-SFT-Trajectories) | Mar–Apr 2026 | all (~3.7k) | no | Execution-grounded SWE trajectories; `tool`→`user` |
| 6 | `high_coder_sft` | [Crownelius/High-Coder-SFT-Medium](https://huggingface.co/datasets/Crownelius/High-Coder-SFT-Medium) | May 2026 | 10k | no | 124k long-form synthetic code (prompt + full source file) |
| 7 | `agentic_sft_new` | [WaltonFuture/agentic-sft-new](https://huggingface.co/datasets/WaltonFuture/agentic-sft-new) | May 2026 | 10k | no | Merged agentic SFT — tools, edits, multi-hop |
| 8 | `agentic_cot_coding` | [mepartha/Agentic-Chain-of-Thought-Coding-SFT-Dataset-v1.1](https://huggingface.co/datasets/mepartha/Agentic-Chain-of-Thought-Coding-SFT-Dataset-v1.1) | 2026 | 10k | no | Agentic CoT coding traces (`[thinking]` prefix when present) |
| 9 | `nemotron_opencode` | [nvidia/Nemotron-SFT-OpenCode-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) | Mar 2026 | 10k | no | OpenCode-style tool-calling across 6 splits |
| 10 | `agent_trove` | [open-thoughts/AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) | Apr 2026 | 10k | no | 1.7M general agentic traces (coding-heavy subset) |

\* `zen_agentic`: HF repo is currently a **placeholder** (no public shard files). Loader fails fast with card instructions (`oss@hanzo.ai`). When shards land, use `llm-dataprep[zed]` for zstd JSONL streaming.

### Recommended ingest order

1. **Small + ungated first** (validate pipeline): `swe_next`, `high_coder_sft`, `nemotron_opencode`
2. **High-signal agentic**: `coderforge_preview`, `agentic_sft_new`, `agentic_cot_coding`, `agent_trove`
3. **Gated / large** (after `HF_TOKEN`): `swe_chat`, `ultradata_sft_2605`
4. **When available**: `zen_agentic`

### Gated datasets

```bash
# One-time login (token saved to ~/.cache/huggingface/token)
hf auth login
hf auth whoami

# Accept terms on each gated dataset card in the browser, then:
uv run --package llm-dataprep public-ingest --datasets swe_chat,ultradata_sft_2605

# Optional: export token for CI/other shells (overrides cached login)
export HF_TOKEN=hf_...
```

**Rate limits:** unauthenticated Hub requests hit low IP limits fast. After `hf auth login`, ingest uses your account quota. If you already hit 429, wait ~30–60 min or ingest one dataset at a time with `--datasets swe_next --max-rows 1000`.

## Legacy (disabled by default)

Pre–Top-10 bootstrap sets — still in registry, off in config:

| ID | Repo | Notes |
|----|------|-------|
| `opencode_broad` | `EER6/nvidia-OpenCodeInstruct-broad` | judge≥4, test≥0.8 |
| `opencode_refined` | `EER6/nvidia-OpenCodeInstruct-refined` | strict subset |
| `nemotron_swe` | `nvidia/Nemotron-Cascade-SFT-SWE` | Dec 2025 Cascade SWE |
| `self_code_align` | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | exec-filtered |
| `magicoder_75k` | `ise-uiuc/Magicoder-OSS-Instruct-75K` | diversity booster |

Optional extras (enabled in config, not Top 10):

| ID | Repo | Notes |
|----|------|-------|
| `codex_7m` | `Modotte/CodeX-7M-Non-Thinking` | 7.36M instruction pairs |
| `codex_2m_thinking` | `Modotte/CodeX-2M-Thinking` | 2.19M with reasoning |

## Commands

```bash
# List registry (Top 10 + legacy)
uv run --package llm-dataprep public-ingest --list

# Ingest all enabled sets (skips gated without HF_TOKEN)
uv run --package llm-dataprep public-ingest --skip-gated

# Single dataset
uv run --package llm-dataprep public-ingest --datasets coderforge_preview --max-rows 1000

# Then curate (public + personal raw — coordinate with other ingest jobs)
uv run --package llm-dataprep curate-raw --no-gitleaks --no-presidio

# Full phase1 with public + personal
uv run --package llm-dataprep phase1 --fresh-raw --public --include-subagents --repo /path/to/git/repo
```

## Training mix (PLAN)

- Mature target: **75–85% personal**, **15–25% public**
- Bootstrap: ingest Top 10 with conservative caps; raise caps after curation audit
- Public rows: `source=public`, `exec=pass` where verified (trajectories, tests, exec-filter)
- Filter aggressively in curation: style match, min chars, secrets/PII — not topic/refusal

## Licenses

Check each dataset card before redistribution. Ingest is local-only; do not commit raw HF rows to git.
