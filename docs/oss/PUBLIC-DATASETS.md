# Public Hugging Face datasets (May 2026)

Ingest into the **same** `data/raw` → `curate-raw` pipeline as personal agent logs. Policy unchanged: **secrets/PII only** — no topic/refusal filtering.

## GitHub public harvest (true-data supplement)

Find **real** agent session JSONL on public GitHub (Cursor transcripts, Codex rollouts, Copilot events, etc.) via code search — not synthetic HF bulk.

**Setup:** add `GITHUB_TOKEN` to `.env` (fine-grained PAT with public read / Contents read-only). Queries and limits: `config/github-harvest.yaml`.

```bash
# Search + download → data/raw/public-github-sessions-YYYY-MM-DD.jsonl
make github-harvest

# Preview hits without downloading
make github-harvest-dry

# Harvest + safety scan + curate (public-github files only)
make github-harvest-full

# One query, small batch
uv run --package llm-dataprep github-harvest --query cursor_agent_transcripts --max-files 20
```

**Rate limits:** code search ~9 req/min authenticated; the harvester sleeps 7s between search pages. State is tracked in `data/github_harvest/state.json` — re-runs skip unchanged blobs (same SHA). After tightening queries, run once with `--reset-state` so old broad hits are not retried.

**31 harness queries** (registry in `github_harvest_registry.py`): Cursor, Codex, Claude Code, **Pi**, **OpenCode** (session + message), Gemini CLI, Copilot (Chronicle + history), Kimi, Factory, OpenClaw, OpenHands, Amp, Continue, Cline, Roo Code, Watchfire, Mux, Kiro, Antigravity, Trae, Qwen, Vibe, Aider, **Goose**. Tier-A queries paginate **10 pages** (1000 hits max); warm runs rotate search pages via `state.json` / Redis `queries` cursor.

**Max-yield playbook (12 GB / one PAT):**

| Knob | Default | Why |
|------|---------|-----|
| `max_files_per_run` | 220 | ~9 code-search minutes + download headroom |
| `default_max_pages` | 5 | Tier-B queries; tier-A override `max_pages: 10` |
| `code_search_min_interval_s` | 7.0 | ~9 req/min — do **not** parallelize code search on one token |
| `graphql_pending_flush` | 16 | Batch downloads without starving search |
| `enabled_queries` | — | Rotate ~9 tier-A ids/day to spread the 10 req/min budget |

```bash
# Day 1 — Cursor + Codex + Claude
uv run --package llm-dataprep github-harvest \
  --query cursor_agent_transcripts --query codex_rollout_sessions --query claude_code_projects

# Day 2 — Pi + OpenCode + Gemini
uv run --package llm-dataprep github-harvest \
  --query pi_agent_sessions --query opencode_storage_session --query gemini_cli_chats
```

Each uses GitHub `path:` regex + post-hit path regex + content sniff. SQLite-only harnesses (Crush, T3) remain local-ingest only.

```bash
uv run --package llm-dataprep github-harvest --list-queries
uv run --package llm-dataprep github-harvest --query pi_agent_sessions --query opencode_storage_message --dry-run
```

Benchmark/harbor job trees (`/jobs/`, `terminal_bench`, etc.) are excluded by path regex.

**Rate-limit tips (implemented):**

| Tip | Implementation |
|-----|----------------|
| Authenticate with PAT | `GITHUB_TOKEN` in `.env` (required) |
| Pagination | `per_page=100` on code search |
| Backoff + headers | Exponential backoff; honors `Retry-After` + `X-RateLimit-Reset`; sleeps when **`code_search`** bucket exhausted |
| Cache | `data/github_harvest/state.json` + optional **Redis** (`make redis-up`) — SHA cache for fetched + rejected blobs |
| Raw CDN | Downloads try `raw.githubusercontent.com/{repo}/{sha}/{path}` before Contents API |
| **GraphQL batch** | After REST code search, downloads batched via GraphQL `repository.object(expression: "{sha}:{path}")` — hybrid mode falls back to REST/raw CDN |
| GraphQL-only / REST-only | `rate_limit.download_mode: graphql \| rest \| hybrid` in `config/github-harvest.yaml` |

```bash
# Reset cache after query changes
uv run --package llm-dataprep github-harvest --reset-state

# Local Redis (project Valkey on :6380 — does not touch system service)
make redis-up    # REDIS_PASSWORD in .env
make redis-ping
make redis-down
```

Curated rows get `meta.data_source=public`, `meta.public_dataset=github_public`. Enable in training mix via `training_mix.public_dataset_priority` (already listed in `config/default.yaml` when you opt into public rows).

### Optional: Cloudflare R2 blob cache (P2)

GitHub **code search cannot move to Cloudflare** — it always needs your PAT (~10 req/min). CF free tier helps **after** discovery:

| CF service | Free tier | Use for harvest |
|------------|-----------|-----------------|
| **R2** | 10 GB, 1M writes/mo, zero egress | Cache downloaded blobs by SHA — warm re-runs skip GitHub bytes |
| **Workers** | 100K req/day | Optional HTTPS cache proxy (only if multiple harvest hosts) |
| **KV** | 1K writes/day | **Too small** for SHA corpus — keep local Redis |
| **Durable Objects** | 100K req/day | Overkill for single-PAT harvest |

**MVP:** Python → R2 direct (S3 API), key `github-harvest/blobs/{blob_sha}`, fallback to GitHub on miss. Config stub (future): `r2_bucket`, `r2_endpoint` in `github-harvest.yaml`. Do **not** put `GITHUB_TOKEN` in a public Worker.

## Removed from pipeline (May 2026)

Dropped — generic instruct / low agentic ROI for pyro-coder (not ingested):

`ultradata_sft_2605`, `high_coder_sft`, `agent_trove`, `codex_7m`, `codex_2m_thinking`, `magicoder_75k`, `opencode_broad`, `nemotron_swe`, `self_code_align`

## Top 10 (Dec 2025 – May 2026)

Final public-ingest plan for `pyro-coder` bootstrap. Caps in `config/default.yaml` → `public_datasets.datasets.*.max_rows`.

| Rank | ID | Hugging Face | Released | Default cap | Gated | Why ingest |
|------|-----|--------------|----------|-------------|-------|------------|
| 1 | `ultradata_sft_2605` | [openbmb/UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605) | May 2026 | 50k | no | **Knowledge + Code + Math** configs (`no_think` split) |
| 2 | `swe_chat` | [SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) | Apr 2026 | all | **yes** | Real Cursor / Claude Code wild sessions + tool calls |
| — | `swe_zero_openhands` | [nvidia/SWE-Zero-openhands-trajectories](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories) | 2026 | all | no | OpenHands SWE-Zero trajectories + patches |
| — | `swe_zero_12m` | [AlienKevin/SWE-ZERO-12M-trajectories](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories) | 2026 | all (`Submitted` default) | no | 12M mini-swe-agent trajectories |
| 3 | `coderforge_preview` | [togethercomputer/CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) | Feb 2026 | 10k | no | Test-verified long-horizon agent trajectories (`filtered_reward1`) |
| 4 | `zen_agentic` | [zenlm/zen-agentic-dataset](https://huggingface.co/datasets/zenlm/zen-agentic-dataset) | May 2026 | 5k | no* | Hidden gem — real Claude Code + git history (~12B tokens) |
| 5 | `swe_next` | [TIGER-Lab/SWE-Next-SFT-Trajectories](https://huggingface.co/datasets/TIGER-Lab/SWE-Next-SFT-Trajectories) | Mar–Apr 2026 | all (~3.7k) | no | Execution-grounded SWE trajectories; `tool`→`user` |
| 6 | `high_coder_sft` | [Crownelius/High-Coder-SFT-Medium](https://huggingface.co/datasets/Crownelius/High-Coder-SFT-Medium) | May 2026 | 10k | no | 124k long-form synthetic code (prompt + full source file) |
| 7 | `agentic_sft_new` | [WaltonFuture/agentic-sft-new](https://huggingface.co/datasets/WaltonFuture/agentic-sft-new) | May 2026 | 10k | no | Merged agentic SFT — tools, edits, multi-hop |
| 8 | `agentic_cot_coding` | [mepartha/Agentic-Chain-of-Thought-Coding-SFT-Dataset-v1.1](https://huggingface.co/datasets/mepartha/Agentic-Chain-of-Thought-Coding-SFT-Dataset-v1.1) | 2026 | 10k | no | Agentic CoT coding traces (`[thinking]` prefix when present) |
| 9 | `nemotron_opencode` | [nvidia/Nemotron-SFT-OpenCode-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1) | Mar 2026 | 10k | no | OpenCode-style tool-calling across 6 splits |
| 10 | `agent_trove` | [open-thoughts/AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove) | Apr 2026 | 10k | no | 1.7M general agentic traces (coding-heavy subset) |
| — | `cooper_qwen9b_coop_claude` | [CooperBench/qwen9b-coop-claude-code](https://huggingface.co/datasets/CooperBench/qwen9b-coop-claude-code) | 2026 | all (~368 pairs) | no | Two-agent Claude Code coop trajectories on Qwen3.5-9B |

\* `zen_agentic`: HF repo is currently a **placeholder** (no public shard files). Loader fails fast with card instructions (`oss@hanzo.ai`). When shards land, use `llm-dataprep[zed]` for zstd JSONL streaming.

### Recommended ingest order

1. **Small + ungated first** (validate pipeline): `swe_next`, `cooper_qwen9b_coop_claude`, `high_coder_sft`, `nemotron_opencode`
2. **High-signal agentic**: `coderforge_preview`, `agentic_sft_new`, `agentic_cot_coding`, `agent_trove`, `ling_coder_sft`, `nemotron_swe_v2`, `scale_swe`
3. **Gated / large** (after `hf auth login`): `swe_chat`, `ultradata_sft_2605`, `codex_7m`
4. **When available**: `zen_agentic`

```bash
make public-ingest PUBLIC_DATASETS="cooper_qwen9b_coop_claude,swe_next,ling_coder_sft"
```

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

Optional SFT extras (enabled in config, same pipeline as Top 10):

| ID | Repo | Notes |
|----|------|-------|
| `ling_coder_sft` | `inclusionAI/Ling-Coder-SFT` | ShareGPT `messages` |
| `nemotron_swe_v2` | `nvidia/Nemotron-SFT-SWE-v2` | Split `agentless` only (streaming) |
| `scale_swe` | `AweAI-Team/Scale-SWE` | `problem_statement` + `patch` |
| `codex_7m` | `Modotte/CodeX-7M-Non-Thinking` | 7.36M instruction pairs |
| `codex_2m_thinking` | `Modotte/CodeX-2M-Thinking` | 2.19M with reasoning |
| `cooper_qwen9b_coop_claude` | `CooperBench/qwen9b-coop-claude-code` | `agent1_traj.json` + `agent2_traj.json` per pair |

## Operator flow (use this — not ad-hoc `hf download` / background one-offs)

**Default ingest (`make public-ingest`)** uses **fast mode** (huggingface_hub ≥1.16 + datasets ≥4.7):

1. `snapshot_download(..., local_dir=..., max_workers=N)` — full HF repo to `data/hf_cache/<dataset_id>/` (parallel, auto-resume, hf_xet). No deprecated `resume_download` / `local_dir_use_symlinks`.
2. Convert from **local Parquet via PyArrow batch iteration** (`memory_map=True`, shard-by-shard) — avoids datasets rebuilding a monolithic Arrow cache across all shards.
3. Multi-shard sets use parallel workers (`convert_workers`); capped smoke runs (`--max-rows N`) stay single-process so trajectory limits apply correctly.
4. Write `data/raw/public-*.jsonl`

**Incremental runs:** before download/convert, one Hub API call compares `lastModified` + revision (`sha`) to `data/hf_cache/<id>/.ingest_state.json`. If Hub is not newer and the raw JSONL exists, that dataset is skipped entirely. Legacy caches (download marker + raw file only) bootstrap the same check from file mtimes. Force full refresh: `PUBLIC_REFRESH=1` or `--refresh-download`.

Tune in `config/default.yaml` → `public_datasets.ingest` (`download_workers`, `convert_workers`).

Override cache location: `export LLM_HF_CACHE_DIR=/mnt/fast/hf-cache`

Legacy row-by-row Hub streaming (debug only): `make public-ingest-stream` or `--remote-stream`.

```bash
hf auth login
export LLM_DATA_DIR=/mnt/your-large-disk/llm-data   # optional; overrides data/

# All enabled public sets + personal pipeline
make phase1-public REPO=/path/to/repo

# Or ingest public only, then curate + warehouse
HF_TOKEN=hf_... make data-public
make prepare-mixed
```

Equivalent uv:

```bash
uv run --package llm-dataprep phase1 --public --skip-gated --fresh-raw --include-subagents --repo /path/to/repo
uv run --package llm-dataprep public-ingest --skip-gated --replace
uv run --package llm-dataprep curate-raw --no-gitleaks --no-presidio
# … warehouse-sync-registry, warehouse-load, training-manifest, training-extract
```

`phase1 --public` calls `public-ingest --replace` for every dataset with `enabled: true` in `config/default.yaml`.

## Commands

```bash
make public-list
make public-ingest                                    # all enabled (skip gated without HF_TOKEN)
HF_TOKEN=hf_... make public-ingest                    # include gated sets
make public-ingest PUBLIC_DATASETS="coderforge_preview swe_next"
make curate-fast                                      # bulk public + personal raw
make phase1-public REPO=/path/to/git/repo
```

Equivalent uv:

```bash
uv run --package llm-dataprep public-ingest --list
uv run --package llm-dataprep public-ingest --skip-gated
uv run --package llm-dataprep public-ingest --datasets coderforge_preview --max-rows 1000
uv run --package llm-dataprep curate-raw --no-gitleaks --no-presidio
uv run --package llm-dataprep phase1 --fresh-raw --public --include-subagents --repo /path/to/git/repo
```

## Training mix (PLAN)

- Mature target: **75–85% personal**, **15–25% public**
- Bootstrap: ingest Top 10 with conservative caps; raise caps after curation audit
- Public rows: `source=public`, `exec=pass` where verified (trajectories, tests, exec-filter)
- Filter aggressively in curation: style match, min chars, secrets/PII — not topic/refusal

## Licenses

Check each dataset card before redistribution. Ingest is local-only; do not commit raw HF rows to git.
