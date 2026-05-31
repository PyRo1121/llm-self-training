# LLM Self Training — operator shortcuts
# Usage: make help
# Override: make train RUN=smoke-test  |  make phase1 REPO=/path/to/repo

SHELL := /bin/bash
ROOT := $(CURDIR)

# --- defaults (override on CLI: make train RUN=my-run) ---
RUN              ?= pyro-coder-bootstrap
MANIFEST         ?= personal-first
TRAIN_FILE       ?= data/train/$(MANIFEST).jsonl
REPO             ?=
HF_TOKEN         ?=
PUBLIC_DATASETS  ?=
PERSONAL_RATIO   ?=
HF_DATASET       ?=
PUBLIC_CAP       ?=
UV               := uv run --package

.PHONY: help
help: ## Show targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Setup
# =============================================================================

.PHONY: sync sync-all sync-train sync-dataprep sync-safety
sync: ## Core + API deps
	uv sync --package llm-core --package llm-api

sync-all: ## Core, train, eval, dataprep, API
	uv sync --package llm-core --package llm-train --package llm-eval --package llm-api
	uv sync --package llm-dataprep --extra git

sync-train: ## Train stack (Unsloth)
	uv sync --package llm-train --extra unsloth

sync-dataprep: ## Ingest + git extras
	uv sync --package llm-dataprep --extra git

sync-safety: ## Presidio PII scanner
	uv sync --package llm-dataprep --extra safety

# =============================================================================
# Data — ingest
# =============================================================================

.PHONY: ingest ingest-list agent-ingest public-ingest public-list
ingest: sync-dataprep ## Pull local agent logs → data/raw (HARNESS=cursor,codex optional)
	$(UV) llm-dataprep agent-ingest \
		$(if $(HARNESS),--harness $(HARNESS),) \
		$(if $(filter 0,$(INCLUDE_SUBAGENTS)),,--include-subagents) \
		$(if $(MAX_CODEX_MB),--max-codex-mb $(MAX_CODEX_MB),) \
		$(if $(REPO),--repo $(REPO),)

agent-ingest: ingest ## Alias for ingest

ingest-list: ## List agent harnesses
	$(UV) llm-dataprep agent-ingest --list-harnesses

public-ingest: sync-dataprep ## Pull HF public datasets → data/raw (fast local parquet default)
	$(if $(HF_TOKEN),HF_TOKEN=$(HF_TOKEN),) \
	HF_XET_HIGH_PERFORMANCE=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=24 \
	$(UV) llm-dataprep public-ingest \
		$(if $(PUBLIC_DATASETS),--datasets $(PUBLIC_DATASETS),) \
		$(if $(PUBLIC_MAX_ROWS),--max-rows $(PUBLIC_MAX_ROWS),) \
		$(if $(HF_TOKEN),,--skip-gated) \
		$(if $(PUBLIC_REFRESH),--refresh-download,) \
		--replace

public-ingest-stream: sync-dataprep ## Legacy Hub streaming ingest (slow; debugging only)
	$(if $(HF_TOKEN),HF_TOKEN=$(HF_TOKEN),) \
	$(UV) llm-dataprep public-ingest --remote-stream \
		$(if $(PUBLIC_DATASETS),--datasets $(PUBLIC_DATASETS),) \
		$(if $(PUBLIC_MAX_ROWS),--max-rows $(PUBLIC_MAX_ROWS),) \
		$(if $(HF_TOKEN),,--skip-gated) \
		--replace

public-list: ## List public dataset registry
	$(UV) llm-dataprep public-ingest --list

# =============================================================================
# Data — sanitize / curate (secrets + PII)
# =============================================================================

.PHONY: sanitize scan curate curate-fast audit-sample
sanitize: sync-safety ## Scan raw JSONL for secrets + PII (gitleaks if on PATH)
	$(UV) llm-dataprep scan-raw --gitleaks --gitleaks-per-file

scan: sanitize ## Alias for sanitize

curate: sync-dataprep ## Raw → curated (honors safety-failures; gitleaks+presidio if installed)
	$(UV) llm-dataprep curate-raw

curate-fast: sync-dataprep ## Curate without per-row scanners (public bulk)
	$(UV) llm-dataprep curate-raw --no-gitleaks --no-presidio \
		--latest-per-prefix \
		--exclude-glob 'public-stack-v2-dedup*.jsonl'

audit-sample: ## 50-row secrets/PII audit from latest curated file
	@latest=$$(ls -t data/curated/curated*.jsonl 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then echo "No curated/*.jsonl — run make curate first"; exit 1; fi; \
	$(UV) llm-dataprep audit-sample --curated "$$latest"

# =============================================================================
# Data — full Phase 1 pipeline
# =============================================================================

.PHONY: phase1 phase1-public
phase1: sync-dataprep ## Ingest → scan → curate → link → replay → audit
	$(UV) llm-dataprep phase1 \
		--include-subagents \
		--gitleaks --gitleaks-per-file \
		$(if $(REPO),--repo $(REPO),) \
		--mark-exec

phase1-public: sync-dataprep ## phase1 + public HF ingest first
	$(if $(HF_TOKEN),HF_TOKEN=$(HF_TOKEN),) \
	$(UV) llm-dataprep phase1 \
		--public \
		--include-subagents \
		$(if $(HF_TOKEN),,--skip-gated) \
		$(if $(REPO),--repo $(REPO),) \
		--mark-exec

# =============================================================================
# Warehouse + training manifests
# =============================================================================

.PHONY: warehouse-load warehouse-smoke lake-stats
warehouse-load: ## Index latest curated JSONL into control_plane.db
	$(UV) llm-dataprep warehouse-load --latest

warehouse-smoke: ## Quick warehouse health check
	$(UV) llm-core warehouse-smoke

lake-stats: ## Stats on raw + latest curated
	$(UV) llm-dataprep lake-stats --latest-curated

.PHONY: manifest manifest-personal manifest-mixed extract extract-personal extract-mixed
manifest: manifest-mixed ## Default: 80/20 personal/public mix (config training_mix)

manifest-mixed: warehouse-smoke ## Build mixed manifest (personal + public datasets)
	$(UV) llm-dataprep training-manifest \
		--manifest-id $(MANIFEST) \
		$(if $(PERSONAL_RATIO),--personal-ratio $(PERSONAL_RATIO),) \
		$(if $(PUBLIC_CAP),--public-cap $(PUBLIC_CAP),)

manifest-personal: ## Personal rows only (no public HF data)
	$(UV) llm-dataprep training-manifest \
		--manifest-id personal-only \
		--personal-only

extract: ## Write train JSONL from manifest (MANIFEST=… TRAIN_FILE=…)
	$(UV) llm-dataprep training-extract \
		--manifest-id $(MANIFEST) \
		--out $(TRAIN_FILE)

extract-mixed: ## Extract default mixed train file
	$(MAKE) manifest-mixed MANIFEST=personal-first
	$(MAKE) extract MANIFEST=personal-first TRAIN_FILE=data/train/personal-first.jsonl

extract-personal: ## Extract personal-only train file
	$(MAKE) manifest-personal
	$(MAKE) extract MANIFEST=personal-only TRAIN_FILE=data/train/personal-only.jsonl

# =============================================================================
# Public data → train-ready (ingest + curate + warehouse + manifest + extract)
# =============================================================================

.PHONY: data-public data-public-fast prepare-mixed prepare-personal
data-public: public-ingest curate warehouse-load ## Ingest public HF + curate + index
	@echo "Public data indexed. Run: make prepare-mixed"

data-public-fast: public-ingest curate-fast warehouse-load ## Public ingest + fast curate (no presidio)

prepare-mixed: extract-mixed ## Manifest + extract for mixed train (80/20)
prepare-personal: extract-personal ## Manifest + extract personal-only

# =============================================================================
# GPU
# =============================================================================

.PHONY: gpu-clear gpu-status
gpu-clear: ## Reclaim VRAM (ollama, hyprwhspr, ghost VRAM)
	$(UV) llm-core clear-gpu-vram

gpu-status: ## Show GPU compute processes
	nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# =============================================================================
# Train — preflight + runs
# =============================================================================

.PHONY: train-preflight train-preflight-promote train-dry-run
train-preflight: sync-train ## Bootstrap preflight (no GPU train)
	$(UV) llm-train train-preflight

train-preflight-promote: sync-train ## Promote profile preflight
	$(UV) llm-train train-preflight --promote

train-dry-run: sync-train prepare-mixed ## Dataset stats only, no GPU
	$(UV) llm-train train-qlora --dry-run --train-file $(TRAIN_FILE)

.PHONY: train train-smoke train-personal train-mixed train-promote
train-smoke: sync-train prepare-mixed ## ~5 step smoke (RUN=smoke-test)
	$(UV) llm-train train-qlora --smoke --run-name $(if $(filter smoke-test,$(RUN)),$(RUN),smoke-test)

train: sync-train prepare-mixed ## Full bootstrap QLoRA (RUN=pyro-coder-bootstrap)
	$(UV) llm-train train-qlora --run-name $(RUN) --train-file $(TRAIN_FILE)

train-personal: sync-train prepare-personal ## Train on personal data only
	$(UV) llm-train train-qlora \
		--run-name $(if $(filter pyro-coder-bootstrap,$(RUN)),pyro-coder-personal,$(RUN)) \
		--train-file data/train/personal-only.jsonl

train-mixed: train ## Alias — personal + public HF mix (default)

train-promote: sync-train prepare-mixed ## Promote profile (Unsloth, higher rank)
	$(UV) llm-train train-preflight --promote
	$(UV) llm-train train-qlora --promote --run-name $(RUN) --train-file $(TRAIN_FILE)

# =============================================================================
# Post-train — register, eval, export
# =============================================================================

.PHONY: train-register eval export phase2-done
train-register: ## Register run in warehouse (RUN=…)
	$(UV) llm-train train-register --run-name $(RUN)

eval: ## Eval gate (placeholder suites OK for bootstrap)
	$(UV) llm-eval run-eval --train-run $(RUN) --no-smoke-chat

export: ## Merge adapter → exports/ (needs CUDA)
	$(UV) llm-train train-export \
		--adapter-dir runs/$(RUN)/adapter \
		--out exports/$(RUN)

phase2-done: ## Register + eval after bootstrap train
	./scripts/phase2-complete.sh $(RUN)

# =============================================================================
# Dev services
# =============================================================================

.PHONY: api dashboard test test-gpu-mutex verify-phase15 lint
api: sync ## Control plane :8080
	$(UV) llm-api llm-api

dashboard: ## Vite UI :5173
	cd apps/dashboard && (bun install --frozen-lockfile 2>/dev/null || bun install) && bun run dev

test: ## Unit tests (core)
	uv run pytest packages/core/tests/test_gpu_mutex.py -q

test-gpu-mutex: test ## Alias

verify-phase15: ## API health + dashboard build
	./scripts/verify-phase15.sh

lint: ## Ruff check packages + API (no GPU)
	uv run ruff check packages apps/api

# =============================================================================
# Cloud — Jarvis H100 (docs/oss/CLOUD-TRAIN.md)
# =============================================================================

.PHONY: cloud-pack cloud-export-personal cloud-jarvis cloud-jarvis-smoke cloud-train-local
cloud-export-personal: ## Export tier-1 personal → data/cloud/personal/ (private repo only)
	bash scripts/cloud/export-personal-for-git.sh

cloud-pack: ## Local tier-1 → private HF dataset (alternative to git)
	bash scripts/cloud/pack-personal.sh

cloud-jarvis-smoke: ## Jarvis H100 smoke (ingest + 5 train steps)
	bash scripts/cloud/run-jarvis.sh --smoke-only

cloud-jarvis: ## Jarvis H100 full managed run (jl CLI)
	bash scripts/cloud/run-jarvis.sh

cloud-train-local: sync-all ## Full cloud pipeline on this machine (needs GPU)
	LLM_CONFIG_PROFILE=cloud-h100 bash scripts/cloud/train-cloud.sh \
		--run pyro-coder-h100-v1 \
		--personal-dataset $(HF_DATASET) \
		--personal-ratio $(PERSONAL_RATIO)
