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
	@echo ""
	@echo "  One-shot:"
	@grep -E '^(prep|train|train-cloud):.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[33m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  All targets:"
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Setup
# =============================================================================

.PHONY: sync sync-all sync-train sync-dataprep sync-safety
sync: ## Core + API deps
	uv sync --package llm-core --package llm-api

sync-all: ## Core, train, eval, dataprep, API (single sync — avoids extra churn)
	uv sync \
		--package llm-core \
		--package llm-train \
		--package llm-eval \
		--package llm-api \
		--package llm-dataprep \
		--extra full

sync-train: ## Train stack (Unsloth)
	uv sync --package llm-train --extra unsloth

sync-dataprep: ## Ingest + git + Presidio extras
	uv sync --package llm-dataprep --extra full

sync-safety: ## Alias — same as sync-dataprep (uv sync drops extras if split)
	uv sync --package llm-dataprep --extra full

sync-harvest: ## GitHub harvest (dataprep + redis for local cache)
	uv sync --package llm-dataprep --extra full --extra harvest

.PHONY: redis-up redis-down redis-ping
redis-up: ## Start project-local Valkey on :6380 (REDIS_PASSWORD in .env)
	@chmod +x scripts/redis-local.sh
	@if [ -f .env ]; then set -a; source .env; set +a; fi; \
	bash scripts/redis-local.sh

redis-down: ## Stop project-local Valkey
	@chmod +x scripts/redis-local-stop.sh
	bash scripts/redis-local-stop.sh

redis-ping: ## Ping local Redis
	@if [ -f .env ]; then set -a; source .env; set +a; fi; \
	redis-cli -p $${REDIS_PORT:-6380} -a "$$REDIS_PASSWORD" ping

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
	@if [ -f .env ]; then set -a; source .env; set +a; fi; \
	$(if $(HF_TOKEN),export HF_TOKEN='$(HF_TOKEN)';,) \
	HF_XET_HIGH_PERFORMANCE=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=24 \
	$(UV) llm-dataprep public-ingest \
		$(if $(PUBLIC_DATASETS),--datasets $(PUBLIC_DATASETS),) \
		$(if $(PUBLIC_MAX_ROWS),--max-rows $(PUBLIC_MAX_ROWS),) \
		$$( [ -n "$${HF_TOKEN:-}" ] || echo --skip-gated ) \
		$(if $(PUBLIC_REFRESH),--refresh-download,) \
		--replace

public-ingest-stream: sync-dataprep ## Legacy Hub streaming ingest (slow; debugging only)
	@if [ -f .env ]; then set -a; source .env; set +a; fi; \
	$(if $(HF_TOKEN),export HF_TOKEN='$(HF_TOKEN)';,) \
	$(UV) llm-dataprep public-ingest --remote-stream \
		$(if $(PUBLIC_DATASETS),--datasets $(PUBLIC_DATASETS),) \
		$(if $(PUBLIC_MAX_ROWS),--max-rows $(PUBLIC_MAX_ROWS),) \
		$$( [ -n "$${HF_TOKEN:-}" ] || echo --skip-gated ) \
		--replace

public-list: ## List public dataset registry
	$(UV) llm-dataprep public-ingest --list

.PHONY: github-harvest github-harvest-dry github-harvest-full redis-up redis-down redis-ping
github-harvest: sync-harvest ## GitHub code search → data/raw/public-github-sessions-*.jsonl
	@if [ -f .env ]; then set -a; source .env; set +a; fi; \
	$(UV) llm-dataprep github-harvest

github-harvest-dry: sync-harvest ## Dry-run harvest (search hits only; no download)
	@if [ -f .env ]; then set -a; source .env; set +a; fi; \
	$(UV) llm-dataprep github-harvest --dry-run

github-harvest-full: sync-safety sync-harvest ## Harvest → scan → curate public-github rows only
	@if [ -f .env ]; then set -a; source .env; set +a; fi; \
	$(UV) llm-dataprep github-harvest && \
	SCAN_WORKERS=$${SCAN_WORKERS:-4} $(UV) llm-dataprep scan-raw \
		--glob 'public-github-*.jsonl' --gitleaks --gitleaks-per-file \
		--workers $${SCAN_WORKERS:-4} && \
	CURATE_WORKERS=$${CURATE_WORKERS:-8} $(UV) llm-dataprep curate-raw \
		--honor-safety-failures --workers $${CURATE_WORKERS:-8} \
		--glob 'public-github-*.jsonl'

# =============================================================================
# Data — sanitize / curate (secrets + PII; block/warn + allowlist)
# =============================================================================

SAFETY_FIXTURES ?= packages/dataprep/tests/fixtures/safety_eval.jsonl

.PHONY: sanitize scan safety-eval curate curate-fast audit-sample
sanitize: sync-safety ## Pipeline 1/3: parallel scan-raw (gitleaks-per-file, Presidio) → safety-failures
	SCAN_WORKERS=$${SCAN_WORKERS:-8} $(UV) llm-dataprep scan-raw --gitleaks --gitleaks-per-file --workers $${SCAN_WORKERS:-8}

scan: sanitize ## Alias — scan-raw → safety-failures (then make curate)

safety-eval: sync-safety ## Fixture regression (block/warn, diff mode; P/R/F1)
	$(UV) llm-dataprep safety-eval --fixtures $(SAFETY_FIXTURES)

curate: sync-dataprep ## Pipeline 2/3: curate-raw (honors safety-failures; batch Presidio)
	CURATE_WORKERS=$${CURATE_WORKERS:-8} $(UV) llm-dataprep curate-raw --honor-safety-failures --workers $${CURATE_WORKERS:-8}

curate-fast: sync-dataprep ## Curate without per-row scanners (public bulk)
	$(UV) llm-dataprep curate-raw --no-gitleaks --no-presidio \
		--latest-per-prefix \
		--exclude-glob 'public-stack-v2-dedup*.jsonl'

audit-sample: ## Pipeline 3/3: 50-row audit from latest curated (operator sign-off)
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

lake-stats: ## Stats on latest curated (skips full raw scan)
	$(UV) llm-dataprep lake-stats --latest-curated --skip-raw

# =============================================================================
# One-shot operators (start here)
# =============================================================================

.PHONY: prep prep-bg train train-cloud
prep: ## Personal + public data → data/train/personal-first.jsonl (.env loads HF_TOKEN)
	bash scripts/prep-all.sh $(if $(REPO),--repo $(REPO),)

prep-bg: ## prep in background (log: logs/prep-*.log)
	@mkdir -p logs
	@if [[ -f logs/prep.pid ]] && kill -0 "$$(cat logs/prep.pid)" 2>/dev/null; then \
		echo "prep already running (pid $$(cat logs/prep.pid)) — tail -f logs/prep-*.log"; \
		exit 0; \
	fi
	@rm -f logs/prep.lock
	@LOG="logs/prep-$$(date -u +%Y%m%dT%H%M%SZ).log"; \
	nohup bash -c 'exec 9>logs/prep.lock; flock -n 9 || { echo "prep: flock failed"; exit 1; }; bash scripts/prep-all.sh' \
		> "$$LOG" 2>&1 & \
		echo $$! > logs/prep.pid; \
		echo "prep running in background (pid $$(cat logs/prep.pid)) — tail -f $$LOG"

train: gpu-clear ## Local GPU QLoRA (run make prep first; RUN=pyro-coder-bootstrap)
	$(MAKE) sync-train prepare-mixed
	$(UV) llm-train train-qlora --run-name $(RUN) --train-file $(TRAIN_FILE)

train-cloud: ## Rent Vast H100 + ingest/train/export (HF_TOKEN in .env)
	bash scripts/cloud/run-vast.sh

train-cloud-smoke: ## Vast H100 smoke (5 train steps)
	bash scripts/cloud/run-vast.sh --smoke-only

.PHONY: manifest manifest-personal manifest-mixed extract extract-personal extract-mixed
manifest: manifest-mixed ## Default: personal-first mix (config training_mix; set PERSONAL_RATIO=0.8 for 80/20)

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

.PHONY: parse-all-public
parse-all-public: ## Full public parse smallest→largest (ingest+scan+curate); logs/parse-public-*.log
	FORCE_RECONVERT=$${FORCE_RECONVERT:-0} bash scripts/parse-public-ordered.sh

data-public-fast: public-ingest curate-fast warehouse-load ## Public ingest + fast curate (no presidio)

prepare-mixed: extract-mixed ## Manifest + extract for mixed train (override PERSONAL_RATIO in env)
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

.PHONY: train-smoke train-personal train-mixed train-promote
train-smoke: gpu-clear sync-train prepare-mixed ## ~5 step smoke (RUN=smoke-test)
	$(UV) llm-train train-qlora --smoke --run-name $(if $(filter smoke-test,$(RUN)),$(RUN),smoke-test)

train-personal: gpu-clear sync-train prepare-personal ## Train on personal data only
	$(UV) llm-train train-qlora \
		--run-name $(if $(filter pyro-coder-bootstrap,$(RUN)),pyro-coder-personal,$(RUN)) \
		--train-file data/train/personal-only.jsonl

train-mixed: train ## Alias — personal + public HF mix (default)

train-promote: gpu-clear sync-train prepare-mixed ## Promote profile (Unsloth, higher rank)
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
# Cloud — Vast.ai (primary) + Jarvis (docs/oss/CLOUD-TRAIN.md, docs/cloud/)
# =============================================================================

.PHONY: cloud-pack cloud-export-personal cloud-vast cloud-vast-smoke cloud-vast-pull cloud-vast-destroy \
	cloud-jarvis cloud-jarvis-smoke cloud-train-local
cloud-export-personal: ## Export tier-1 personal → data/cloud/personal/ (+ harness shards)
	bash scripts/cloud/export-personal-for-git.sh

cloud-export-all: sync-dataprep ## Ingest all local harnesses → curate → cloud export
	$(UV) llm-dataprep agent-ingest
	$(UV) llm-dataprep curate-raw --no-gitleaks --no-presidio --latest-per-prefix \
		--exclude-glob 'public-*.jsonl'
	$(MAKE) cloud-export-personal

cloud-pack: ## Local tier-1 → private HF dataset (alternative to git)
	bash scripts/cloud/pack-personal.sh

cloud-vast-smoke: ## Vast H100: rent + smoke (5 train steps)
	bash scripts/cloud/run-vast.sh --smoke-only

cloud-vast: ## Vast H100: rent + full ingest/train/export
	bash scripts/cloud/run-vast.sh

cloud-vast-pull: ## Pull runs/exports from last (or INSTANCE=) Vast box
	bash scripts/cloud/vast-pull.sh

cloud-vast-destroy: ## Destroy Vast instance (stop billing)
	bash scripts/cloud/vast-destroy.sh

cloud-jarvis-smoke: ## Jarvis H100 smoke (ingest + 5 train steps)
	bash scripts/cloud/run-jarvis.sh --smoke-only

cloud-jarvis: ## Jarvis H100 full managed run (jl CLI)
	bash scripts/cloud/run-jarvis.sh

cloud-train-local: sync-all ## Full cloud pipeline on this machine (needs GPU)
	LLM_CONFIG_PROFILE=cloud-h100 bash scripts/cloud/train-cloud.sh \
		--run pyro-coder-h100-v1 \
		$(if $(HF_DATASET),--personal-dataset $(HF_DATASET),) \
		$(if $(PERSONAL_RATIO),--personal-ratio $(PERSONAL_RATIO),)
