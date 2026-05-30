# Ollama inventory (this machine)

Snapshot for RAG Phase 3+ — not wired in code yet.

## Inference (training / daily chat)

| Model | Size | Project role |
|-------|------|----------------|
| `qwen2.5-coder:7b` | 4.7 GB | **Default** — `config/default.yaml` `inference_model` |
| `qwen2.5:7b` | 4.7 GB | General fallback |
| `qwen3.5:9b` | 6.6 GB | Optional stronger chat (VRAM) |
| `phi4-reasoning:14b` | 11 GB | Debug/judge only per PLAN — not daily driver on 12 GB |

## Embeddings (RAG)

| Model | Size | Notes |
|-------|------|--------|
| **`qwen3-embedding:4b`** | 2.5 GB | **Preferred** — already installed; `/api/embed` OK; PLAN bake-off target |
| `nomic-embed-text` | 274 MB | Old default in config; fine for smoke tests |
| `mxbai-embed-large` | 669 MB | Alternative |
| `jeffh/intfloat-multilingual-e5-small:f32` | 476 MB | Multilingual / smaller |

**Config:** `config/default.yaml` → `embed_model: qwen3-embedding:4b`

**Chroma:** when ingest runs, use one embed model per collection; re-embed if switching off nomic.

## Reranking

No dedicated rerank model in `ollama list` yet.

PLAN default path: `rank_bm25` + RRF → CPU cross-encoder (`sentence-transformers` / **bge-reranker-v2-m3**) or **Qwen3-Reranker** via **Transformers/vLLM**, not Ollama chat.

Community `dengcao/Qwen3-Reranker-*` on Ollama often **does not** work as a real reranker in RAG stacks (no proper `/api/embed`-style rerank API). If you pulled a rerank model elsewhere, note the name here.

```yaml
# config/default.yaml (placeholder)
rerank:
  backend: null   # sentence_transformers | vllm | ollama_experimental
  model: null     # e.g. BAAI/bge-reranker-v2-m3 or Qwen/Qwen3-Reranker-0.6B
```

## GPU rule (unchanged)

One heavy job: train **or** chat **or** embed batch **or** rerank batch. `ollama stop` before train.
