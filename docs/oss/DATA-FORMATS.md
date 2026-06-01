# Data formats

JSONL schemas from ingest through training. Warehouse stores **metadata pointers only** — see [ARCHITECTURE.md](ARCHITECTURE.md).

## Pipeline

```
data/raw/*.jsonl           one row per message turn
       ↓ curate-raw
data/curated/*.jsonl       messages[] + meta (tier, safety, labels)
       ↓ warehouse-load + training-manifest + training-extract
data/train/*.jsonl         manifest-selected rows + sample_weight
       ↓ load_messages_dataset
HF Dataset                 messages, _sample_weight, _data_source
```

## Raw JSONL (`data/raw/`)

One JSON object per line. Sessions reconstructed by `(harness, session_id)`.

### Personal harness row

```json
{
  "source": "cursor",
  "harness": "cursor",
  "session_id": "39aa3e6f-60e3-41ee-bb16-99ba64f89c4f",
  "source_path": "/home/user/.cursor/projects/.../agent-transcripts/uuid/uuid.jsonl",
  "line_no": 3,
  "role": "user",
  "text": "Fix the auth middleware bug in ...",
  "has_tool_use": false,
  "ingested_at": "2026-05-30T12:00:00+00:00"
}
```

### Public HF row

```json
{
  "source": "public",
  "harness": "public_swe_next",
  "dataset_id": "swe_next",
  "session_id": "2a55af89-4e4f-4460-b18f-42a07287ae76",
  "source_path": "hf://TIGER-Lab/SWE-Next-SFT-Trajectories",
  "line_no": 1,
  "role": "user",
  "text": "...",
  "label": "accepted",
  "exec": "pass",
  "verify": "public_verified",
  "map_tool_to_user": false,
  "ingested_at": "2026-05-30T17:20:07+00:00"
}
```

**File naming:** `{prefix}-YYYY-MM-DD.jsonl` (append mode same day).

Legacy raw rows may include `stack_index: true` from old experiments; `curate-raw` skips them and excludes `public-stack-v2-dedup*.jsonl` by default.

### Safety failure row

Written by `scan-raw` to `data/raw/safety-failures-YYYY-MM-DD.jsonl` when policy quarantines the row. Warn-only rows (under `quarantine_severity: block`) go to `safety-warn-*.jsonl` instead.

```json
{
  "source_file": "/abs/path/to/raw/file.jsonl",
  "line_no": 42,
  "harness": "cursor",
  "session_id": "...",
  "role": "user",
  "source_path": "/home/user/.cursor/.../uuid.jsonl",
  "block_count": 1,
  "warn_count": 0,
  "max_severity": "block",
  "safety": {
    "ok": false,
    "findings": [
      {"source": "regex", "kind": "openai_key", "detail": "sk-...", "start": 12, "end": 48}
    ]
  },
  "text_preview": "first 240 chars..."
}
```

| Field | Notes |
|-------|-------|
| `block_count` / `warn_count` | Findings after allowlist + severity split |
| `max_severity` | `block` \| `warn` \| `none` |
| `safety.ok` | `false` when row quarantined |
| `safety.findings[]` | See [finding object](#safety-finding-object) |

Entire session quarantined at curate if any ingest row matches `(source_file, line_no)` in failures.

### Safety finding object

Used in `safety-failures`, curated `meta.safety`, and `audit-sample` output.

| Field | Required | Notes |
|-------|----------|-------|
| `source` | yes | `regex` \| `gitleaks` \| `presidio` \| `json` |
| `kind` | yes | Pattern or entity id (e.g. `github_pat`, `EMAIL_ADDRESS`, gitleaks rule) |
| `detail` | yes | Match snippet; Presidio includes `score=0.85` |
| `start`, `end` | no | Char offsets in scanned text (regex/Presidio) |
| `severity` | audit only | `block` \| `warn` — set by `audit-sample` / policy helpers, not raw `scan-raw` |

`audit-sample` adds `block_count`, `warn_count`, and `raw_findings_count` on its `safety` object.

## Curated JSONL (`data/curated/`)

### Shape

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": { ... }
}
```

### `meta` fields

| Field | Values / notes |
|-------|----------------|
| `label` | `accepted` \| `edited_heavily` \| `rejected` |
| `exec` | `pass` \| `fail` \| `unknown` |
| `verify` | `cursor_ok`, `git_linked`, `public_verified`, `unknown`, … |
| `train_tier` | `0` drop, `1` SFT, `2` replay-only |
| `harness` | `cursor`, `public_swe_chat`, … |
| `data_source` | `personal` \| `public` |
| `public_dataset` | e.g. `swe_next` (public rows) |
| `session_id`, `project` | traceability |
| `chunk_index`, `chunk_count` | when session chunked |
| `safety_ok` | bool |
| `safety` | `{ok, findings[]}` |
| `style_tags` | e.g. `["debug", "security"]` — metadata only |
| `tone` | `neutral` \| `urgent` \| `collaborative` |
| `linked_commits` | from `link-logs-to-diffs` |

`sample_weight` is **not** set at curate — injected at `training-extract`.

## Tier-1 gate

From `packages/dataprep/src/llm_dataprep/tier1.py`:

| Condition | Result |
|-----------|--------|
| `safety_ok == false` | tier **0** |
| `label == rejected` | tier **0** |
| quality fail (min messages/chars) | tier **0** |
| `label` ∉ `{accepted, edited_heavily}` | tier **0** |
| `exec==pass` OR `verify` ∈ `{cursor_ok, pass}` | tier **1** |
| `bootstrap_mode: true` AND exec/verify both `unknown` | tier **1** |
| else | tier **2** (replay) |

**Bootstrap** (`curation.bootstrap_mode: true`): personal rows reach tier-1 without exec/verify until git linking.

**Manifest SQL:** `train_tier = 1 AND safety_ok = 1`.

## Curation thresholds (`config/default.yaml`)

| Key | Default |
|-----|---------|
| `min_messages` | 2 |
| `min_message_chars` | 40 |
| `min_total_chars` | 200 |
| `max_chars_per_message` | 16000 |
| `max_messages_per_example` | 24 |
| `chunk_overlap_messages` | 4 |
| `skip_roles` | `[developer, system]` |

## Train JSONL (`data/train/`)

Same as curated plus manifest fields:

```json
{
  "messages": [ ... ],
  "meta": {
    "label": "accepted",
    "train_tier": 1,
    "harness": "codex",
    "sample_weight": 1.0,
    "data_source": "personal",
    "style_tags": ["debug"]
  }
}
```

### Sample weights

| Source | Default weight | Config key |
|--------|------------------|------------|
| personal | 1.0 | `training_mix.personal_sample_weight` |
| public | 0.25 | `training_mix.public_sample_weight` |

Loader adds columns `_sample_weight` (floor 0.05) and `_data_source` for `WeightedRandomSampler`.

### Mix policy

With `personal_ratio: 1.0` (default) and `prioritize_personal: true`, public rows are capped out unless you lower `personal_ratio` (e.g. 0.80):

- All personal tier-1 rows included
- Public capped at `floor(personal_count × 0.25)` unless `public_cap` set
- Public fill order: `public_dataset_priority` list in config

## Train-time char caps

| Profile | Cap |
|---------|-----|
| Bootstrap | `max(2000, max_seq × 4)` if yaml null |
| Promote | 12288 chars/message |

Unsloth pre-tokenizes with **keep_end** truncate at VRAM-planned `max_seq`.

## Replay JSONL (`data/replay/`)

Same as curated plus `meta.replay_stratum: true`. Tier-2 all + 25% tier-1 sample.

## Eval suite JSONL (`eval/internal/`)

### diff_apply (primary gate)

```json
{
  "id": "my-repo-fix-001",
  "repo": "/path/to/frozen/worktree",
  "base_commit": "abc123...",
  "prompt": "Apply the patch for the described fix.",
  "patch": "diff --git a/foo.py b/foo.py\n...",
  "test_cmd": "pytest tests/test_foo.py",
  "meta": {"note": "real frozen snapshot"}
}
```

Placeholder detection: `REPLACE_ME`, `id` ending `-example-001`, meta note containing "replace".

### style

```json
{
  "id": "style-001",
  "prompt": "Refactor to match project style.",
  "context": "def foo():\n    pass\n",
  "meta": {"style_tags": ["python"]}
}
```

### debug

```json
{
  "id": "debug-001",
  "prompt": "Fix the bug: tests fail with AssertionError.",
  "context": "def add(a, b):\n    return a - b\n",
  "expected_signal": "tests_pass"
}
```

### retrieval_gold

```json
{
  "id": "rag-001",
  "query": "Where is training config defined?",
  "expected_doc_id": "config-default-yaml",
  "meta": {"note": "after RAG ingest"}
}
```

## Warehouse pointer (manifest row)

Not full messages — used internally:

```json
{
  "curated_id": "sha256-prefix",
  "source_file": "/path/to/curated-2026-05-30.jsonl",
  "source_line": 42,
  "harness": "codex",
  "data_source": "personal",
  "sample_weight": 1.0
}
```

## Related

- [OSS-RELEASE.md](OSS-RELEASE.md) — safety policy
- [PUBLIC-DATASETS.md](PUBLIC-DATASETS.md) — HF registry
