# Campaign R50 — 1000 agents (50 rounds × 20)

## Schedule

| Round | Mode | Agents |
|-------|------|--------|
| 1 | AUDIT | 20 | ✓ |
| 2 | FIX | 20 | ✓ (15 substantive) |
| 3 | AUDIT | 20 | in progress |
| 4 | FIX | 20 | queued |
| … | alt | … |
| 50 | AUDIT | 20 | final sign-off |

**Odd = audit. Even = fix.**

## Agent rules (all rounds)

### Audit
- 2026 standards, real bugs, ≥80% coverage on scope, no AI slop, no duplicate logic
- **Mandatory:** Context7, Exa, Shell (pytest/ruff), Repo path:line
- Verdict: PASS | FAIL | INCOMPLETE

### Fix
- Re-verify with Context7 + Exa before edit
- Filter noise; minimal surgical diffs; dedupe shared types/helpers
- Run scoped pytest + ruff; paste exit codes

## Scopes S01–S20

| ID | Path |
|----|------|
| S01 | github_harvest_registry.py |
| S02 | github_harvest parsers A (cursor/claude/gemini) |
| S03 | github_harvest parsers B (copilot/qwen/cline/roo) |
| S04 | github_harvest parsers C (codex/opencode/pi/amp) |
| S05 | routing detect_harness / parse_blob_text |
| S06 | cache + graphql |
| S07 | harvest app + CLI |
| S08 | cursor_transcripts + claude_sessions |
| S09 | safety_policy/quarantine/eval |
| S10 | public loaders + fast_ingest |
| S11 | phase1_run + filters + scan_raw |
| S12 | mix_policy + curate_raw + raw_io |
| S13 | core warehouse + paths + gpu_mutex |
| S14 | train qlora/config/preflight |
| S15 | eval run_eval + suites |
| S16 | presidio_custom + diff_scan |
| S17 | harvest test coverage |
| S18 | safety test coverage |
| S19 | config + Makefile + scripts |
| S20 | docs/oss drift |

## Round log

### R1 AUDIT ✓ — 19 FAIL, 1 PASS (S16)
### R2 FIX ✓ — 15 verified bug fixes; 163 tests pass
### R3 AUDIT ✓ — 14 PASS / 6 FAIL (coverage + docs)
### R4 FIX ✓ — app env, docs/mix alignment, coverage tests (app 97%, safety_eval 99%, transcripts 83–96%)
### R5 AUDIT ✓ — dedupe FAIL (18+ dup clusters), ruff PASS (E402 waived), HTTP tests added
### R6 FIX ✓ — message_blocks.py shared helpers, ruff per-file-ignores, API version 2026-03-10 on REST client

**Progress:** 8 / 50 rounds (~160 agents). **Target:** 1000 agents total.

### R7–R8 ✓ message_blocks PASS; qwen dedupe; 248 tests; ruff clean

### R9 AUDIT — in progress (agents 161–180)
