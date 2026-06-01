# Campaign R25 — 500 agents (25 rounds × 20)

## Schedule

| Round | Mode | Agents | Focus |
|-------|------|--------|-------|
| 1 | AUDIT | 20 | Harvest core + registry |
| 2 | FIX | 20 | Round-1 must-fix only (noise filtered) |
| 3 | AUDIT | 20 | Safety + public ingest |
| 4 | FIX | 20 | Round-3 must-fix |
| … | … | … | … |
| 25 | AUDIT | 20 | Final sign-off sweep |

**Odd rounds:** audit. **Even rounds:** fix.

## Agent rules (all rounds)

### Audit rounds
- 2026 standards (typing, error handling, no dead code)
- Real bugs only — cite `path:line`
- Target **≥80% line coverage** on scoped modules (report gap)
- Reject AI slop ( redundant abstractions, filler comments, speculative helpers)
- **Mandatory:** Context7 (`resolve-library-id` + `query-docs`), Exa (`web_search_exa`), Shell (pytest/ruff in scope), Repo (Read/Grep)
- Verdict: PASS | FAIL | INCOMPLETE (if MCP blocked)

### Fix rounds
- Re-verify every candidate fix with Context7 + Exa before editing
- **Filter noise:** skip style-only, hypothetical, or unverified claims
- Minimal surgical diffs; add tests when fixing real bugs
- Run scoped pytest; paste exit code

## Scopes (rotate across rounds)

| ID | Module / path |
|----|----------------|
| S01 | `github_harvest_registry.py` |
| S02 | `github_harvest.py` parsers A (cursor/claude/gemini) |
| S03 | `github_harvest.py` parsers B (copilot/qwen/cline/roo) |
| S04 | `github_harvest.py` parsers C (codex/opencode/pi/amp) |
| S05 | `github_harvest.py` routing (`detect_harness`, `parse_blob_text`) |
| S06 | `github_harvest_cache.py` + `github_harvest_graphql.py` |
| S07 | `github_harvest_app.py` + rate limit tests |
| S08 | `cursor_transcripts.py` + `claude_sessions.py` |
| S09 | `safety_policy.py` + `safety_quarantine.py` + `safety_eval.py` |
| S10 | `public/loaders.py` + `fast_ingest.py` + `hf_cache.py` |
| S11 | `phase1_run.py` + `filters.py` + `scan_raw.py` |
| S12 | `mix_policy.py` + `curate_raw.py` + `raw_io.py` |
| S13 | `packages/core` warehouse + paths |
| S14 | `packages/train` qlora/config/preflight |
| S15 | `packages/eval` run_eval + suites |
| S16 | `presidio_custom.py` + `diff_scan.py` |
| S17 | `tests/test_github_harvest*.py` coverage gaps |
| S18 | `tests/test_safety*.py` + quarantine tests |
| S19 | `config/*.yaml` + Makefile + scripts |
| S20 | `docs/oss/*` vs implementation drift |

## Round log

### Round 1 — AUDIT (complete)
- Started: 2026-05-31
- Agents: S01–S20
- Verdict: **FAIL** (19/20 scopes; S16 presidio/diff PASS)
- Top real bugs: kiro extension, qwen stream-json, openhands SDK, amp sniff, cache Redis guard, safety diff harness, turso FK, train weighted sampler, eval auto-pass, chunk_messages fallback, load_swe_chat OOM

### Round 2 — FIX (in progress)
- Agents: F01–F20 — verified must-fix only; skip coverage-only unless trivial test
