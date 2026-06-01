"""Run internal eval suites; emit per-suite verdict JSON for promote gate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from llm_core.control_plane import ensure_warehouse, register_benchmark_run
from llm_core.paths import eval_dir, runs_dir
from llm_eval.suites import load_suite, suite_is_placeholder_only, suite_names


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ollama_models(host: str) -> set[str]:
    try:
        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=10.0)
        r.raise_for_status()
        data = r.json()
        return {m.get("name", "") for m in data.get("models", [])}
    except Exception:
        return set()


def _smoke_prompt(suite: str, task: dict[str, Any]) -> str:
    if suite == "retrieval_gold":
        text = task.get("query") or task.get("prompt") or "Say OK."
    else:
        text = task.get("prompt") or "Say OK."
    return str(text)[:500]


def _ollama_chat(host: str, model: str, prompt: str) -> str:
    r = httpx.post(
        f"{host.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content") or ""


def evaluate_suite(
    name: str,
    *,
    model: str,
    ollama_host: str,
    strict: bool,
    smoke_chat: bool,
) -> dict[str, Any]:
    tasks = load_suite(name)

    if not tasks:
        return {
            "suite": name,
            "verdict": "fail",
            "reason": "empty_suite",
            "tasks": 0,
            "passed": 0,
        }

    placeholder_only = suite_is_placeholder_only(tasks)

    if placeholder_only:
        if strict:
            return {
                "suite": name,
                "verdict": "fail",
                "reason": "placeholder_tasks_only",
                "tasks": len(tasks),
                "passed": 0,
            }
        return {
            "suite": name,
            "verdict": "pass",
            "reason": "placeholder_suite_skipped",
            "tasks": len(tasks),
            "passed": len(tasks),
        }

    passed = 0
    if smoke_chat and tasks:
        try:
            reply = _ollama_chat(
                ollama_host,
                model,
                _smoke_prompt(name, tasks[0]),
            )
            if len(reply.strip()) >= 2:
                passed = 1
        except Exception as exc:
            return {
                "suite": name,
                "verdict": "fail",
                "reason": f"ollama_chat_failed: {exc}",
                "tasks": len(tasks),
                "passed": 0,
            }

    if passed > 0:
        verdict = "pass"
    elif not smoke_chat:
        verdict = "incomplete"
    else:
        verdict = "fail"
    return {
        "suite": name,
        "verdict": verdict,
        "reason": "smoke_chat" if smoke_chat else "manual_tasks_required",
        "tasks": len(tasks),
        "passed": passed,
    }


def run_all(
    *,
    model: str,
    ollama_host: str,
    strict: bool,
    smoke_chat: bool,
    train_run: str | None,
) -> dict[str, Any]:
    suites = suite_names()
    results = [
        evaluate_suite(
            s,
            model=model,
            ollama_host=ollama_host,
            strict=strict,
            smoke_chat=smoke_chat,
        )
        for s in suites
    ]
    all_pass = all(r["verdict"] == "pass" for r in results)
    out = {
        "model": model,
        "ollama_host": ollama_host,
        "strict": strict,
        "evaluated_at": _utc_now(),
        "suites": results,
        "verdict": "pass" if all_pass else "fail",
        "train_run": train_run,
    }

    if train_run:
        conn = ensure_warehouse()
        try:
            for row in results:
                register_benchmark_run(
                    conn,
                    suite=row["suite"],
                    train_run_name=train_run,
                    status="completed",
                    scores=row,
                )
        finally:
            conn.close()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal eval suites (promote gate)")
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="Ollama model tag for smoke chat",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail placeholder-only suites (required for real promote)",
    )
    parser.add_argument(
        "--no-smoke-chat",
        action="store_true",
        help="Skip live Ollama chat probe",
    )
    parser.add_argument(
        "--train-run",
        default=None,
        help="Link results to training_runs.run_name (e.g. pyro-coder-bootstrap)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write eval_report.json (default: runs/<train-run>/ or eval/)",
    )
    args = parser.parse_args()

    models = _ollama_models(args.ollama_host)
    if args.model not in models and not args.no_smoke_chat:
        print(
            f"Warning: model {args.model!r} not in Ollama ({len(models)} tags). "
            "Use --no-smoke-chat for placeholder-only pass.",
            file=sys.stderr,
        )

    report = run_all(
        model=args.model,
        ollama_host=args.ollama_host,
        strict=args.strict,
        smoke_chat=not args.no_smoke_chat,
        train_run=args.train_run,
    )

    out_path = args.out
    if out_path is None:
        if args.train_run:
            out_path = runs_dir() / args.train_run / "eval_report.json"
        else:
            out_path = eval_dir() / "last_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)

    if report["verdict"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
