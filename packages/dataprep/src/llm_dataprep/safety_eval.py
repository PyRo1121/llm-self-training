"""Evaluate safety scanners against labeled JSONL fixtures (regex-only, reproducible)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm_dataprep.diff_scan import scan_diff_text
from llm_dataprep.filters import SafetyReport, scan_text

_DIFF_LABELS = frozenset({"diff", "git-diff", "git_diff", "git-diffs", "git"})


def _default_fixtures() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "safety_eval.jsonl"


def _use_diff_scan(row: dict[str, Any]) -> bool:
    mode = str(row.get("mode") or "").strip().lower()
    if mode == "diff":
        return True
    label = str(row.get("label") or "").strip().lower()
    if label in _DIFF_LABELS or label.startswith("diff_"):
        return True
    return False


def _scan_fixture(text: str, row: dict[str, Any]) -> SafetyReport:
    """Regex-only scan (no gitleaks/Presidio) for deterministic fixture eval."""
    if _use_diff_scan(row):
        return scan_diff_text(text, use_regex=True)
    return scan_text(text, use_regex=True, use_gitleaks=False, use_presidio=False)


def _predicted_block(report: SafetyReport) -> bool:
    return not report.ok


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    n: int


def compute_metrics(expect_block: Iterable[bool], predicted_block: Iterable[bool]) -> Metrics:
    exp = list(expect_block)
    pred = list(predicted_block)
    if len(exp) != len(pred):
        raise ValueError(f"length mismatch: expect={len(exp)} pred={len(pred)}")
    tp = fp = fn = tn = 0
    for e, p in zip(exp, pred, strict=True):
        if e and p:
            tp += 1
        elif not e and p:
            fp += 1
        elif e and not p:
            fn += 1
        else:
            tn += 1
    n = len(exp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn, n=n)


def load_fixtures(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "text" not in row:
                raise ValueError(f"{path}:{line_no}: missing 'text'")
            if "expect_block" not in row:
                raise ValueError(f"{path}:{line_no}: missing 'expect_block'")
            if "label" not in row:
                raise ValueError(f"{path}:{line_no}: missing 'label'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no fixture rows")
    return rows


def evaluate_fixtures(rows: list[dict[str, Any]]) -> tuple[Metrics, dict[str, Metrics]]:
    expect: list[bool] = []
    predicted: list[bool] = []
    by_label_exp: dict[str, list[bool]] = {}
    by_label_pred: dict[str, list[bool]] = {}

    for row in rows:
        text = row["text"]
        if not isinstance(text, str):
            raise TypeError("fixture 'text' must be a string")
        label = str(row["label"])
        exp = bool(row["expect_block"])
        report = _scan_fixture(text, row)
        pred = _predicted_block(report)
        expect.append(exp)
        predicted.append(pred)
        by_label_exp.setdefault(label, []).append(exp)
        by_label_pred.setdefault(label, []).append(pred)

    overall = compute_metrics(expect, predicted)
    per_label = {
        label: compute_metrics(by_label_exp[label], by_label_pred[label])
        for label in sorted(by_label_exp)
    }
    return overall, per_label


def _format_metrics(name: str, m: Metrics) -> str:
    return (
        f"{name}: n={m.n} precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} "
        f"(tp={m.tp} fp={m.fp} fn={m.fn} tn={m.tn})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safety scanner fixture eval (regex-only; scan_text / scan_diff_text)"
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help="JSONL with text, expect_block, label (default: tests/fixtures/safety_eval.jsonl)",
    )
    parser.add_argument(
        "--per-label",
        action="store_true",
        help="Print precision/recall/F1 per label (default: overall only)",
    )
    args = parser.parse_args()
    fixtures = args.fixtures or _default_fixtures()
    if not fixtures.is_file():
        raise SystemExit(f"fixtures not found: {fixtures}")

    rows = load_fixtures(fixtures)
    overall, per_label = evaluate_fixtures(rows)

    print(f"fixtures: {fixtures.resolve()}")
    print(_format_metrics("overall", overall))
    if args.per_label:
        for label in sorted(per_label):
            print(_format_metrics(f"label={label}", per_label[label]))


if __name__ == "__main__":
    main()
