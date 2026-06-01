"""Diff-aware safety scan: added-line extraction, allowlist, password blocks."""

from __future__ import annotations

from llm_dataprep.diff_scan import extract_added_lines, scan_diff_record, scan_diff_text


def test_extract_added_lines_strips_diff_markers() -> None:
    diff = """@@ -1,3 +1,4 @@
--- a/config.env
+++ b/config.env
 context unchanged
-old=value
+new=value
+AKIAIOSFODNN7EXAMPLE
"""
    assert extract_added_lines(diff) == (
        "context unchanged\nnew=value\nAKIAIOSFODNN7EXAMPLE"
    )


def test_extract_added_lines_empty_and_no_additions() -> None:
    assert extract_added_lines("") == ""
    assert extract_added_lines("--- a/x\n+++ b/x\n-old\n context") == "context"


def test_allowlist_suppresses_example_key_in_diff() -> None:
    diff = """@@ -0,0 +1 @@
+export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
"""
    report = scan_diff_text(diff)
    assert report.ok is True
    assert report.findings == []


def test_allowlist_suppresses_example_github_pat_in_diff() -> None:
    diff = """@@ -0,0 +1 @@
+ghp_1234567890123456789012345678901234567890
"""
    report = scan_diff_text(diff)
    assert report.ok is True
    assert report.findings == []


def test_allowlist_suppresses_allowlisted_password_value() -> None:
    diff = """@@ -0,0 +1 @@
+password=password123
"""
    report = scan_diff_text(diff)
    assert report.ok is True
    assert report.findings == []


def test_real_password_in_added_line_blocks() -> None:
    diff = """@@ -0,0 +1 @@
+password=Hunter2RealSecret99
"""
    report = scan_diff_text(diff)
    assert report.ok is False
    assert len(report.findings) == 1
    assert report.findings[0].kind == "password_assignment"
    assert report.findings[0].detail == "Hunter2RealSecret99"


def test_scan_diff_record_uses_text_field() -> None:
    record = {
        "harness": "git-diffs",
        "text": "@@ -0,0 +1 @@\n+password=TotallyRealSecret42\n",
    }
    report = scan_diff_record(record)
    assert report.ok is False
    assert report.findings[0].kind == "password_assignment"
