"""Repo-specific Presidio PatternRecognizers (HF, Cursor, Turso).

Registered on the shared AnalyzerEngine factory used by ``scan_presidio``.
Regex semantics match ``filters._SECRET_PATTERNS`` for hf_token, cursor_token, turso_token.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from presidio_analyzer import Pattern, PatternRecognizer

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.recognizer_registry import RecognizerRegistry

CUSTOM_ENTITIES: frozenset[str] = frozenset({"HF_TOKEN", "CURSOR_TOKEN", "TURSO_TOKEN"})

_HF_PATTERN = Pattern(name="hf_token", regex=r"hf_[A-Za-z0-9]{20,}", score=0.9)
_CURSOR_PATTERN = Pattern(
    name="cursor_token",
    regex=r"cursor_[A-Za-z0-9_\-]{20,}",
    score=0.9,
)
_TURSO_PATTERNS = (
    Pattern(name="libsql_url", regex=r"(?i)libsql://[^\s\"']+", score=0.9),
    Pattern(name="turso_prefix", regex=r"turso_[A-Za-z0-9_]{16,}", score=0.9),
)


def build_custom_recognizers() -> list[PatternRecognizer]:
    return [
        PatternRecognizer(supported_entity="HF_TOKEN", patterns=[_HF_PATTERN]),
        PatternRecognizer(supported_entity="CURSOR_TOKEN", patterns=[_CURSOR_PATTERN]),
        PatternRecognizer(supported_entity="TURSO_TOKEN", patterns=list(_TURSO_PATTERNS)),
    ]


def register_custom_recognizers(registry: RecognizerRegistry) -> None:
    for recognizer in build_custom_recognizers():
        registry.add_recognizer(recognizer)


@lru_cache(maxsize=1)
def create_analyzer_engine() -> AnalyzerEngine:
    from presidio_analyzer import AnalyzerEngine

    engine = AnalyzerEngine()
    register_custom_recognizers(engine.registry)
    return engine
