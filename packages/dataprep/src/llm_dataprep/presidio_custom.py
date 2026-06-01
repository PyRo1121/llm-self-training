"""Repo-specific Presidio PatternRecognizers (HF, Cursor, Turso).

Registered on the shared AnalyzerEngine factory used by ``scan_presidio``.
Regex semantics match ``filters._SECRET_PATTERNS`` for hf_token, cursor_token, turso_token.

Modes:
  full    — predefined recognizers + spaCy NER (slow)
  pattern — pattern/checksum recognizers only (no spaCy; ~10–50× faster)
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from presidio_analyzer import Pattern, PatternRecognizer

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.recognizer_registry import RecognizerRegistry

PresidioEngineMode = Literal["full", "pattern"]

CUSTOM_ENTITIES: frozenset[str] = frozenset({"HF_TOKEN", "CURSOR_TOKEN", "TURSO_TOKEN"})

_HF_PATTERN = Pattern(name="hf_token", regex=r"hf_[A-Za-z0-9]{20,}", score=0.9)
_CURSOR_PATTERN = Pattern(
    name="cursor_token",
    regex=r"cursor_[A-Za-z0-9_\-]{20,}",
    score=0.9,
)
_TURSO_PATTERNS = (
    Pattern(name="libsql_url", regex=r"(?i)libsql://[^\s\"']+", score=0.9),
    Pattern(name="turso_prefix", regex=r"(?i)turso_[A-Za-z0-9_]{16,}", score=0.9),
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


def _strip_spacy_recognizers(registry: RecognizerRegistry) -> None:
    registry.recognizers = [
        r for r in registry.recognizers if r.__class__.__name__ != "SpacyRecognizer"
    ]


@lru_cache(maxsize=1)
def create_analyzer_engine_full() -> AnalyzerEngine:
    from presidio_analyzer import AnalyzerEngine

    engine = AnalyzerEngine()
    register_custom_recognizers(engine.registry)
    return engine


@lru_cache(maxsize=1)
def create_analyzer_engine_pattern() -> AnalyzerEngine:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.recognizer_registry import RecognizerRegistry

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"])
    _strip_spacy_recognizers(registry)
    register_custom_recognizers(registry)
    return AnalyzerEngine(registry=registry, supported_languages=["en"])


def create_analyzer_engine(*, mode: PresidioEngineMode = "full") -> AnalyzerEngine:
    if mode == "pattern":
        return create_analyzer_engine_pattern()
    return create_analyzer_engine_full()


def clear_analyzer_engine_cache() -> None:
    create_analyzer_engine_full.cache_clear()
    create_analyzer_engine_pattern.cache_clear()
