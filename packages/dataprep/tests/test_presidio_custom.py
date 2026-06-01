"""Presidio custom PatternRecognizers — no spaCy required."""

from __future__ import annotations

import pytest

from llm_dataprep.presidio_custom import (
    CUSTOM_ENTITIES,
    build_custom_recognizers,
    create_analyzer_engine,
    register_custom_recognizers,
)


def _recognizers_by_entity() -> dict[str, object]:
    return {r.supported_entities[0]: r for r in build_custom_recognizers()}


@pytest.mark.parametrize(
    ("entity", "text"),
    [
        ("HF_TOKEN", "export HF_TOKEN=hf_" + "a" * 30),
        ("CURSOR_TOKEN", "key cursor_" + "b" * 25 + " here"),
        ("TURSO_TOKEN", "db libsql://user:pass@host-abc.turso.io/v2"),
        ("TURSO_TOKEN", "turso_" + "c" * 20),
        ("TURSO_TOKEN", "TURSO_" + "d" * 20),
    ],
)
def test_pattern_recognizer_detects_without_spacy(entity: str, text: str) -> None:
    rec = _recognizers_by_entity()[entity]
    hits = rec.analyze(text, [entity])
    assert hits, f"expected {entity} in {text!r}"
    assert hits[0].entity_type == entity
    assert hits[0].score >= 0.85


def test_custom_entities_cover_all_recognizers() -> None:
    assert CUSTOM_ENTITIES == frozenset(_recognizers_by_entity())


def test_register_custom_recognizers() -> None:
    added: list[str] = []

    class _Registry:
        def add_recognizer(self, recognizer: object) -> None:
            added.append(recognizer.supported_entities[0])  # type: ignore[attr-defined]

    register_custom_recognizers(_Registry())  # type: ignore[arg-type]
    assert set(added) == set(CUSTOM_ENTITIES)


def test_create_analyzer_engine_registers_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    create_analyzer_engine.cache_clear()

    class _Registry:
        def __init__(self) -> None:
            self.added: list[str] = []

        def add_recognizer(self, recognizer: object) -> None:
            self.added.append(recognizer.supported_entities[0])  # type: ignore[attr-defined]

    class _Engine:
        def __init__(self) -> None:
            self.registry = _Registry()

    monkeypatch.setattr("presidio_analyzer.AnalyzerEngine", _Engine)
    engine = create_analyzer_engine()
    assert set(engine.registry.added) == set(CUSTOM_ENTITIES)
