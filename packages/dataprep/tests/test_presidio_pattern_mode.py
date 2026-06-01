"""Pattern-only Presidio engine — no SpacyRecognizer."""

from __future__ import annotations

from llm_dataprep.presidio_custom import clear_analyzer_engine_cache, create_analyzer_engine


def test_pattern_engine_excludes_spacy_recognizer() -> None:
    clear_analyzer_engine_cache()
    engine = create_analyzer_engine(mode="pattern")
    names = {r.__class__.__name__ for r in engine.registry.recognizers}
    assert "SpacyRecognizer" not in names
    assert "EmailRecognizer" in names


def test_pattern_engine_detects_custom_hf_token() -> None:
    clear_analyzer_engine_cache()
    engine = create_analyzer_engine(mode="pattern")
    text = "export HF_TOKEN=hf_" + "a" * 30
    hits = engine.analyze(text=text, language="en")
    kinds = {h.entity_type for h in hits}
    assert "HF_TOKEN" in kinds
