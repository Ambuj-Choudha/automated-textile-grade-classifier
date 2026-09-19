"""Tests for translation-key coverage.

Runs at dev/CI time (not at server boot) — translation keys are static and
only change when a developer edits config.GRADES or translations.py.
"""
import pytest

from app.reporting.translations import translations, validate_grade_translations
from app.settings import config as cfg


def test_configured_grades_have_all_required_translation_keys():
    """Every grade in cfg.GRADES must have its trio of keys in the English dict."""
    validate_grade_translations(cfg.GRADES)


def test_validator_raises_on_missing_grade_key():
    with pytest.raises(RuntimeError) as exc:
        validate_grade_translations(("nonexistent_grade",))
    msg = str(exc.value)
    assert "nonexistent_grade_grade_number" in msg
    assert "nonexistent_grade_grade_result" in msg
    assert "invalid_nonexistent_grade_grade" in msg


def test_all_non_english_dicts_have_same_keys_as_english():
    """A language falling back silently to the key name looks like broken UI."""
    en_keys = set(translations["en"].keys())
    for lang, table in translations.items():
        if lang == "en":
            continue
        missing = en_keys - set(table.keys())
        assert not missing, f"'{lang}' is missing keys: {sorted(missing)}"
