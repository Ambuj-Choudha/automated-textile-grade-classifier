"""Tests for app/helpers/utils.py.

Pure validation functions are tested directly.
Filesystem-dependent functions (get_available_samples, check_required_images)
use tmp_path and monkeypatch cfg paths to avoid touching the real data tree.
"""
import os
import pytest
from app.helpers.utils import validate_input_number, validate_grade


# ---------------------------------------------------------------------------
# validate_input_number
# ---------------------------------------------------------------------------

class TestValidateInputNumber:
    def test_numeric_string_valid(self):
        assert validate_input_number("001") is True
        assert validate_input_number("12345") is True

    def test_alphanumeric_valid(self):
        # isalnum() includes letters — sample IDs like "A1B2" are valid
        assert validate_input_number("A1B2") is True

    def test_empty_string_invalid(self):
        assert validate_input_number("") is False

    def test_whitespace_only_invalid(self):
        assert validate_input_number("   ") is False

    def test_hyphen_invalid(self):
        assert validate_input_number("001-2") is False

    def test_none_equivalent_invalid(self):
        assert validate_input_number(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_grade
# ---------------------------------------------------------------------------

class TestValidateGrade:
    @pytest.mark.parametrize("grade", ["1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"])
    def test_valid_half_step_grades(self, grade):
        assert validate_grade(grade) is True

    def test_empty_string_invalid(self):
        assert validate_grade("") is False

    def test_zero_invalid(self):
        assert validate_grade("0") is False

    def test_six_invalid(self):
        assert validate_grade("6") is False

    def test_non_half_step_invalid(self):
        assert validate_grade("1.2") is False
        assert validate_grade("3.7") is False

    def test_text_invalid(self):
        assert validate_grade("abc") is False

    def test_none_invalid(self):
        assert validate_grade(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_available_samples (needs cfg patching)
# ---------------------------------------------------------------------------

class TestGetAvailableSamples:
    def _patch_input_dir(self, monkeypatch, tmp_path):
        from app.settings import config as cfg
        monkeypatch.setattr(cfg, "get_input_dir", lambda suffix: str(tmp_path / "input_pictures" / suffix))

    def test_returns_empty_when_dir_missing(self, tmp_path, monkeypatch):
        self._patch_input_dir(monkeypatch, tmp_path)
        from app.helpers.utils import get_available_samples
        assert get_available_samples("for_grading") == []

    def test_returns_sorted_unique_samples(self, tmp_path, monkeypatch):
        self._patch_input_dir(monkeypatch, tmp_path)
        d = tmp_path / "input_pictures" / "for_grading"
        d.mkdir(parents=True)
        for name in ("002-1-1-1.png", "001-0-1-1.png", "001-1-1-2.png", "002-1-1-5.png"):
            (d / name).touch()

        from app.helpers.utils import get_available_samples
        result = get_available_samples("for_grading")
        assert result == ["001", "002"]

    def test_ignores_non_png_files(self, tmp_path, monkeypatch):
        self._patch_input_dir(monkeypatch, tmp_path)
        d = tmp_path / "input_pictures" / "for_grading"
        d.mkdir(parents=True)
        (d / "001-1-1-1.png").touch()
        (d / "README.txt").touch()
        (d / "data.csv").touch()

        from app.helpers.utils import get_available_samples
        result = get_available_samples("for_grading")
        assert result == ["001"]

    def test_ignores_files_without_hyphen(self, tmp_path, monkeypatch):
        self._patch_input_dir(monkeypatch, tmp_path)
        d = tmp_path / "input_pictures" / "for_grading"
        d.mkdir(parents=True)
        (d / "001-1-1-1.png").touch()
        (d / "noHyphen.png").touch()

        from app.helpers.utils import get_available_samples
        result = get_available_samples("for_grading")
        assert result == ["001"]

    def test_training_suffix(self, tmp_path, monkeypatch):
        self._patch_input_dir(monkeypatch, tmp_path)
        d = tmp_path / "input_pictures" / "for_training"
        d.mkdir(parents=True)
        (d / "003-1-1-1.png").touch()

        from app.helpers.utils import get_available_samples
        result = get_available_samples("for_training")
        assert result == ["003"]


# ---------------------------------------------------------------------------
# check_required_images
# ---------------------------------------------------------------------------

class TestCheckRequiredImages:
    def _patch_dirs(self, monkeypatch, tmp_path):
        from app.settings import config as cfg
        monkeypatch.setattr(cfg, "get_input_dir", lambda suffix: str(tmp_path / "input_pictures" / suffix))
        monkeypatch.setattr(cfg, "get_difference_dir", lambda suffix: str(tmp_path / "difference_pictures" / suffix))

    def test_all_missing_returns_false(self, tmp_path, monkeypatch):
        self._patch_dirs(monkeypatch, tmp_path)
        (tmp_path / "input_pictures" / "for_grading").mkdir(parents=True)
        (tmp_path / "difference_pictures" / "for_grading").mkdir(parents=True)

        from app.helpers.utils import check_required_images
        result = check_required_images("001", "1", "1", "for_grading")
        assert result["all_input_present"] is False
        assert result["all_difference_present"] is False
        assert result["present_input_images"] == []
        assert result["present_difference_images"] == []

    def test_all_eight_input_images_present(self, tmp_path, monkeypatch):
        self._patch_dirs(monkeypatch, tmp_path)
        inp = tmp_path / "input_pictures" / "for_grading"
        dif = tmp_path / "difference_pictures" / "for_grading"
        inp.mkdir(parents=True)
        dif.mkdir(parents=True)
        for i in range(1, 9):
            (inp / f"001-1-1-{i}.png").touch()

        from app.helpers.utils import check_required_images
        result = check_required_images("001", "1", "1", "for_grading")
        assert result["all_input_present"] is True
        assert len(result["present_input_images"]) == 8

    def test_all_eight_diff_images_present(self, tmp_path, monkeypatch):
        self._patch_dirs(monkeypatch, tmp_path)
        inp = tmp_path / "input_pictures" / "for_grading"
        dif = tmp_path / "difference_pictures" / "for_grading"
        inp.mkdir(parents=True)
        dif.mkdir(parents=True)
        for i in range(1, 9):
            (dif / f"001-1-1-{i}-dif.png").touch()

        from app.helpers.utils import check_required_images
        result = check_required_images("001", "1", "1", "for_grading")
        assert result["all_difference_present"] is True
        assert len(result["present_difference_images"]) == 8

    def test_partial_images_reports_correct_count(self, tmp_path, monkeypatch):
        self._patch_dirs(monkeypatch, tmp_path)
        inp = tmp_path / "input_pictures" / "for_grading"
        inp.mkdir(parents=True)
        (tmp_path / "difference_pictures" / "for_grading").mkdir(parents=True)
        for i in range(1, 5):  # only 4 of 8
            (inp / f"001-1-1-{i}.png").touch()

        from app.helpers.utils import check_required_images
        result = check_required_images("001", "1", "1", "for_grading")
        assert result["all_input_present"] is False
        assert len(result["present_input_images"]) == 4

    def test_different_trial_numbers(self, tmp_path, monkeypatch):
        """Filenames include the trial number — mismatched trial returns nothing."""
        self._patch_dirs(monkeypatch, tmp_path)
        inp = tmp_path / "input_pictures" / "for_grading"
        inp.mkdir(parents=True)
        (tmp_path / "difference_pictures" / "for_grading").mkdir(parents=True)
        for i in range(1, 9):
            (inp / f"001-1-2-{i}.png").touch()  # trial 2

        from app.helpers.utils import check_required_images
        # Looking for trial 1, but files are trial 2
        result = check_required_images("001", "1", "1", "for_grading")
        assert result["all_input_present"] is False
        assert len(result["present_input_images"]) == 0

    def test_training_suffix_uses_correct_dirs(self, tmp_path, monkeypatch):
        self._patch_dirs(monkeypatch, tmp_path)
        inp = tmp_path / "input_pictures" / "for_training"
        dif = tmp_path / "difference_pictures" / "for_training"
        inp.mkdir(parents=True)
        dif.mkdir(parents=True)
        for i in range(1, 9):
            (inp / f"002-1-1-{i}.png").touch()
            (dif / f"002-1-1-{i}-dif.png").touch()

        from app.helpers.utils import check_required_images
        result = check_required_images("002", "1", "1", "for_training")
        assert result["all_input_present"] is True
        assert result["all_difference_present"] is True
