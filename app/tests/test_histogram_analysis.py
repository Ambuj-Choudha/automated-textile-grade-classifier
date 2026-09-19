"""Tests for app/analysis/histogram_analysis.py pure helpers.

The tests cover:
  - _coerce_grade: grade parsing and half-step rounding
  - _restamp_grade: replacing the Grade column in rows
  - _write_csv_with_headers: file writing, append mode, parent-dir creation
  - analyze_difference_images: CSV output shape (requires cv2 + PIL; skipped if absent)
"""
import csv
import os
import pytest

# _coerce_grade, _restamp_grade, _write_csv_with_headers are pure Python;
# import them directly (no mocking needed).
from app.analysis.histogram_analysis import (
    _coerce_grade,
    _restamp_grade,
    _write_csv_with_headers,
)


# ---------------------------------------------------------------------------
# _coerce_grade
# ---------------------------------------------------------------------------

class TestCoerceGrade:
    def test_integer_string(self):
        assert _coerce_grade("3", "pilling") == pytest.approx(3.0)

    def test_float_string(self):
        assert _coerce_grade("3.5", "pilling") == pytest.approx(3.5)

    def test_bare_float(self):
        assert _coerce_grade(2.5, "matting") == pytest.approx(2.5)

    def test_comma_decimal(self):
        # German-locale CSV cells sometimes arrive as "3,5"
        assert _coerce_grade("3,5", "fuzzing") == pytest.approx(3.5)

    def test_rounds_up_to_nearest_half(self):
        assert _coerce_grade(2.6, "pilling") == pytest.approx(2.5)

    def test_rounds_down_to_nearest_half(self):
        assert _coerce_grade(2.4, "pilling") == pytest.approx(2.5)

    def test_whole_number_float(self):
        assert _coerce_grade(4.0, "pilling") == pytest.approx(4.0)

    def test_invalid_string_returns_none(self):
        assert _coerce_grade("abc", "pilling") is None

    def test_none_returns_none(self):
        assert _coerce_grade(None, "pilling") is None

    def test_empty_string_returns_none(self):
        assert _coerce_grade("", "pilling") is None


# ---------------------------------------------------------------------------
# _restamp_grade
# ---------------------------------------------------------------------------

class TestRestampGrade:
    def test_replaces_last_column_with_new_grade(self):
        rows = [["a.png", 100.0, 10.0, 255.0, 80.0, 2.0]]
        result = _restamp_grade(rows, 3.5)
        assert result[0] == ["a.png", 100.0, 10.0, 255.0, 80.0, 3.5]

    def test_prefix_columns_unchanged(self):
        rows = [["img.png", 1.0, 2.0, 3.0, 4.0, 5.0]]
        result = _restamp_grade(rows, 1.5)
        assert result[0][:-1] == ["img.png", 1.0, 2.0, 3.0, 4.0]

    def test_multiple_rows(self):
        rows = [
            ["a.png", 100.0, 10.0, 255.0, 80.0, 2.0],
            ["b.png", 120.0, 12.0, 240.0, 90.0, 2.0],
        ]
        result = _restamp_grade(rows, 4.0)
        assert all(r[-1] == 4.0 for r in result)

    def test_original_rows_not_mutated(self):
        rows = [["a.png", 100.0, 10.0, 255.0, 80.0, 2.0]]
        _restamp_grade(rows, 3.5)
        assert rows[0][-1] == 2.0

    def test_empty_input_returns_empty(self):
        assert _restamp_grade([], 3.0) == []


# ---------------------------------------------------------------------------
# _write_csv_with_headers
# ---------------------------------------------------------------------------

class TestWriteCsvWithHeaders:
    def test_creates_file_with_header_and_data(self, tmp_path):
        path = tmp_path / "out.csv"
        _write_csv_with_headers(str(path), ["A", "B", "C"], [[1, 2, 3], [4, 5, 6]])
        rows = list(csv.reader(open(str(path))))
        assert rows[0] == ["A", "B", "C"]
        assert rows[1] == ["1", "2", "3"]
        assert rows[2] == ["4", "5", "6"]

    def test_overwrites_existing_file_by_default(self, tmp_path):
        path = tmp_path / "out.csv"
        _write_csv_with_headers(str(path), ["X"], [[1]])
        _write_csv_with_headers(str(path), ["X"], [[2]])
        rows = list(csv.reader(open(str(path))))
        # Only new content present
        assert rows == [["X"], ["2"]]

    def test_append_mode_does_not_repeat_header(self, tmp_path):
        path = tmp_path / "out.csv"
        _write_csv_with_headers(str(path), ["A", "B"], [[1, 2]])
        _write_csv_with_headers(str(path), ["A", "B"], [[3, 4]], append_mode=True)
        rows = list(csv.reader(open(str(path))))
        assert rows[0] == ["A", "B"]  # header appears once
        assert rows[1] == ["1", "2"]
        assert rows[2] == ["3", "4"]
        assert len(rows) == 3

    def test_append_mode_on_empty_file_writes_header(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("")  # exists but empty
        _write_csv_with_headers(str(path), ["A"], [[1]], append_mode=True)
        rows = list(csv.reader(open(str(path))))
        assert rows[0] == ["A"]
        assert rows[1] == ["1"]

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "subdir" / "nested" / "out.csv"
        _write_csv_with_headers(str(path), ["X"], [[42]])
        assert path.exists()

    def test_multiple_appends_accumulate_rows(self, tmp_path):
        path = tmp_path / "out.csv"
        for i in range(1, 4):
            _write_csv_with_headers(str(path), ["N"], [[i]], append_mode=True)
        rows = list(csv.reader(open(str(path))))
        assert rows[0] == ["N"]
        assert [r[0] for r in rows[1:]] == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# analyze_difference_images — CSV output shape
# Requires cv2 (OpenCV) and PIL; skipped if either is unavailable.
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2", reason="cv2 not installed")
PIL_Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")


def _make_gray_png(path, value: int = 128, size: tuple = (10, 10)) -> None:
    """Write a small uniform-gray PNG using Pillow."""
    import numpy as np
    from PIL import Image
    img = Image.fromarray(
        __import__("numpy").full(size, value, dtype=__import__("numpy").uint8), mode="L"
    )
    img.save(str(path))


class TestAnalyzeDifferenceImages:
    """Tests that exercise analyze_difference_images end-to-end.

    The entire app uses paths relative to the CWD (project root).
    We set CWD to tmp_path so all relative-path I/O lands in an isolated
    temp tree without touching the real data directories.
    """

    def _setup_diff_images(self, base: "Path", sample="001", stage="1", trial="1"):
        """Create 8 small gray PNG diff images under base/data/difference_pictures/for_training/."""
        from app.settings import config as cfg
        diff_dir = base / cfg.get_difference_dir(cfg.SUFFIX_TRAINING)
        diff_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 9):
            _make_gray_png(diff_dir / f"{sample}-{stage}-{trial}-{i}-dif.png", value=128)
        return diff_dir

    def test_writes_per_image_and_averaged_csvs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._setup_diff_images(tmp_path)

        output_dir = tmp_path / "training_features"
        from app.analysis.histogram_analysis import analyze_difference_images
        analyze_difference_images("001", "1", "1", grades={"pilling": 3.0}, output_dir=str(output_dir))

        assert (output_dir / "pilling_per_image_features.csv").exists(), "per-image CSV not created"
        assert (output_dir / "pilling_averaged_features.csv").exists(), "averaged CSV not created"

    def test_per_image_csv_has_eight_data_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._setup_diff_images(tmp_path)

        output_dir = tmp_path / "training_features"
        from app.analysis.histogram_analysis import analyze_difference_images
        analyze_difference_images("001", "1", "1", grades={"pilling": 3.0}, output_dir=str(output_dir))

        rows = list(csv.reader(open(str(output_dir / "pilling_per_image_features.csv"))))
        assert len(rows) == 9  # header + 8 data rows
        assert rows[0] == list(("Image", "Mean", "Std", "Max", "Mode", "Grade"))

    def test_averaged_csv_has_one_data_row(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._setup_diff_images(tmp_path)

        output_dir = tmp_path / "training_features"
        from app.analysis.histogram_analysis import analyze_difference_images
        analyze_difference_images("001", "1", "1", grades={"pilling": 3.0}, output_dir=str(output_dir))

        rows = list(csv.reader(open(str(output_dir / "pilling_averaged_features.csv"))))
        assert len(rows) == 2  # header + 1 average row
        assert float(rows[1][-1]) == pytest.approx(3.0)  # Grade column

    def test_multiple_grades_write_separate_csvs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._setup_diff_images(tmp_path)

        output_dir = tmp_path / "training_features"
        from app.analysis.histogram_analysis import analyze_difference_images
        analyze_difference_images(
            "001", "1", "1",
            grades={"pilling": 3.0, "matting": 2.5, "fuzzing": 4.0},
            output_dir=str(output_dir),
        )

        for grade in ("pilling", "matting", "fuzzing"):
            assert (output_dir / f"{grade}_per_image_features.csv").exists()
            assert (output_dir / f"{grade}_averaged_features.csv").exists()

    def test_grades_stamped_correctly_per_grade(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._setup_diff_images(tmp_path)

        output_dir = tmp_path / "training_features"
        from app.analysis.histogram_analysis import analyze_difference_images
        analyze_difference_images(
            "001", "1", "1",
            grades={"pilling": 2.0, "matting": 4.5},
            output_dir=str(output_dir),
        )

        pilling_avg = list(csv.reader(open(str(output_dir / "pilling_averaged_features.csv"))))
        assert float(pilling_avg[1][-1]) == pytest.approx(2.0)

        matting_avg = list(csv.reader(open(str(output_dir / "matting_averaged_features.csv"))))
        assert float(matting_avg[1][-1]) == pytest.approx(4.5)

    def test_appends_across_multiple_calls(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        output_dir = tmp_path / "training_features"
        from app.analysis.histogram_analysis import analyze_difference_images

        for trial in ("1", "2"):
            self._setup_diff_images(tmp_path, trial=trial)
            analyze_difference_images("001", "1", trial, grades={"pilling": 3.0}, output_dir=str(output_dir))

        rows = list(csv.reader(open(str(output_dir / "pilling_per_image_features.csv"))))
        assert len(rows) == 17  # header + 8 rows/trial × 2 trials
