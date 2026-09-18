"""Tests for app/settings/config.py path builders.

All functions are pure (no I/O), so no fixtures are needed.
"""
import os
import pytest
from app.settings import config as cfg


class TestItanetPaths:
    def test_run_dir_structure(self):
        for grade in ("pilling", "matting", "fuzzing"):
            result = cfg.get_itanet_run_dir(grade)
            assert result == os.path.join("models", "itanet", grade, "run")

    def test_fls_dir_structure(self):
        for grade in ("pilling", "matting", "fuzzing"):
            result = cfg.get_itanet_fls_dir(grade)
            assert result == os.path.join("models", "itanet", grade, "data")

    def test_archive_dir_structure(self):
        for grade in ("pilling", "matting", "fuzzing"):
            result = cfg.get_itanet_archive_dir(grade)
            assert result == os.path.join("models", "itanet", "archive", grade)

    def test_run_and_fls_share_grade_parent(self):
        """run/ and data/ must be siblings under models/itanet/<grade>/."""
        run = cfg.get_itanet_run_dir("pilling")
        fls = cfg.get_itanet_fls_dir("pilling")
        assert os.path.dirname(run) == os.path.dirname(fls)


class TestDataPaths:
    def test_input_dir_for_grading(self):
        assert cfg.get_input_dir("for_grading") == os.path.join(
            "data", "input_pictures", "for_grading"
        )

    def test_input_dir_for_training(self):
        assert cfg.get_input_dir("for_training") == os.path.join(
            "data", "input_pictures", "for_training"
        )

    def test_difference_dir_for_grading(self):
        assert cfg.get_difference_dir("for_grading") == os.path.join(
            "data", "difference_pictures", "for_grading"
        )

    def test_difference_dir_for_training(self):
        assert cfg.get_difference_dir("for_training") == os.path.join(
            "data", "difference_pictures", "for_training"
        )

    def test_suffix_constants(self):
        assert cfg.SUFFIX_GRADING == "for_grading"
        assert cfg.SUFFIX_TRAINING == "for_training"


class TestFilenameBuilders:
    def test_input_filename_format(self):
        assert cfg.make_input_filename("001", "2", "3", 4) == "001-2-3-4.png"

    def test_input_filename_stage_zero(self):
        assert cfg.make_input_filename("A1B2", "0", "10", 8) == "A1B2-0-10-8.png"

    def test_difference_filename_format(self):
        assert cfg.make_difference_filename("001", "2", "3", 4) == "001-2-3-4-dif.png"

    def test_difference_filename_ends_with_dif(self):
        name = cfg.make_difference_filename("S", "1", "1", 1)
        assert name.endswith("-dif.png")

    def test_all_eight_positions(self):
        for pos in range(1, 9):
            name = cfg.make_input_filename("001", "1", "1", pos)
            assert name == f"001-1-1-{pos}.png"
            diff = cfg.make_difference_filename("001", "1", "1", pos)
            assert diff == f"001-1-1-{pos}-dif.png"


class TestConstants:
    def test_grades_tuple_contains_all_three(self):
        assert set(cfg.GRADES) == {"pilling", "matting", "fuzzing"}

    def test_grades_is_ordered_as_expected(self):
        assert cfg.GRADES[0] == "pilling"
        assert cfg.GRADES[1] == "matting"
        assert cfg.GRADES[2] == "fuzzing"

    def test_clip_range_valid(self):
        lo, hi = cfg.CLIP_RANGE
        assert lo == 1.0
        assert hi == 5.0

    def test_feature_columns_present(self):
        for col in ("Mean", "Std", "Max", "Mode"):
            assert col in cfg.FEATURE_COLUMNS
