"""Unit tests for itanet_recall + itanet_training.

These tests exercise the pure-Python glue — file formatters, CSV loading,
enum values, guard rails — without needing the actual DLL or a trained
network. Two goals:

1. Verify the recent API changes (decimal threading, .NET-shape guard,
   NaN detection, DLL guard, seed removal, run/data folder split).
2. Catch future regressions:
   - HAUPTDEF.H enum drift — if the DLL changes ``CREATE_NEW_NET`` from
     1 to something else, ``test_enum_values_pin_hauptdef_h`` breaks first.
   - On-disk contract drift — if RecallData.dat's byte layout changes
     (whole-number encoding, decimal separator, CRLF), the byte-level tests
     fail immediately instead of the DLL access-violating at recall time.

Run:  env/Scripts/python.exe -m pytest app/tests/test_itanet_unit.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from itanet_recall import (  # noqa: E402
    CREATE_NEW_NET,
    ITANetFileReader,
    ITANetFiles,
    ITANetFileWriter,
    ITANetManager,
    KOMMA_ZU_PUNKT,
    LOAD_FROM_FILE,
    PUNKT_ZU_KOMMA,
    SHUFFLE_OFF,
    SHUFFLE_ON,
    _load_recall_csv,
    change_directory,
    decimal_to_komma_punkt,
    _grade_fls_dir,
    _grade_run_dir,
    default_dll_path,
    format_number,
    load_itanet_dll,
    predict_from_csv,
)
from itanet_training import (  # noqa: E402
    PRISTINE_NET_MAX_BYTES,
    _archive_existing_net,
    _fmt_grade,
    _fmt_value,
    _guard_net_matches_net_type,
    csv_to_training_dats,
)


# --------------------------------------------------------------------------- #
# Enum values pinned to ITA-net-repo INCLUDE/HAUPTDEF.H
# --------------------------------------------------------------------------- #

def test_enum_values_pin_hauptdef_h():
    """Guards against silent drift if HAUPTDEF.H is renumbered on the C side.

    If any of these fail, the DLL will still accept the call but do the
    wrong thing (e.g. train when we meant recall). Update the C side and
    this test together.
    """
    assert CREATE_NEW_NET == 1
    assert LOAD_FROM_FILE == 2
    assert KOMMA_ZU_PUNKT == 0
    assert PUNKT_ZU_KOMMA == 1
    assert SHUFFLE_OFF == 0
    assert SHUFFLE_ON == 1


# --------------------------------------------------------------------------- #
# Formatters — the DLL's pattern loader is byte-picky
# --------------------------------------------------------------------------- #

class TestFormatNumber:
    def test_comma_whole_keeps_trailing_zero(self):
        # bare "1" causes access violation inside run_recall_session
        assert format_number(1.0, "comma") == "1,0"
        assert format_number(104.0, "comma") == "104,0"

    def test_comma_fractional(self):
        assert format_number(11.35, "comma") == "11,35"

    def test_point_whole_keeps_trailing_zero(self):
        assert format_number(1.0, "point") == "1.0"
        assert format_number(104.0, "point") == "104.0"

    def test_point_fractional(self):
        assert format_number(11.35, "point") == "11.35"

    def test_negative_whole(self):
        assert format_number(-3.0, "comma") == "-3,0"
        assert format_number(-3.0, "point") == "-3.0"


class TestFmtValue:
    def test_comma_and_point(self):
        assert _fmt_value("1", "comma") == "1,0"
        assert _fmt_value("1", "point") == "1.0"
        assert _fmt_value("11.35", "comma") == "11,35"
        assert _fmt_value("11.35", "point") == "11.35"

    def test_comma_string_input_normalized(self):
        # csv cells sometimes arrive as German comma-decimal strings
        assert _fmt_value("11,35", "point") == "11.35"


class TestFmtGrade:
    def test_whole_grade_bare_int(self):
        # grades don't need the ",0" suffix — the pattern loader accepts bare
        # ints in the grade file
        assert _fmt_grade("4", "comma") == "4"
        assert _fmt_grade("4", "point") == "4"

    def test_fractional_grade(self):
        assert _fmt_grade("3.5", "comma") == "3,5"
        assert _fmt_grade("3.5", "point") == "3.5"


def test_decimal_to_komma_punkt_mapping():
    assert decimal_to_komma_punkt("comma") == KOMMA_ZU_PUNKT
    assert decimal_to_komma_punkt("point") == PUNKT_ZU_KOMMA


# --------------------------------------------------------------------------- #
# ITANetFileWriter — on-disk byte contract
# --------------------------------------------------------------------------- #

class TestWriteDataFile:
    def test_header_and_crlf_comma(self, tmp_path: Path):
        path = tmp_path / "RecallData.dat"
        ITANetFileWriter.write_data_file(str(path), [[11.35, 1.0], [12.0, 2.5]], "comma")
        body = path.read_bytes()
        # header: <n>\t<cols>\r\n
        assert body.startswith(b"2\t2\r\n")
        # all lines end in CRLF
        assert body.endswith(b"\r\n")
        # comma decimals, whole numbers with trailing ,0
        assert b"11,35\t1,0\r\n" in body
        assert b"12,0\t2,5\r\n" in body
        # no bare ints leaked through
        assert b"\t1\r\n" not in body
        assert b"\t12\r\n" not in body

    def test_header_and_crlf_point(self, tmp_path: Path):
        path = tmp_path / "RecallData.dat"
        ITANetFileWriter.write_data_file(str(path), [[11.35, 1.0]], "point")
        body = path.read_bytes()
        assert body.startswith(b"1\t2\r\n")
        assert b"11.35\t1.0\r\n" in body
        # no accidental commas when point mode is asked for
        assert b"," not in body

    def test_creates_parent_dir(self, tmp_path: Path):
        path = tmp_path / "subdir" / "RecallData.dat"
        ITANetFileWriter.write_data_file(str(path), [[1.0]], "comma")
        assert path.exists()


class TestWriteFileList:
    def test_recall_fls_format(self, tmp_path: Path):
        path = tmp_path / "recall.fls"
        ITANetFileWriter.write_filelist(
            str(path),
            [ITANetFiles.RECALL_INPUT, ITANetFiles.RECALL_OUTPUT,
             ITANetFiles.TRAIN_INPUT, ITANetFiles.TRAIN_TARGET, ITANetFiles.NET],
        )
        lines = path.read_text().splitlines()
        assert lines == [
            "RecallData.dat", "Recall_output.dat",
            "TrainingData.dat", "TrainingGrades.dat", "Neuronalesnetz.NET",
        ]


# --------------------------------------------------------------------------- #
# ITANetFileReader.read_recall_output — NaN detection is the recent change
# --------------------------------------------------------------------------- #

class TestReadRecallOutput:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ITANetFileReader.read_recall_output(str(tmp_path / "nope.dat"))

    def test_reads_header_plus_values(self, tmp_path: Path):
        path = tmp_path / "Recall_output.dat"
        path.write_text("3 1\n2.5\n3.0\n4.5\n")
        assert ITANetFileReader.read_recall_output(str(path)) == [2.5, 3.0, 4.5]

    def test_reads_without_header(self, tmp_path: Path):
        # Not all outputs prepend a "<n> <cols>" line; the reader auto-detects
        path = tmp_path / "Recall_output.dat"
        path.write_text("2.5\n3.0\n")
        assert ITANetFileReader.read_recall_output(str(path)) == [2.5, 3.0]

    def test_handles_comma_decimals(self, tmp_path: Path):
        path = tmp_path / "Recall_output.dat"
        path.write_text("2 1\n2,5\n3,0\n")
        assert ITANetFileReader.read_recall_output(str(path)) == [2.5, 3.0]

    def test_raises_on_all_nan(self, tmp_path: Path):
        path = tmp_path / "Recall_output.dat"
        path.write_text("2 1\n-nan(ind)\n-nan(ind)\n")
        with pytest.raises(RuntimeError, match="diverged weights"):
            ITANetFileReader.read_recall_output(str(path))

    def test_raises_on_partial_nan(self, tmp_path: Path):
        """Any NaN row is a problem — hiding partial divergence behind a
        downstream 'prediction count mismatch' obscures the real cause."""
        path = tmp_path / "Recall_output.dat"
        path.write_text("3 1\n2.5\n-nan(ind)\n3.0\n")
        with pytest.raises(RuntimeError, match=r"NaN on 1 of 3 lines"):
            ITANetFileReader.read_recall_output(str(path))

    def test_raises_on_bare_nan(self, tmp_path: Path):
        # not just -nan(ind); the CRT sometimes prints "nan" bare
        path = tmp_path / "Recall_output.dat"
        path.write_text("1 1\nnan\n")
        with pytest.raises(RuntimeError, match="diverged weights"):
            ITANetFileReader.read_recall_output(str(path))

    def test_empty_file(self, tmp_path: Path):
        path = tmp_path / "Recall_output.dat"
        path.write_text("")
        assert ITANetFileReader.read_recall_output(str(path)) == []


# --------------------------------------------------------------------------- #
# ITANetManager / load_itanet_dll — DLL guard
# --------------------------------------------------------------------------- #

class TestITANetManagerInit:
    def test_missing_dll_raises_with_actionable_message(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="build.py"):
            ITANetManager(
                run_dir=str(tmp_path / "run"),
                dll_path=str(tmp_path / "does_not_exist.dll"),
            )

    def test_creates_run_and_fls_dirs(self, tmp_path: Path):
        # A dummy file stands in for the DLL — presence check only, no load.
        fake_dll = tmp_path / "fake.dll"
        fake_dll.write_bytes(b"")
        run = tmp_path / "run"
        fls = tmp_path / "data"
        ITANetManager(run_dir=str(run), dll_path=str(fake_dll), fls_dir=str(fls))
        assert run.is_dir()
        assert fls.is_dir()

    def test_default_fls_dir_is_sibling_of_run(self, tmp_path: Path):
        """The DLL hardcodes ``..\\data\\recall.fls`` — verify the default
        matches that layout."""
        fake_dll = tmp_path / "fake.dll"
        fake_dll.write_bytes(b"")
        run = tmp_path / "itanet" / "run"
        mgr = ITANetManager(run_dir=str(run), dll_path=str(fake_dll))
        assert mgr.fls_dir == run.parent / "data"


def test_load_itanet_dll_missing_raises():
    with pytest.raises(FileNotFoundError, match="build.py"):
        load_itanet_dll("/tmp/definitely_not_here.dll")


# --------------------------------------------------------------------------- #
# csv_to_training_dats — decimal threading (the bug from the review)
# --------------------------------------------------------------------------- #

class TestCsvToTrainingDats:
    def _sample_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "features.csv"
        csv_path.write_text(
            "Image,Mean,Std,Max,Mode,Grade\n"
            "a.png,11.35,8.79,104.0,1.0,4\n"
            "b.png,12.5,9.16,132.0,1.0,3.5\n"
        )
        return csv_path

    def test_comma_writes_german_decimals(self, tmp_path: Path):
        csv_path = self._sample_csv(tmp_path)
        data_out = tmp_path / "TrainingData.dat"
        grades_out = tmp_path / "TrainingGrades.dat"
        n, k = csv_to_training_dats(csv_path, data_out, grades_out, decimal="comma")
        assert (n, k) == (2, 4)

        data_body = data_out.read_bytes()
        assert data_body.startswith(b"2\t4\r\n")
        assert b"11,35\t8,79\t104,0\t1,0\r\n" in data_body
        assert b"." not in data_body  # comma mode → no dots

        grades_body = grades_out.read_bytes()
        # whole grade → bare int; fractional grade → comma decimal
        assert grades_body == b"2\t1\r\n4\r\n3,5\r\n"

    def test_point_writes_dot_decimals(self, tmp_path: Path):
        """The bug this catches: pre-fix, _fmt_value/_fmt_grade ignored the
        decimal arg and always wrote commas, so ``decimal='point'`` produced
        a .dat file the DLL couldn't parse with ``komma_punkt=1``."""
        csv_path = self._sample_csv(tmp_path)
        data_out = tmp_path / "TrainingData.dat"
        grades_out = tmp_path / "TrainingGrades.dat"
        csv_to_training_dats(csv_path, data_out, grades_out, decimal="point")

        data_body = data_out.read_bytes()
        assert b"11.35\t8.79\t104.0\t1.0\r\n" in data_body
        assert b"," not in data_body  # point mode → no commas

        grades_body = grades_out.read_bytes()
        assert grades_body == b"2\t1\r\n4\r\n3.5\r\n"

    def test_missing_grade_column_raises(self, tmp_path: Path):
        csv_path = tmp_path / "no_grade.csv"
        csv_path.write_text("Image,Mean\na.png,11.35\n")
        with pytest.raises(ValueError, match="Grade"):
            csv_to_training_dats(csv_path, tmp_path / "d.dat", tmp_path / "g.dat")


# --------------------------------------------------------------------------- #
# _guard_net_matches_net_type — the new .NET-shape check
# --------------------------------------------------------------------------- #

class TestGuardNetType:
    def _make_net(self, tmp_path: Path, size: int) -> Path:
        p = tmp_path / "Neuronalesnetz.NET"
        p.write_bytes(b"x" * size)
        return p

    def test_pristine_ok_for_create_new(self, tmp_path: Path):
        # Should not raise
        _guard_net_matches_net_type(self._make_net(tmp_path, 26), CREATE_NEW_NET)

    def test_weights_ok_for_load(self, tmp_path: Path):
        _guard_net_matches_net_type(self._make_net(tmp_path, 5000), LOAD_FROM_FILE)

    def test_weights_rejected_for_create_new(self, tmp_path: Path):
        """The dangerous case: user forgot to restore pristine .NET and asked
        for CREATE_NEW_NET — new_net would misread the weights-bearing file
        and produce garbage silently."""
        with pytest.raises(RuntimeError, match="weights-bearing"):
            _guard_net_matches_net_type(
                self._make_net(tmp_path, 5000), CREATE_NEW_NET
            )

    def test_pristine_rejected_for_load(self, tmp_path: Path):
        """The reverse: LOAD_FROM_FILE on a pristine .NET — load_net crashes
        or produces nonsense."""
        with pytest.raises(RuntimeError, match="pristine"):
            _guard_net_matches_net_type(
                self._make_net(tmp_path, 26), LOAD_FROM_FILE
            )

    def test_boundary_at_threshold(self, tmp_path: Path):
        """Files right at the 500-byte boundary count as pristine (<=), so
        the boundary itself passes CREATE_NEW_NET."""
        _guard_net_matches_net_type(
            self._make_net(tmp_path, PRISTINE_NET_MAX_BYTES), CREATE_NEW_NET
        )
        with pytest.raises(RuntimeError):
            _guard_net_matches_net_type(
                self._make_net(tmp_path, PRISTINE_NET_MAX_BYTES + 1),
                CREATE_NEW_NET,
            )


# --------------------------------------------------------------------------- #
# _archive_existing_net — must archive .TRN alongside .NET
# --------------------------------------------------------------------------- #

class TestArchiveExistingNet:
    def test_archives_both_files(self, tmp_path: Path):
        run = tmp_path / "run"
        run.mkdir()
        (run / ITANetFiles.NET).write_bytes(b"net-bytes")
        (run / ITANetFiles.TRN).write_bytes(b"trn-bytes")

        archive_root = tmp_path / "archive"
        dest = _archive_existing_net(run, archive_root)

        assert dest is not None
        assert (dest / ITANetFiles.NET).read_bytes() == b"net-bytes"
        assert (dest / ITANetFiles.TRN).read_bytes() == b"trn-bytes"

    def test_no_net_returns_none(self, tmp_path: Path):
        run = tmp_path / "run"
        run.mkdir()
        # .TRN exists but no .NET — nothing to archive
        (run / ITANetFiles.TRN).write_bytes(b"trn-bytes")
        assert _archive_existing_net(run, tmp_path / "archive") is None

    def test_missing_trn_still_archives_net_with_warning(
        self, tmp_path: Path, capsys
    ):
        run = tmp_path / "run"
        run.mkdir()
        (run / ITANetFiles.NET).write_bytes(b"net-bytes")

        dest = _archive_existing_net(run, tmp_path / "archive")
        assert dest is not None
        assert (dest / ITANetFiles.NET).exists()
        assert not (dest / ITANetFiles.TRN).exists()

        captured = capsys.readouterr()
        assert "warning" in captured.out.lower()
        assert "Neuronalesnetz.TRN" in captured.out

    def test_stamp_is_utc_iso_ish(self, tmp_path: Path):
        run = tmp_path / "run"
        run.mkdir()
        (run / ITANetFiles.NET).write_bytes(b"net-bytes")
        (run / ITANetFiles.TRN).write_bytes(b"trn-bytes")

        dest = _archive_existing_net(run, tmp_path / "archive")
        # Format: 20260823T170104Z
        import re
        assert re.fullmatch(r"\d{8}T\d{6}Z", dest.name)


# --------------------------------------------------------------------------- #
# _load_recall_csv — the CSV-shape parser with a history of bugs
# --------------------------------------------------------------------------- #

class TestLoadRecallCsv:
    def test_drops_first_non_numeric_column(self, tmp_path: Path):
        csv_path = tmp_path / "in.csv"
        csv_path.write_text(
            "Image,Mean,Std,Max,Mode\n"
            "a.png,11.35,8.79,104.0,1.0\n"
        )
        feature_df, cols = _load_recall_csv(str(csv_path))
        assert cols == ["Mean", "Std", "Max", "Mode"]
        assert list(feature_df.iloc[0]) == [11.35, 8.79, 104.0, 1.0]

    def test_drops_grade_column(self, tmp_path: Path):
        csv_path = tmp_path / "in.csv"
        csv_path.write_text(
            "Image,Mean,Std,Max,Mode,Grade\n"
            "a.png,11.35,8.79,104.0,1.0,4\n"
        )
        _, cols = _load_recall_csv(str(csv_path))
        assert "Grade" not in cols

    def test_drops_summary_rows(self, tmp_path: Path):
        """Analysis CSVs append Average/Grade/Backend rows. All of them plus
        the two real data rows go in; only the two data rows should come out."""
        csv_path = tmp_path / "analysis.csv"
        csv_path.write_text(
            "Image,Mean,Std,Max,Mode\n"
            "a-1-dif.png,11.37,8.70,103.0,1.0\n"
            "a-2-dif.png,11.74,9.16,132.0,1.0\n"
            "Average,11.55,8.93,117.5,1.0\n"
            "Grade,4.0,,,\n"
            "Backend,itanet_dll,,,\n"
        )
        feature_df, _ = _load_recall_csv(str(csv_path))
        # 2 image rows + Average row survive (Average is all-numeric); the
        # Grade and Backend rows have empty/non-numeric cells and drop out.
        assert len(feature_df) == 3

    def test_single_row_temp_csv(self, tmp_path: Path):
        """The shape the grading flow builds: one row with a non-numeric
        'Image' name. The Image column is dropped, leaving one all-numeric
        row that must survive."""
        csv_path = tmp_path / "one.csv"
        csv_path.write_text(
            "Image,Mean,Std,Max,Mode\n"
            "00000-100-1,11.35,8.79,104.0,1.0\n"
        )
        feature_df, _ = _load_recall_csv(str(csv_path))
        assert len(feature_df) == 1

    def test_feature_cols_override(self, tmp_path: Path):
        csv_path = tmp_path / "in.csv"
        csv_path.write_text(
            "Image,Mean,Std,Max,Mode\n"
            "a.png,11.35,8.79,104.0,1.0\n"
        )
        _, cols = _load_recall_csv(str(csv_path), feature_cols=["Mean", "Std"])
        assert cols == ["Mean", "Std"]

    def test_feature_cols_missing_raises(self, tmp_path: Path):
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("Image,Mean\na.png,11.35\n")
        with pytest.raises(ValueError, match="Missing columns"):
            _load_recall_csv(str(csv_path), feature_cols=["NopeSuchColumn"])


# --------------------------------------------------------------------------- #
# predict_from_csv — API surface (seed removed, kwargs preserved)
# --------------------------------------------------------------------------- #

class TestPredictFromCsvApi:
    def test_seed_kwarg_no_longer_accepted(self, tmp_path: Path):
        """Regression: the ``seed`` parameter was removed when the DLL lost
        its ``itanet_seed`` export. Callers passing seed= should fail loudly,
        not silently accept an unused arg."""
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("Mean\n1.0\n")
        with pytest.raises(TypeError, match="seed"):
            predict_from_csv(
                csv_path=str(csv_path),
                data_dir=str(tmp_path / "run"),
                dll_path=str(tmp_path / "fake.dll"),
                seed=42,  # type: ignore[call-arg]
            )

    def test_missing_dll_raises_before_write(self, tmp_path: Path):
        """The DLL guard in ITANetManager.__init__ should fire before any
        .dat file gets written."""
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("Mean\n1.0\n")
        with pytest.raises(FileNotFoundError, match="build.py"):
            predict_from_csv(
                csv_path=str(csv_path),
                data_dir=str(tmp_path / "run"),
                dll_path=str(tmp_path / "missing.dll"),
            )
        # Confirm the guard fired early — no RecallData.dat was written.
        assert not (tmp_path / "run" / ITANetFiles.RECALL_INPUT).exists()

    def test_empty_csv_after_filter_raises(self, tmp_path: Path):
        """CSV with only summary rows and no data rows should raise."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text(
            "Image,Mean\n"
            "Grade,,\n"  # gets filtered out
        )
        # dummy DLL file so the ITANetManager guard passes; the ValueError
        # about empty data must fire first
        (tmp_path / "fake.dll").write_bytes(b"")
        with pytest.raises(ValueError, match="No data rows"):
            predict_from_csv(
                csv_path=str(csv_path),
                data_dir=str(tmp_path / "run"),
                dll_path=str(tmp_path / "fake.dll"),
            )


# --------------------------------------------------------------------------- #
# Default paths — the run/data folder split
# --------------------------------------------------------------------------- #

class TestDefaultPaths:
    def test_defaults_are_absolute(self):
        # These get used as CLI argument resolvers — must be usable from any cwd.
        assert Path(_grade_run_dir("pilling")).is_absolute()
        assert Path(_grade_fls_dir("pilling")).is_absolute()
        assert Path(default_dll_path()).is_absolute()

    def test_run_and_fls_are_siblings(self):
        """The DLL hardcodes ``..\\data\\recall.fls`` relative to CWD, so each
        grade's run/fls dirs must be siblings — one level up shares a parent."""
        for grade in ("pilling", "matting", "fuzzing"):
            run = Path(_grade_run_dir(grade))
            fls = Path(_grade_fls_dir(grade))
            assert run.parent == fls.parent

    def test_default_dll_is_shared_common(self):
        """The DLL lives in models/itanet/common/, not inside any grade's run/."""
        dll = Path(default_dll_path())
        assert dll.parent.name == "common"
        assert dll.parent.parent.name == "itanet"


# --------------------------------------------------------------------------- #
# change_directory — cwd restored on exit even if body raises
# --------------------------------------------------------------------------- #

class TestChangeDirectory:
    def test_restores_cwd(self, tmp_path: Path):
        import os
        prev = os.getcwd()
        with change_directory(str(tmp_path)):
            assert Path(os.getcwd()).resolve() == tmp_path.resolve()
        assert os.getcwd() == prev

    def test_restores_cwd_on_exception(self, tmp_path: Path):
        import os
        prev = os.getcwd()
        with pytest.raises(RuntimeError):
            with change_directory(str(tmp_path)):
                raise RuntimeError("body failed")
        assert os.getcwd() == prev
