"""PDF report tests + visual-check entrypoint.

Two entry points share the same _TRIAL_SCENARIOS fixture data:

    pytest app/tests/test_pdf_report.py        # automated: asserts on PDF text
    python app/tests/test_pdf_report.py        # visual:   writes PDFs to reports/visual_tests/

The visual path is for eyeballing layout, table shape, and reference-image
embedding. PDFs land in reports/visual_tests/ (gitignored).
"""
import csv
import os
import sys
import time
from pathlib import Path

# pytest is a dev-only dependency. The visual-check __main__ block doesn't need
# it, so when running this file directly without pytest installed, fall back to
# a tiny no-op shim that makes the decorators below transparent.
if __name__ != "__main__":
    import pytest
else:
    try:
        import pytest  # type: ignore
    except ImportError:
        class _NoopDecorator:
            def __getattr__(self, _name):
                def deco(*args, **kwargs):
                    if args and callable(args[0]) and not kwargs:
                        return args[0]
                    return lambda f: f
                return deco
        pytest = _NoopDecorator()
        pytest.mark = _NoopDecorator()  # type: ignore[attr-defined]

# Add project root to sys.path so `app.*` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.reporting.pdf_report import generate_pilling_report
from app.reporting.results import (
    _build_by_rubs,
    _scan_stage_results_for_sample,
    export_results,
)


# --- helpers -----------------------------------------------------------------

def _read_pdf_text(path: str) -> str:
    """Read text from a PDF. Skip test if pypdf is unavailable."""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        pytest.skip(f"pypdf not installed to read PDF text: {e}")
    reader = PdfReader(path)
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(chunks)


def _write_analysis_csv(dirpath: Path, sample: str, stage: int, trial: int,
                        grade_name: str, predicted_grade: str) -> None:
    """Write an analysis CSV mirroring histogram_analysis.py output format.

    Filename: {sample}-{stage}-{trial}-{grade}-analysis.csv
    Payload:  headers, 8 image rows, Average row, Grade row, Backend row.
    """
    fname = f"{sample}-{stage}-{trial}-{grade_name}-analysis.csv"
    path = dirpath / fname
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image", "Mean", "Std", "Max", "Mode"])
        for i in range(1, 9):
            w.writerow([f"{sample}-{stage}-{trial}-{i}-dif.png", 8.0, 0.0, 8.0, 8.0])
        w.writerow(["Average", 8.0, 0.0, 8.0, 8.0])
        w.writerow(["Grade", predicted_grade, "", "", ""])
        w.writerow(["Backend", "itanet_dll", "", "", ""])


@pytest.fixture
def grading_results_dir(tmp_path, monkeypatch):
    """Redirect cfg.GRADING_RESULTS_DIR + cfg.REPORTS_DIR to tmp_path so tests
    are self-contained and don't touch the real output/reports trees."""
    from app.settings import config as cfg
    results_dir = tmp_path / "grading_results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir()
    reports_dir.mkdir()
    monkeypatch.setattr(cfg, "GRADING_RESULTS_DIR", str(results_dir))
    monkeypatch.setattr(cfg, "REPORTS_DIR", str(reports_dir))

    # export_results reads reports_dir at call time; chdir so relative paths
    # (used by generate_pilling_report for the logo image) still resolve to
    # the project root.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cwd_before = os.getcwd()
    os.chdir(project_root)
    try:
        yield {"results": results_dir, "reports": reports_dir}
    finally:
        os.chdir(cwd_before)


# --- scan + build unit tests -------------------------------------------------

class TestScanStageResultsForSample:
    def test_returns_empty_when_dir_missing(self, tmp_path, monkeypatch):
        from app.settings import config as cfg
        monkeypatch.setattr(cfg, "GRADING_RESULTS_DIR", str(tmp_path / "missing"))
        assert _scan_stage_results_for_sample("001", "pilling") == {}

    def test_filters_by_grade(self, grading_results_dir):
        d = grading_results_dir["results"]
        _write_analysis_csv(d, "007", 100, 1, "pilling", "3")
        _write_analysis_csv(d, "007", 100, 1, "matting", "2.5")
        _write_analysis_csv(d, "007", 100, 1, "fuzzing", "4")

        assert _scan_stage_results_for_sample("007", "pilling") == {100: ["3"]}
        assert _scan_stage_results_for_sample("007", "matting") == {100: ["2.5"]}
        assert _scan_stage_results_for_sample("007", "fuzzing") == {100: ["4"]}

    def test_collects_multiple_stages(self, grading_results_dir):
        d = grading_results_dir["results"]
        for stage in (100, 200, 500):
            _write_analysis_csv(d, "007", stage, 1, "pilling", "3")

        result = _scan_stage_results_for_sample("007", "pilling")
        assert sorted(result.keys()) == [100, 200, 500]

    def test_collects_multiple_trials_per_stage(self, grading_results_dir):
        d = grading_results_dir["results"]
        for trial, val in enumerate(["3", "2.5", "3.5"], start=1):
            _write_analysis_csv(d, "007", 100, trial, "pilling", val)

        result = _scan_stage_results_for_sample("007", "pilling")
        assert result == {100: ["2.5", "3", "3.5"]}

    def test_ignores_other_samples(self, grading_results_dir):
        d = grading_results_dir["results"]
        _write_analysis_csv(d, "007", 100, 1, "pilling", "3")
        _write_analysis_csv(d, "999", 100, 1, "pilling", "5")

        assert _scan_stage_results_for_sample("007", "pilling") == {100: ["3"]}

    def test_skips_malformed_csv(self, grading_results_dir):
        d = grading_results_dir["results"]
        # A file whose contents lack a "Grade" row should be quietly skipped.
        bad = d / "007-100-1-pilling-analysis.csv"
        bad.write_text("Image,Mean\n007-100-1-1-dif.png,8.0\n", encoding="utf-8")
        assert _scan_stage_results_for_sample("007", "pilling") == {}


class TestBuildByRubs:
    def test_single_trial_pads_with_na(self):
        assert _build_by_rubs({100: ["3"]}) == {100: ["3", "NA", "NA", "3"]}

    def test_two_trials_averaged(self):
        assert _build_by_rubs({100: ["3", "4"]}) == {100: ["3", "4", "NA", "3.50"]}

    def test_three_trials_averaged(self):
        assert _build_by_rubs({100: ["3", "3.5", "4"]}) == {100: ["3", "3.5", "4", "3.50"]}

    def test_more_than_three_trials_uses_first_three(self):
        # Column layout is fixed at 3 result columns; extras must not crash.
        result = _build_by_rubs({100: ["3", "3", "3", "5"]})
        assert result == {100: ["3", "3", "3", "3.00"]}


# --- export_results end-to-end (per-grade) -----------------------------------

class TestExportResultsPerGrade:
    def _seed_all_grades(self, d: Path, sample: str, stages=(100, 200, 300)):
        for stage in stages:
            _write_analysis_csv(d, sample, stage, 1, "pilling", "3")
            _write_analysis_csv(d, sample, stage, 1, "matting", "2.5")
            _write_analysis_csv(d, sample, stage, 1, "fuzzing", "4")

    def test_returns_false_when_no_csvs(self, grading_results_dir):
        assert export_results("999", "100", "150", "Op") is False

    def test_produces_pdf_when_only_pilling_present(self, grading_results_dir):
        d = grading_results_dir["results"]
        _write_analysis_csv(d, "011", 100, 1, "pilling", "3")

        assert export_results("011", "100", "150", "Op") is True
        pdf = grading_results_dir["reports"] / "011-Op-report.pdf"
        assert pdf.exists()

    def test_produces_pdf_with_all_three_grades(self, grading_results_dir):
        d = grading_results_dir["results"]
        self._seed_all_grades(d, "012")

        assert export_results("012", "100", "150", "Op") is True
        pdf = grading_results_dir["reports"] / "012-Op-report.pdf"
        assert pdf.exists()

        text = _read_pdf_text(str(pdf))
        assert "Pilling" in text
        assert "Matting" in text
        assert "Fuzzing" in text
        for rub in (100, 200, 300):
            assert f"{rub} rev." in text

    def test_matting_only_still_produces_pdf(self, grading_results_dir):
        """If only matting has CSVs (unusual but possible), the report should
        still list its rub levels."""
        d = grading_results_dir["results"]
        _write_analysis_csv(d, "013", 100, 1, "matting", "2.5")
        _write_analysis_csv(d, "013", 200, 1, "matting", "3")

        assert export_results("013", "100", "150", "Op") is True
        pdf = grading_results_dir["reports"] / "013-Op-report.pdf"
        assert pdf.exists()
        text = _read_pdf_text(str(pdf))
        assert "100 rev." in text
        assert "200 rev." in text

    def test_operator_name_sanitized_in_filename(self, grading_results_dir):
        d = grading_results_dir["results"]
        _write_analysis_csv(d, "014", 100, 1, "pilling", "3")

        assert export_results("014", "100", "150", "Op Test!@#") is True
        # Non-alnum stripped except '-' and '_'
        assert (grading_results_dir["reports"] / "014-OpTest-report.pdf").exists()


# --- generate_pilling_report direct unit tests ------------------------------

@pytest.mark.parametrize("scenario_id,scenario", [
    ("single_trial", {
        "sample_number": "001",
        "operator": "SingleTrialTest",
        "rub_levels": [100, 200, 300],
        "results_by_rubs": {
            100: ["3", "NA", "NA", "3"],
            200: ["3", "NA", "NA", "3"],
            300: ["2.5", "NA", "NA", "2.5"],
        },
    }),
    ("two_trials", {
        "sample_number": "002",
        "operator": "TwoTrialTest",
        "rub_levels": [100, 200, 300, 320],
        "results_by_rubs": {
            100: ["3", "3", "NA", "3"],
            200: ["3", "2.5", "NA", "2.75"],
            300: ["2.5", "2.5", "NA", "2.5"],
            320: ["2", "2.5", "NA", "2.25"],
        },
    }),
    ("three_trials", {
        "sample_number": "003",
        "operator": "ThreeTrialTest",
        "rub_levels": [100, 200, 300, 550, 600],
        "results_by_rubs": {
            100: ["4", "3.5", "4", "3.83"],
            200: ["3.5", "3", "3.5", "3.33"],
            300: ["3", "3", "2.5", "2.83"],
            550: ["2.5", "2", "2.5", "2.33"],
            600: ["2", "2", "2.5", "2.17"],
        },
    }),
])
def test_generate_pilling_report_trial_scenarios(tmp_path, scenario_id, scenario):
    """PDF generation across single/two/three-trial result shapes."""
    out_file = tmp_path / f"{scenario['sample_number']}-{scenario['operator']}-{scenario_id}.pdf"

    ok = generate_pilling_report(
        sample_number=scenario["sample_number"],
        stage_number=None,
        load_weight_g="150",
        operator_name=scenario["operator"],
        results_by_rubs=scenario["results_by_rubs"],
        output_path=str(out_file),
        rub_levels=scenario["rub_levels"],
    )
    assert ok is True
    assert out_file.exists()

    time.sleep(0.2)  # Windows filesystem settle
    text = _read_pdf_text(str(out_file))

    assert "Pilling Test Report" in text
    assert f"Sample Number: {scenario['sample_number']}" in text
    assert f"Operator Name: {scenario['operator']}" in text
    for rub in scenario["rub_levels"]:
        assert f"{rub} rev." in text


def test_generate_pilling_report_with_all_three_grade_columns(tmp_path):
    """When matting_by_rubs and fuzzing_by_rubs are provided, the PDF should
    include all three grade section labels."""
    results_by_rubs = {100: ["3", "3", "3", "3"], 200: ["3", "3", "3", "3"]}
    matting_by_rubs = {100: ["2.5", "NA", "NA", "2.5"], 200: ["3", "NA", "NA", "3"]}
    fuzzing_by_rubs = {100: ["4", "NA", "NA", "4"], 200: ["3.5", "NA", "NA", "3.5"]}
    out_file = tmp_path / "all-grades.pdf"

    ok = generate_pilling_report(
        sample_number="ABC",
        stage_number=None,
        load_weight_g="150",
        operator_name="Op",
        results_by_rubs=results_by_rubs,
        matting_by_rubs=matting_by_rubs,
        fuzzing_by_rubs=fuzzing_by_rubs,
        output_path=str(out_file),
        rub_levels=[100, 200],
    )
    assert ok is True
    assert out_file.exists()

    text = _read_pdf_text(str(out_file))
    assert "Pilling" in text
    assert "Matting" in text
    assert "Fuzzing" in text


# --- visual check entrypoint -------------------------------------------------
# Running the file directly writes one PDF per scenario to reports/visual_tests/
# for eyeballing layout.

_VISUAL_SCENARIOS = {
    "single_trial": {
        "sample_number": "001", "operator": "SingleTrialTest",
        "rub_levels": [100, 200, 300],
        "results_by_rubs": {
            100: ["3", "NA", "NA", "3"],
            200: ["3", "NA", "NA", "3"],
            300: ["2.5", "NA", "NA", "2.5"],
        },
    },
    "three_trials_all_grades": {
        "sample_number": "003", "operator": "AllGrades",
        "rub_levels": [100, 200, 300],
        "results_by_rubs": {
            100: ["4", "3.5", "4", "3.83"],
            200: ["3.5", "3", "3.5", "3.33"],
            300: ["3", "3", "2.5", "2.83"],
        },
        "matting_by_rubs": {
            100: ["2.5", "2.5", "3", "2.67"],
            200: ["3", "2.5", "3", "2.83"],
            300: ["3.5", "3", "3", "3.17"],
        },
        "fuzzing_by_rubs": {
            100: ["4", "4.5", "4", "4.17"],
            200: ["4", "3.5", "4", "3.83"],
            300: ["3.5", "3", "3.5", "3.33"],
        },
    },
}


if __name__ == "__main__":
    output_dir = Path("reports") / "visual_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for scenario_id, s in _VISUAL_SCENARIOS.items():
        out = output_dir / f"{s['sample_number']}-{s['operator']}-{scenario_id}.pdf"
        ok = generate_pilling_report(
            sample_number=s["sample_number"],
            stage_number=None,
            load_weight_g="150",
            operator_name=s["operator"],
            results_by_rubs=s["results_by_rubs"],
            matting_by_rubs=s.get("matting_by_rubs"),
            fuzzing_by_rubs=s.get("fuzzing_by_rubs"),
            output_path=str(out),
            rub_levels=s["rub_levels"],
        )
        if ok:
            print(f"  [ok]   {out}")
            successes += 1
        else:
            print(f"  [fail] {out}")
    print(f"\nGenerated {successes}/{len(_VISUAL_SCENARIOS)} PDFs.")
    sys.exit(0 if successes == len(_VISUAL_SCENARIOS) else 1)
