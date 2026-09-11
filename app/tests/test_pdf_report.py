"""PDF report tests + visual-check entrypoint.

Two entry points share the same _TRIAL_SCENARIOS fixture data:

    pytest app/tests/test_pdf_report.py        # automated: asserts on PDF text
    python app/tests/test_pdf_report.py        # visual:   writes PDFs to reports/visual_tests/

The visual path is for eyeballing layout, table shape, and reference-image
embedding. PDFs land in reports/visual_tests/ (gitignored).
"""
import os
import sys
import tempfile
import shutil
import time
import importlib
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
                    # bare @decorator: args=(func,), kwargs={}
                    if args and callable(args[0]) and not kwargs:
                        return args[0]
                    # @decorator(...) form: returns identity decorator
                    return lambda f: f
                return deco
        pytest = _NoopDecorator()        # pytest.fixture, pytest.fixture(...)
        pytest.mark = _NoopDecorator()    # type: ignore[attr-defined]

# Add project root to sys.path so `app.*` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Only import pdf_report here (does not touch Streamlit)
from app.reporting.pdf_report import generate_pilling_report

# --- helpers -----------------------------------------------------------------

def _read_pdf_text(path: str) -> str:
    """
    Read text from a PDF using PyPDF2 (if available). If PyPDF2 isn't installed,
    the test that needs text will be skipped.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        pytest.skip(f"PyPDF2 not installed to read PDF text: {e}")
    reader = PdfReader(path)
    text_chunks = []
    for page in reader.pages:
        try:
            text_chunks.append(page.extract_text() or "")
        except Exception:
            # Some readers may fail to extract; keep best-effort
            continue
    return "\n".join(text_chunks)

# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_number() -> str:
    # Matches a fixture under output/grading_results/.
    return "00000"

@pytest.fixture
def clean_reports_dir(tmp_path):
    """
    Redirect the 'reports' directory used by export_results to a temp folder by
    chdir into project root and creating a 'reports' folder there.
    After test, cleanup happens automatically with tmp_path.
    """
    # Move into project root (this test file is in .../Automated_Pilling_Grade_Classifier/app/tests/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cwd_before = os.getcwd()
    os.chdir(project_root)

    # Backup any existing 'reports' and use a temporary one for the test
    backup_dir = None
    if os.path.isdir("reports"):
        backup_dir = tempfile.mkdtemp(prefix="reports_backup_")
        shutil.move("reports", backup_dir)

    os.makedirs("reports", exist_ok=True)
    yield

    # Cleanup: remove the temp reports created during test
    try:
        shutil.rmtree("reports", ignore_errors=True)
    except Exception:
        pass

    # Restore previous reports if any
    if backup_dir:
        shutil.move(os.path.join(backup_dir, "reports"), project_root)
        shutil.rmtree(backup_dir, ignore_errors=True)

    os.chdir(cwd_before)

# --- tests -------------------------------------------------------------------

@pytest.mark.pdf
def test_scan_collects_dynamic_stages(sample_number):
    """
    Ensures scanning finds all stage CSVs and returns sorted rub counts.
    """
    # Lazy import after conftest stubbed streamlit
    utils = importlib.import_module("app.helpers.utils")
    # Given attachments, expect these six stages
    expected = {100, 200, 300, 600, 700}
    results = utils._scan_stage_results_for_sample(sample_number)
    assert set(results.keys()) == expected, f"Found rub levels: {sorted(results.keys())}"

    # Also ensure each entry has a non-empty grade string
    assert all(str(v).strip() for v in results.values())

@pytest.mark.pdf
def test_export_results_creates_single_pdf_and_contains_expected_text(sample_number, clean_reports_dir):
    """
    End-to-end: export_results should produce one combined PDF per sample, with:
    - Title and ISO statement
    - Details (sample, abradant, operator, load)
    - Rows for all discovered rub levels (dynamic)
    """
    utils = importlib.import_module("app.helpers.utils")
    operator = "UnitTest"
    load_weight = "150"
    # stage_number is not used for combining; any string is fine
    ok = utils.export_results(sample_number, stage_number="100", load_weight=load_weight, operator_name=operator)
    assert ok is True

    # The report filename pattern created by export_results
    pdf_path = os.path.join("reports", f"{sample_number}-{operator}-report.pdf")
    assert os.path.isfile(pdf_path), f"Expected PDF at {pdf_path}"

    # Give the filesystem a brief moment on Windows
    time.sleep(0.2)

    text = _read_pdf_text(pdf_path)

    # Header and statement
    assert "Pilling Test Report" in text
    assert "ISO-12945-2 standards" in text

    # Details
    assert f"Sample Number: {sample_number}" in text
    assert "Abradant used: Similar Fabric" in text
    assert f"Operator Name: {operator}" in text
    assert f"Loading Weight (gms): {load_weight}" in text

    # Dynamic rows for all available stages
    for rub in (100, 200, 300, 600):
        assert f"{rub} rev." in text

@pytest.mark.pdf
def test_generate_pilling_report_mimics_combined_report(sample_number, tmp_path):
    """
    Focus on main PDF logic only: generate a report for a predefined set of rub levels
    and verify expected text without importing Streamlit-dependent utilities.
    """
    operator = "UnitTest"
    load_weight = "150"
    rub_levels = [100, 200, 300, 320, 550, 600]
    # Provide mirrored/symmetric panel grades as strings, as used by the app
    results_by_rubs = {
        100: ["3", "3", "3", "3"],
        200: ["3", "3", "3", "3"],
        300: ["3", "3", "3", "3"],
        320: ["3", "3", "3", "3"],
        550: ["3", "3", "3", "3"],
        600: ["3", "3", "3", "3"],
    }
    out_file = tmp_path / f"{sample_number}-{operator}-report.pdf"

    ok = generate_pilling_report(
        sample_number=sample_number,
        stage_number=None,
        load_weight_g=load_weight,
        operator_name=operator,
        results_by_rubs=results_by_rubs,
        output_path=str(out_file),
        rub_levels=rub_levels,
    )
    assert ok is True
    assert out_file.exists()

    # Give the filesystem a brief moment on Windows
    time.sleep(0.2)

    text = _read_pdf_text(str(out_file))

    # Header and statement
    assert "Pilling Test Report" in text
    assert "ISO-12945-2 standards" in text

    # Details
    assert f"Sample Number: {sample_number}" in text
    assert "Abradant used: Similar Fabric" in text
    assert f"Operator Name: {operator}" in text
    assert f"Loading Weight (gms): {load_weight}" in text

    # Rows for provided stages
    for rub in rub_levels:
        assert f"{rub} rev." in text

@pytest.mark.pdf
def test_generate_pilling_report_accepts_custom_levels(tmp_path):
    """
    Unit test for the PDF generator with custom rub levels and mirrored results.
    """
    out_file = tmp_path / "custom.pdf"
    results_by_rubs = {
        75: ["2.5", "2.5", "2.5", "2.5"],
        150: ["3", "3", "3", "3"],
    }
    ok = generate_pilling_report(
        sample_number="XYZ",
        stage_number=None,
        load_weight_g="123",
        operator_name="Tester",
        results_by_rubs=results_by_rubs,
        output_path=str(out_file),
        rub_levels=[75, 150],
    )
    assert ok is True
    assert out_file.exists()

    text = _read_pdf_text(str(out_file))
    assert "Pilling Test Report" in text
    assert "75 rev." in text and "150 rev." in text

_TRIAL_SCENARIOS = {
    # one grade per rub + NA placeholders + average
    "single_trial": {
        "sample_number": "001",
        "operator": "SingleTrialTest",
        "rub_levels": [100, 200, 300],
        "results_by_rubs": {
            100: ["3", "NA", "NA", "3"],
            200: ["3", "NA", "NA", "3"],
            300: ["2.5", "NA", "NA", "2.5"],
        },
    },
    "two_trials": {
        "sample_number": "002",
        "operator": "TwoTrialTest",
        "rub_levels": [100, 200, 300, 320],
        "results_by_rubs": {
            100: ["3", "3", "NA", "3"],
            200: ["3", "2.5", "NA", "2.75"],
            300: ["2.5", "2.5", "NA", "2.5"],
            320: ["2", "2.5", "NA", "2.25"],
        },
    },
    "three_trials": {
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
    },
    # edge case: different number of trials per rub level
    "mixed_trials": {
        "sample_number": "004",
        "operator": "MixedTrialsTest",
        "rub_levels": [100, 200, 300],
        "results_by_rubs": {
            100: ["4", "NA", "NA", "4"],
            200: ["3.5", "3", "NA", "3.25"],
            300: ["3", "2.5", "3", "2.83"],
        },
    },
}


@pytest.mark.pdf
@pytest.mark.parametrize("scenario_id", list(_TRIAL_SCENARIOS.keys()))
def test_generate_pilling_report_trial_scenarios(tmp_path, scenario_id):
    """PDF generation across single/two/three/mixed-trial result shapes."""
    s = _TRIAL_SCENARIOS[scenario_id]
    load_weight = "150"
    out_file = tmp_path / f"{s['sample_number']}-{s['operator']}-{scenario_id}.pdf"

    ok = generate_pilling_report(
        sample_number=s["sample_number"],
        stage_number=None,
        load_weight_g=load_weight,
        operator_name=s["operator"],
        results_by_rubs=s["results_by_rubs"],
        output_path=str(out_file),
        rub_levels=s["rub_levels"],
    )
    assert ok is True
    assert out_file.exists()

    time.sleep(0.2)  # Windows filesystem settle
    text = _read_pdf_text(str(out_file))

    assert "Pilling Test Report" in text
    assert f"Sample Number: {s['sample_number']}" in text
    assert f"Operator Name: {s['operator']}" in text
    for rub in s["rub_levels"]:
        assert f"{rub} rev." in text


# --- visual check entrypoint -------------------------------------------------
# Running the file directly (not via pytest) writes one PDF per _TRIAL_SCENARIOS
# entry so a developer can open them and verify the layout looks right. None of
# the pytest test functions run in this mode — only the __main__ block does.

if __name__ == "__main__":
    output_dir = Path("reports") / "visual_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for scenario_id, s in _TRIAL_SCENARIOS.items():
        out = output_dir / f"{s['sample_number']}-{s['operator']}-{scenario_id}.pdf"
        ok = generate_pilling_report(
            sample_number=s["sample_number"],
            stage_number=None,
            load_weight_g="150",
            operator_name=s["operator"],
            results_by_rubs=s["results_by_rubs"],
            output_path=str(out),
            rub_levels=s["rub_levels"],
        )
        if ok:
            print(f"  [ok]   {out}")
            successes += 1
        else:
            print(f"  [fail] {out}")
    print(f"\nGenerated {successes}/{len(_TRIAL_SCENARIOS)} PDFs.")
    sys.exit(0 if successes == len(_TRIAL_SCENARIOS) else 1)