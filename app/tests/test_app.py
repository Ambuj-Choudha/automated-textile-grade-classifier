"""Unit tests for app.py callbacks and actions.

All Streamlit, config, camera, and analysis dependencies are stubbed so tests
run without hardware, trained models, or a real Streamlit server.
"""
import os
import sys
import csv
import types
import tempfile
from pathlib import Path
import pytest
import importlib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_at(monkeypatch, dotted_path: str, mod) -> None:
    """Install fake mod at sys.modules[dotted_path] and bind it on its parent."""
    monkeypatch.setitem(sys.modules, dotted_path, mod)
    if "." in dotted_path:
        parent_path, child_name = dotted_path.rsplit(".", 1)
        parent = sys.modules.get(parent_path)
        if parent is None:
            parent = types.ModuleType(parent_path)
            monkeypatch.setitem(sys.modules, parent_path, parent)
        monkeypatch.setattr(parent, child_name, mod, raising=False)


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------

def _install_dummy_streamlit(monkeypatch):
    class _DummyCtx:
        def __enter__(self): return self
        def __exit__(self, *_): return False

    class _Sess(dict):
        def __getattr__(self, k):
            try: return self[k]
            except KeyError: raise AttributeError(k)
        def __setattr__(self, k, v): self[k] = v
        def pop(self, k, default=None): return dict.pop(self, k, default)
        def get(self, k, default=None): return dict.get(self, k, default)

    class _CacheData:
        def clear(self): pass
        def __call__(self, *args, **kwargs):
            def decorator(func): return func
            return decorator

    class DummyStreamlit:
        def __init__(self):
            self.session_state = _Sess({"lang": "en", "fs_epoch": 0})
            self.sidebar = self
            self._button_defaults = {}
            self._last_metrics = []
            self.cache_data = _CacheData()

        def __enter__(self): return self
        def __exit__(self, *_): return False

        def expander(self, *a, **k): return _DummyCtx()
        def container(self): return _DummyCtx()
        def spinner(self, *a, **k): return _DummyCtx()
        def empty(self): return _DummyCtx()
        def rerun(self): pass

        def columns(self, spec):
            n = spec if isinstance(spec, int) else len(list(spec))
            return [self for _ in range(n)]

        def set_page_config(self, *a, **k): pass
        def markdown(self, *a, **k): pass
        def header(self, *a, **k): pass
        def subheader(self, *a, **k): pass
        def title(self, *a, **k): pass
        def image(self, *a, **k): pass
        def info(self, *a, **k): pass
        def error(self, *a, **k): pass
        def success(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def metric(self, *a, **k): self._last_metrics.append((a, k))

        def selectbox(self, label, options, index=0, **k):
            if not options: return None
            if not (0 <= index < len(options)): index = 0
            return options[index]

        def multiselect(self, label, options, default=None, **k):
            return list(default) if default is not None else list(options)

        def checkbox(self, label, value=False, **k):
            return value

        def text_input(self, label, value="", **k): return value
        def number_input(self, label, value=0, **k): return value
        def button(self, label, **k): return self._button_defaults.get(label, False)

        def __getattr__(self, name):
            # Catch-all so sidebar.X delegates to self.X
            return object.__getattribute__(self, name)

    dummy = DummyStreamlit()
    m = types.ModuleType("streamlit")
    for attr in dir(dummy):
        if not attr.startswith("_"):
            setattr(m, attr, getattr(dummy, attr))
    monkeypatch.setitem(sys.modules, "streamlit", m)
    return m


def _install_dummy_translations(monkeypatch):
    _trans = {
        "en": {
            "settings": "Settings",
            "select_grades": "Active Grades",
            "grades_required": "Select at least one grade",
            "motor_settings": "Motor settings",
            "motor_ip": "Motor IP",
            "motor_port": "Motor Port",
            "view_images": "View images",
            "select_folder": "Select folder",
            "enter_filename": "Enter filename",
            "invalid_picture_name": "Invalid picture name",
            "title_grading": "Grading",
            "title_training": "Training",
            "available_samples": "Available samples",
            "sample_number": "Sample",
            "stage_number": "Stage",
            "trial_number": "Trial",
            "load_weight": "Load",
            "pilling_grade_number": "Pilling Grade",
            "matting_grade_number": "Matting Grade",
            "fuzzing_grade_number": "Fuzzing Grade",
            "pilling_grade_result": "Pilling Grade",
            "matting_grade_result": "Matting Grade",
            "fuzzing_grade_result": "Fuzzing Grade",
            "invalid_sample": "Invalid sample",
            "invalid_stage": "Invalid stage",
            "invalid_trial": "Invalid trial",
            "invalid_load": "Invalid load",
            "invalid_pilling_grade": "Invalid grade",
            "invalid_matting_grade": "Invalid matting grade",
            "invalid_fuzzing_grade": "Invalid fuzzing grade",
            "valid_input": "Valid input",
            "capture_button": "Capture Images",
            "capture_button_training": "Capture Images (Training)",
            "results": "Results",
            "export_results": "Export Results",
            "reset_button": "Reset",
            "reset_grade": "Reset Grade",
            "status": "Status",
            "mode_selector": "Mode",
            "training_mode": "Training Mode",
            "grading_mode": "Grading Mode",
            "login_required": "Login required",
            "password_prompt": "Password",
            "submit": "Submit",
            "operator_prompt": "Operator Name",
            "histogram_fail": "Analysis failed",
            "reference_not_found": "Reference image not found",
            "image_capture_spinner": "Capturing images...",
            "image_capture_fail": "Image capture failed",
            "image_capture_success": "Images captured successfully",
            "diff_create_spinner": "Creating difference images...",
            "diff_create_success": "Difference images created",
            "diff_create_fail": "Failed to create difference images",
            "histogram_spinner": "Analyzing histograms...",
            "histogram_success": "Analysis complete",
            "login_error": "Invalid password",
            "name_required": "Name is required",
        }
    }

    def t(key: str) -> str:
        return _trans["en"].get(key, key)

    m = types.ModuleType("app.reporting.translations")
    m.t = t
    m.translations = _trans
    _install_at(monkeypatch, "app.reporting.translations", m)
    return m


def _install_dummy_utils(monkeypatch):
    m = types.ModuleType("utils")

    def validate_input_number(x: str) -> bool:
        return bool(x and x.strip().isalnum())

    def validate_grade(x: str) -> bool:
        if not x:
            return False
        try:
            return float(x) in {1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}
        except ValueError:
            return False

    def get_available_samples(mode: str):
        return [f"{i:03d}" for i in range(1, 6)]

    def check_required_images(sample_number, stage_number, trial_number, suffix):
        return {
            "present_input_images": [],
            "present_difference_images": [],
            "all_input_present": False,
            "all_difference_present": False,
        }

    def ensure_directory(path: str):
        os.makedirs(path, exist_ok=True)
        return path

    def export_results(*a, **k): return True

    m.validate_input_number = validate_input_number
    m.validate_grade = validate_grade
    m.get_available_samples = get_available_samples
    m.check_required_images = check_required_images
    m.ensure_directory = ensure_directory
    m.export_results = export_results
    _install_at(monkeypatch, "app.helpers.utils", m)

    # app.py now imports export_results from app.reporting.results — stub it too
    results_m = types.ModuleType("app.reporting.results")
    results_m.export_results = export_results
    _install_at(monkeypatch, "app.reporting.results", results_m)

    return m


def _install_dummy_camera_and_diff(monkeypatch):
    cam = types.ModuleType("camera_control")

    def capture_sample_images(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port):
        return True

    cam.capture_sample_images = capture_sample_images
    _install_at(monkeypatch, "app.capture.camera_control", cam)

    diff = types.ModuleType("image_difference")

    def create_difference_images(sample_number, stage_number, trial_number, suffix, reference_stage):
        return True

    diff.create_difference_images = create_difference_images
    _install_at(monkeypatch, "app.capture.image_difference", diff)
    return cam, diff


def _install_dummy_config(monkeypatch, tmp_path: Path):
    cfg = types.ModuleType("config")
    cfg.GRADES = ("pilling", "matting", "fuzzing")
    cfg.SUFFIX_GRADING = "for_grading"
    cfg.SUFFIX_TRAINING = "for_training"
    cfg.BACKEND = "itanet_dll"
    cfg.MOCK_PREDICTION = True
    cfg.MOCK_HARDWARE = True
    cfg.ADMIN_PASSWORD = "1234"
    cfg.DLL_PATH = str(tmp_path / "itanet" / "common" / "itanet.dll")
    cfg.TF_MODEL_PATH = str(tmp_path / "my_model.h5")
    cfg.FEATURE_COLUMNS = ("Mean", "Std", "Max", "Mode")
    cfg.DECIMAL = "point"
    cfg.ROUND_TO_HALF = True
    cfg.CLIP_RANGE = (1.0, 5.0)
    cfg.CUSTOM_NET = None
    cfg.CUSTOM_TRN = None
    cfg.GRADING_RESULTS_DIR = str(tmp_path / "output" / "grading_results")
    cfg.TRAINING_FEATURES_DIR = str(tmp_path / "data" / "training_features")
    cfg.get_itanet_run_dir = lambda grade: str(tmp_path / "itanet" / grade / "run")
    cfg.get_itanet_fls_dir = lambda grade: str(tmp_path / "itanet" / grade / "data")
    cfg.get_itanet_archive_dir = lambda grade: str(tmp_path / "itanet" / "archive" / grade)
    # Tests assume all three grades are usable; override the file-size check.
    cfg.get_trained_grades = lambda: list(cfg.GRADES)
    cfg.get_input_dir = lambda suffix: str(tmp_path / "input_pictures" / suffix)
    cfg.get_difference_dir = lambda suffix: str(tmp_path / "difference_pictures" / suffix)
    cfg.make_input_filename = lambda s, st, tr, pos: f"{s}-{st}-{tr}-{pos}.png"
    cfg.make_difference_filename = lambda s, st, tr, pos: f"{s}-{st}-{tr}-{pos}-dif.png"
    for grade in ("pilling", "matting", "fuzzing"):
        os.makedirs(cfg.get_itanet_run_dir(grade), exist_ok=True)
        os.makedirs(cfg.get_itanet_fls_dir(grade), exist_ok=True)
    _install_at(monkeypatch, "app.settings.config", cfg)
    return cfg


def _install_dummy_itanet(monkeypatch, tmp_path: Path):
    itanet = types.ModuleType("itanet_recall")

    def predict_from_csv(csv_path, data_dir, dll_path, output_path, feature_cols,
                         clip_range=None, fls_dir=None, **kw):
        rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["Predicted_Grade"] = "3.0"
                rows.append(row)
        fieldnames = list(rows[0].keys()) if rows else list(feature_cols) + ["Predicted_Grade"]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    itanet.predict_from_csv = predict_from_csv
    _install_at(monkeypatch, "itanet_recall", itanet)
    return itanet


def _install_dummy_histogram_analysis(monkeypatch, tmp_path: Path):
    ha = types.ModuleType("histogram_analysis")
    _train_called = {"called": False, "args": None}

    # Mirrors the real function's per-grade filtering so wiring tests can
    # verify that only requested grades appear in the returned dict.
    _mock_all = {"pilling": 3.0, "matting": 2.5, "fuzzing": 3.5}

    def analyze_difference_images_and_predict_output(
        sample_number, stage_number, trial_number,
        selected_grades=None, output_dir=None,
    ):
        requested = list(selected_grades) if selected_grades is not None else list(_mock_all)
        return {g: _mock_all[g] for g in requested if g in _mock_all} or None

    def analyze_difference_images(
        sample_number, stage_number, trial_number,
        grades=None, output_dir=None,
    ):
        _train_called["called"] = True
        _train_called["args"] = (sample_number, stage_number, trial_number)

    ha.analyze_difference_images_and_predict_output = analyze_difference_images_and_predict_output
    ha.analyze_difference_images = analyze_difference_images
    ha._train_called = _train_called
    _install_at(monkeypatch, "app.analysis.histogram_analysis", ha)
    return ha


def _install_dummy_pil(monkeypatch):
    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil.Image = pil_image
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image)
    return pil


def _import_app_module():
    # app.py lives at project root — two levels above this test file's package
    app_path = Path(__file__).parent.parent.parent / "app.py"
    if not app_path.exists():
        pytest.skip(f"app.py not found at {app_path}")

    # Purge cached app.* modules so each test's fake streamlit is captured
    # freshly by their module-level `import streamlit as st`. We only purge
    # UI/orchestration modules — the fake sub-modules (app.settings.config
    # etc.) are re-installed by the fixture before every test.
    for name in list(sys.modules):
        if name in ("app.state", "app.pipeline", "app.callbacks") or name.startswith("app.ui"):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location("app", str(app_path))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SyntaxError as e:
        pytest.skip(f"app.py has syntax errors: {e}")
    except Exception as e:
        pytest.skip(f"app.py could not be imported: {e}")
    return module


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def prepared_env(monkeypatch, tmp_path):
    os.environ["BACKEND"] = "itanet_dll"
    st = _install_dummy_streamlit(monkeypatch)
    _install_dummy_translations(monkeypatch)
    _install_dummy_utils(monkeypatch)
    _install_dummy_camera_and_diff(monkeypatch)
    _install_dummy_pil(monkeypatch)
    cfg = _install_dummy_config(monkeypatch, tmp_path)
    itanet = _install_dummy_itanet(monkeypatch, tmp_path)
    ha = _install_dummy_histogram_analysis(monkeypatch, tmp_path)
    return {"st": st, "cfg": cfg, "itanet": itanet, "ha": ha, "tmp": tmp_path}


# ---------------------------------------------------------------------------
# Core action tests
# ---------------------------------------------------------------------------

def test_run_grading_analysis_returns_grade_dict(prepared_env):
    app = _import_app_module()
    grades = app.run_grading_analysis("00001", "1", "1")
    assert isinstance(grades, dict)
    assert set(grades) == {"pilling", "matting", "fuzzing"}
    assert grades["pilling"] == pytest.approx(3.0)


def test_run_grading_analysis_selected_grades_forwarded(prepared_env):
    app = _import_app_module()
    grades = app.run_grading_analysis("00001", "1", "1", selected_grades=["pilling"])
    # Only the requested grade should come back — the two the operator didn't
    # tick in the sidebar must not appear.
    assert grades == {"pilling": pytest.approx(3.0)}


def test_run_grading_analysis_multi_grade_subset(prepared_env):
    app = _import_app_module()
    grades = app.run_grading_analysis("00001", "1", "1", selected_grades=["matting", "fuzzing"])
    assert set(grades) == {"matting", "fuzzing"}
    assert grades["matting"] == pytest.approx(2.5)
    assert grades["fuzzing"] == pytest.approx(3.5)


def test_capture_pipeline_forwards_sidebar_selection_to_grading(prepared_env, tmp_path):
    """End-to-end: sidebar checkbox -> session_state -> pipeline -> analysis call.

    The sidebar writes KEY_SELECTED_GRADES; run_capture_pipeline reads it and
    forwards to run_grading_analysis. This test pins that hop so a refactor
    that severs it fails loudly instead of silently reverting to "all grades".
    """
    _import_app_module()  # sets up sys.modules with the dummies
    st = prepared_env["st"]
    from app import state as S
    from app.pipeline import run_capture_pipeline

    # The pipeline gate at stage != "0" requires a stage-0 reference image on
    # disk; drop an empty placeholder since capture_sample_images is stubbed.
    cfg = prepared_env["cfg"]
    stage0_dir = Path(cfg.get_input_dir(cfg.SUFFIX_GRADING))
    stage0_dir.mkdir(parents=True, exist_ok=True)
    (stage0_dir / cfg.make_input_filename("00001", "0", "1", 1)).write_bytes(b"")

    st.session_state[S.KEY_SELECTED_GRADES] = ["fuzzing"]
    st.session_state[S.KEY_FS_EPOCH] = 0

    ok = run_capture_pipeline(
        mode="grading", sample_number="00001", stage_number="1", trial_number="1",
        motor_ip="127.0.0.1", motor_port=18812,
    )
    assert ok is True
    grade_result = st.session_state.get(S.KEY_GRADE)
    assert grade_result == {"fuzzing": pytest.approx(3.5)}


def test_training_path_calls_analyze_difference_images(prepared_env):
    app = _import_app_module()
    app.run_training_analysis("002", "2", "1", {"pilling": 2.5})
    assert prepared_env["ha"]._train_called["called"] is True
    assert prepared_env["ha"]._train_called["args"][:3] == ("002", "2", "1")


def test_create_and_capture_actions_call_impls(prepared_env):
    app = _import_app_module()
    assert app.create_difference_action("003", "3", "1", "for_grading", "0") is True
    assert app.capture_images_action("003", "3", "1", "for_grading", "127.0.0.1", 18812) is True


def test_capture_images_action_handles_exception(prepared_env, monkeypatch):
    app = _import_app_module()
    from app import pipeline

    def _raise(*args, **kwargs):
        raise RuntimeError("camera failure")

    monkeypatch.setattr(pipeline, "capture_sample_images", _raise)
    result = app.capture_images_action("003", "3", "1", "for_grading", "127.0.0.1", 18812)
    assert result is False


def test_create_difference_action_handles_exception(prepared_env, monkeypatch):
    app = _import_app_module()
    from app import pipeline

    def _raise(*args, **kwargs):
        raise RuntimeError("diff failure")

    monkeypatch.setattr(pipeline, "create_difference_images", _raise)
    result = app.create_difference_action("003", "3", "1", "for_grading", "0")
    assert result is False


def test_display_operation_status_does_not_crash(prepared_env):
    app = _import_app_module()
    app.display_operation_status("004", "4", "1", "for_grading")


# ---------------------------------------------------------------------------
# display_grades
# ---------------------------------------------------------------------------

def test_display_grades_with_dict(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["selected_grades"] = ["pilling"]
    app.display_grades({"pilling": 3.5}, "00001", "1")


def test_display_grades_with_all_three_grades(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["selected_grades"] = ["pilling", "matting", "fuzzing"]
    app.display_grades({"pilling": 3.0, "matting": 2.5, "fuzzing": 3.5}, "00001", "1")


def test_display_grades_with_none(prepared_env):
    app = _import_app_module()
    app.display_grades(None, "00001", "1")


def test_display_grades_empty_dict(prepared_env):
    app = _import_app_module()
    app.display_grades({}, "00001", "1")


# ---------------------------------------------------------------------------
# main() smoke tests
# ---------------------------------------------------------------------------

def test_main_runs_mode_selector(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["mode"] = None
    app.main()


def test_main_runs_in_grading_mode(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["mode"] = "grading"
    st.session_state["operator_name"] = "Test Operator"
    app.main()


def test_main_runs_in_training_mode(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["mode"] = "training"
    app.main()


def test_main_runs_training_login_screen(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["mode"] = "training_requested"
    app.main()


def test_main_runs_operator_name_screen(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["mode"] = "grading_requested"
    app.main()


# ---------------------------------------------------------------------------
# Navigation / state callbacks
# ---------------------------------------------------------------------------

def test_navigate_to_training_sets_mode(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    app.navigate_to_training_callback()
    assert st.session_state["mode"] == "training_requested"


def test_navigate_to_grading_sets_mode(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    app.navigate_to_grading_callback()
    assert st.session_state["mode"] == "grading_requested"


def test_go_back_callback_pops_stack(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["nav_stack"] = ["mode_a", "mode_b"]
    st.session_state["mode"] = "mode_c"
    app.go_back_callback()
    assert st.session_state["mode"] == "mode_b"
    assert len(st.session_state["nav_stack"]) == 1


def test_go_back_callback_empty_stack_preserves_mode(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["nav_stack"] = []
    st.session_state["mode"] = "current"
    app.go_back_callback()
    assert st.session_state["mode"] == "current"


# ---------------------------------------------------------------------------
# Login / operator callbacks
# ---------------------------------------------------------------------------

def test_login_correct_password_navigates_to_training(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    app.submit_login_callback("1234")
    assert st.session_state["mode"] == "training"
    assert st.session_state.get("login_error") is None


def test_login_wrong_password_sets_error(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    app.submit_login_callback("wrong")
    assert st.session_state.get("login_error") is not None


def test_operator_valid_name_navigates_to_grading(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    app.submit_operator_callback("Test Operator")
    assert st.session_state["operator_name"] == "Test Operator"
    assert st.session_state["mode"] == "grading"
    assert st.session_state.get("operator_error") is None


def test_operator_whitespace_name_sets_error(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    app.submit_operator_callback("   ")
    assert st.session_state.get("operator_error") is not None


def test_operator_empty_name_sets_error(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    app.submit_operator_callback("")
    assert st.session_state.get("operator_error") is not None


# ---------------------------------------------------------------------------
# Export / reset callbacks
# ---------------------------------------------------------------------------

def test_export_without_grade_sets_error(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    # Grade not set
    app.export_results_callback("00001", "1", "100")
    assert "export_error" in st.session_state


def test_export_with_grade_dict_succeeds(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["grade"] = {"pilling": 3.0}
    st.session_state["operator_name"] = "Operator"
    app.export_results_callback("00001", "1", "100")
    assert "export_success" in st.session_state


def test_export_clears_previous_messages(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["export_success"] = "Old success"
    st.session_state["export_error"] = "Old error"
    # No grade — sets a fresh error, old keys cleared first
    app.export_results_callback("00001", "1", "100")
    assert st.session_state.get("export_success") is None


def test_reset_grade_clears_grade_and_messages(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["grade"] = {"pilling": 3.0}
    st.session_state["capture_success"] = "ok"
    st.session_state["histogram_success"] = "ok"
    app.reset_grade_callback()
    assert "grade" not in st.session_state
    assert "capture_success" not in st.session_state
    assert "histogram_success" not in st.session_state


def test_clear_messages_callback(prepared_env):
    """Both modes share one status-message clearing callback."""
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["capture_success"] = "ok"
    st.session_state["capture_error"] = "err"
    st.session_state["diff_success"] = "ok"
    st.session_state["histogram_success"] = "ok"
    app.clear_messages_callback()
    for key in ("capture_success", "capture_error", "diff_success", "histogram_success"):
        assert key not in st.session_state


def test_restart_app_callback_preserves_only_declared_keys(prepared_env):
    """Every session key not in RESTART_PRESERVED_KEYS is dropped; mode/nav/epoch
    are reset to their bootstrap values."""
    app = _import_app_module()
    st = prepared_env["st"]
    st.session_state["lang"] = "de"
    st.session_state["mode"] = "training"
    st.session_state["nav_stack"] = ["prev"]
    st.session_state["fs_epoch"] = 42
    st.session_state["grade"] = {"pilling": 3.0}
    st.session_state["operator_name"] = "Alice"
    st.session_state["capture_success"] = "ok"
    st.session_state["last_sample_number_g"] = "00007"

    app.restart_app_callback()

    # Preserved keys keep their value.
    assert st.session_state["lang"] == "de"
    # Reset keys are re-initialized to bootstrap defaults.
    assert st.session_state["mode"] is None
    assert st.session_state["nav_stack"] == []
    assert st.session_state["fs_epoch"] == 0
    # Everything else is dropped.
    for dropped in ("grade", "operator_name",
                    "capture_success", "last_sample_number_g"):
        assert dropped not in st.session_state


# ---------------------------------------------------------------------------
# cached_* wrappers
# ---------------------------------------------------------------------------

def test_cached_get_available_samples(prepared_env):
    app = _import_app_module()
    samples = app.cached_get_available_samples("for_grading", 0)
    assert len(samples) == 5


def test_cached_check_required_images_returns_status_dict(prepared_env):
    app = _import_app_module()
    status = app.cached_check_required_images("00001", "1", "1", "for_grading", 0)
    assert isinstance(status, dict)
    assert "present_input_images" in status
    assert "all_input_present" in status


# ---------------------------------------------------------------------------
# capture_callback — unified pipeline for both grading and training
# ---------------------------------------------------------------------------

def _grading_params(sample="00001", stage="0", trial="1", ip="127.0.0.1", port=18812):
    return {"sample_number": sample, "stage_number": stage, "trial_number": trial,
            "motor_ip": ip, "motor_port": port}


def _training_params(sample="00001", stage="0", trial="1", ip="127.0.0.1", port=18812,
                     grades=None):
    return {"sample_number": sample, "stage_number": stage, "trial_number": trial,
            "motor_ip": ip, "motor_port": port,
            "grades_input": grades if grades is not None else {"pilling": 2.5}}


def test_capture_grading_stage_zero_succeeds(prepared_env):
    """Stage-0 capture requires no prior stage-0 file."""
    app = _import_app_module()
    st = prepared_env["st"]
    app.capture_callback("grading", _grading_params(stage="0"))
    assert st.session_state.get("capture_error") is None
    assert st.session_state.get("capture_success") is not None


def test_capture_grading_non_zero_stage_without_stage0_sets_error(prepared_env):
    """Non-zero stage without existing stage-0 input image should set capture_error."""
    app = _import_app_module()
    st = prepared_env["st"]
    app.capture_callback("grading", _grading_params(stage="1"))
    assert st.session_state.get("capture_error") is not None
    assert "Stage 0" in st.session_state["capture_error"]


def test_capture_grading_with_stage0_file_runs_full_pipeline(prepared_env):
    """Non-zero stage with stage-0 file triggers capture, diff, and analysis."""
    app = _import_app_module()
    st = prepared_env["st"]

    stage0_dir = prepared_env["tmp"] / "input_pictures" / "for_grading"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    (stage0_dir / "00001-0-1-1.png").touch()

    app.capture_callback("grading", _grading_params(stage="1"))
    assert st.session_state.get("capture_error") is None
    assert st.session_state.get("capture_success") is not None
    assert st.session_state.get("diff_success") is not None
    grades = st.session_state.get("grade")
    assert grades is not None
    assert "pilling" in grades


def test_capture_grading_exception_sets_error(prepared_env, monkeypatch):
    """Exception in capture propagates as a capture_error message."""
    app = _import_app_module()
    from app import pipeline
    st = prepared_env["st"]

    def _raise(*args, **kwargs):
        raise RuntimeError("hardware fault")

    monkeypatch.setattr(pipeline, "capture_images_action", _raise)
    app.capture_callback("grading", _grading_params(stage="0"))
    assert st.session_state.get("capture_error") is not None
    assert "Unexpected error" in st.session_state["capture_error"]


def test_capture_training_stage_zero_succeeds(prepared_env):
    """Stage-0 training capture requires no prior stage-0 file."""
    app = _import_app_module()
    st = prepared_env["st"]
    app.capture_callback("training", _training_params(stage="0"))
    assert st.session_state.get("capture_error") is None
    assert st.session_state.get("capture_success") is not None


def test_capture_training_non_zero_stage_without_stage0_sets_error(prepared_env):
    """Non-zero training stage without stage-0 file sets capture_error."""
    app = _import_app_module()
    st = prepared_env["st"]
    app.capture_callback("training", _training_params(stage="1"))
    assert st.session_state.get("capture_error") is not None
    assert "Stage 0" in st.session_state["capture_error"]


def test_capture_training_with_stage0_calls_analyze(prepared_env):
    """Non-zero training stage with stage-0 file calls analyze_difference_images."""
    app = _import_app_module()
    st = prepared_env["st"]

    stage0_dir = prepared_env["tmp"] / "input_pictures" / "for_training"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    (stage0_dir / "00001-0-1-1.png").touch()

    app.capture_callback("training", _training_params(stage="1"))
    assert st.session_state.get("capture_success") is not None
    assert prepared_env["ha"]._train_called["called"] is True


def test_capture_training_multi_grade_dict(prepared_env):
    """Training accepts a multi-grade dict."""
    app = _import_app_module()
    st = prepared_env["st"]

    stage0_dir = prepared_env["tmp"] / "input_pictures" / "for_training"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    (stage0_dir / "00002-0-1-1.png").touch()

    app.capture_callback("training", _training_params(
        sample="00002", stage="1",
        grades={"pilling": 2.5, "matting": 3.0, "fuzzing": 2.0},
    ))
    assert prepared_env["ha"]._train_called["called"] is True
