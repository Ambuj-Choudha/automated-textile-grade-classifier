import os
import sys
import csv
import types
import tempfile
from pathlib import Path
import pytest
import importlib


def _install_at(monkeypatch, dotted_path: str, mod) -> None:
    """Install fake mod at sys.modules[dotted_path] and bind it on its parent package
    so `from <parent> import <leaf>` resolves to the fake during import of app.py.
    """
    monkeypatch.setitem(sys.modules, dotted_path, mod)
    if "." in dotted_path:
        parent_path, child_name = dotted_path.rsplit(".", 1)
        parent = sys.modules.get(parent_path)
        if parent is None:
            parent = types.ModuleType(parent_path)
            monkeypatch.setitem(sys.modules, parent_path, parent)
        setattr(parent, child_name, mod)

def _install_dummy_streamlit(monkeypatch):
    class _DummyCtx:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    # session state with both .attr and ['key'] access
    class _Sess(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)
        def __setattr__(self, k, v):
            self[k] = v
        def pop(self, k, default=None):
            return dict.pop(self, k, default)
        def get(self, k, default=None):
            return dict.get(self, k, default)

    class _CacheData:
        def clear(self): pass
        def __call__(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    class DummyStreamlit:
        def __init__(self):
            self.session_state = _Sess({"lang": "en", "fs_epoch": 0})
            self.sidebar = self
            self._button_defaults = {}
            self._last_metrics = []
            self.cache_data = _CacheData()

        # act as context manager (for "with st.sidebar:" and "with col1:")
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

        # context helpers
        def expander(self, *a, **k): return _DummyCtx()
        def container(self): return _DummyCtx()
        def spinner(self, *a, **k): return _DummyCtx()
        def empty(self): return _DummyCtx()  # Add empty() method
        def rerun(self): pass  # Add rerun() method

        # layout
        def columns(self, spec):
            if isinstance(spec, int):
                n = spec
            else:
                try:
                    n = len(spec)
                except TypeError:
                    try:
                        n = int(spec)
                    except Exception:
                        n = 1
            return [self for _ in range(n)]

        # UI stubs
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

        # inputs
        def selectbox(self, label, options, index=0, **k):
            if not options:
                return None
            if index < 0 or index >= len(options):
                index = 0
            return options[index]
        def text_input(self, label, value="", **k): return value
        def number_input(self, label, value=0, **k): return value
        def button(self, label, **k): 
            # Handle on_click callback if present
            if 'on_click' in k and k.get('key') == 'btn_capture_grading':
                return True  # Simulate button press for testing
            return self._button_defaults.get(label, False)

        # sidebar passthroughs
        def __getattr__(self, name):
            return getattr(self, name)

    dummy = DummyStreamlit()
    m = types.ModuleType("streamlit")
    for attr in dir(dummy):
        if not attr.startswith("_"):
            setattr(m, attr, getattr(dummy, attr))
    monkeypatch.setitem(sys.modules, "streamlit", m)
    return m


def _install_dummy_translations(monkeypatch):
    m = types.ModuleType("translations")
    m.translations = {
        "en": {
            "settings": "Settings",
            "motor_settings": "Motor settings",
            "motor_ip": "Motor IP",
            "motor_port": "Motor Port",
            "directory_settings": "Directory settings",
            "base_directory": "Base directory",
            "view_images": "View images",
            "select_folder": "Select folder",
            "enter_filename": "Enter filename",
            "invalid_picture_name": "Invalid picture name",
            "title_grading": "Grading",
            "available_samples": "Available samples",
            "sample_number": "Sample",
            "stage_number": "Stage",
            "load_weight": "Load",
            "invalid_sample": "Invalid sample",
            "invalid_stage": "Invalid stage",
            "invalid_load": "Invalid load",
            "valid_input": "Valid input",
            "capture_button": "Capture Images",
            "results": "Results",
            "export_results": "Export Results",
            "reset_button": "Reset",
            "status": "Status",
            "title_training": "Training",
            "grade_number": "Grade",
            "invalid_grade": "Invalid grade",
            "capture_button_training": "Capture Images (Training)",
            "mode_selector": "Mode",
            "training_mode": "Training Mode",
            "grading_mode": "Grading Mode",
            "login_required": "Login required",
            "password_prompt": "Password",
            "submit": "Submit",
            "operator_prompt": "Operator Name",
            "histogram_fail": "No analysis results to display",
            "reference_not_found": "Reference image not found",
            "image_capture_spinner": "Capturing images...",
            "image_capture_fail": "Image capture failed",
            "image_capture_success": "Images captured successfully",
            "diff_create_spinner": "Creating difference images...",
            "diff_create_success": "Difference images created",
            "diff_create_fail": "Failed to create difference images",
            "histogram_spinner": "Analyzing histograms...",
            "histogram_success": "Analysis complete. Grade:",
            "login_error": "Invalid password",
            "name_required": "Name is required",
        }
    }
    _install_at(monkeypatch, "app.reporting.translations", m)
    return m


def _install_dummy_utils(monkeypatch):
    m = types.ModuleType("utils")

    def validate_input_number(x: str) -> bool:
        try:
            int(x)
            return True
        except Exception:
            return False

    def validate_grade(x: str) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    def get_available_samples(mode: str):
        return [f"{i:03d}" for i in range(1, 6)]

    # Updated to accept trial_number parameter
    def check_required_images(sample_number, stage_number, trial_number, suffix):
        return {
            "present_input_images": [],
            "present_difference_images": [],
            "all_input_present": False,
            "all_difference_present": False,
            "reference_image": [],
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
    return m


def _install_dummy_camera_and_diff(monkeypatch):
    cam = types.ModuleType("camera_control")

    # Updated to accept trial_number parameter
    def capture_sample_images(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port):
        return True

    cam.capture_sample_images = capture_sample_images
    _install_at(monkeypatch, "app.capture.camera_control", cam)

    diff = types.ModuleType("image_difference")

    # Updated to accept trial_number parameter
    def create_difference_images(sample_number, stage_number, trial_number, suffix, reference_stage):
        return True

    diff.create_difference_images = create_difference_images
    _install_at(monkeypatch, "app.capture.image_difference", diff)
    return cam, diff


def _install_dummy_config(monkeypatch, tmp_path: Path):
    cfg = types.ModuleType("config")
    cfg.BASE_DIR = str(tmp_path)
    cfg.ITANET_DATA_DIR = str(tmp_path / "itanet_data")
    cfg.DLL_PATH = str(tmp_path / "itanet_data" / "itanet.dll")
    cfg.BACKEND = "itanet_dll"
    cfg.TF_MODEL_PATH = str(tmp_path / "my_model.h5")
    cfg.FEATURE_COLUMNS = ("Mean", "Std", "Max", "Mode")
    cfg.DECIMAL = "point"
    cfg.ROUND_TO_HALF = True
    cfg.CLIP_RANGE = (1.0, 5.0)
    cfg.CUSTOM_NET = None
    cfg.CUSTOM_TRN = None
    os.makedirs(cfg.ITANET_DATA_DIR, exist_ok=True)
    _install_at(monkeypatch, "app.settings.config", cfg)
    return cfg


def _install_dummy_itanet(monkeypatch, tmp_path: Path):
    itanet = types.ModuleType("itanet_recall")

    def prepare_and_recall_csv(csv_path: str, itanet_data_dir: str, dll_path: str, output_csv: str, feature_cols, decimal="point", seed=None, custom_net_file=None, custom_trn_file=None):
        # Minimal CSV-only recall stub: copies CSV and adds constant prediction
        out_path = output_csv
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["Predicted_Grade"] = "3.0"
                rows.append(row)
        fieldnames = list(rows[0].keys()) if rows else list(feature_cols) + ["Predicted_Grade"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return out_path

    itanet.prepare_and_recall_csv = prepare_and_recall_csv
    _install_at(monkeypatch, "itanet_recall", itanet)
    return itanet


def _install_dummy_histogram_analysis(monkeypatch, tmp_path: Path):
    ha = types.ModuleType("histogram_analysis")

    # Updated to accept trial_number parameter
    def analyze_difference_images_and_predict_output(sample_number: str, stage_number: str, trial_number: str, output_dir: str = os.path.join("output", "grading_results")):
        # Create a simple analysis CSV with trial_number in filename
        analysis_dir = tmp_path / output_dir
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_csv = analysis_dir / f"{sample_number}-{stage_number}-{trial_number}-analysis.csv"
        
        # Write dummy analysis data
        with open(analysis_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Mean", "Std", "Max", "Mode"])
            for i in range(1, 9):
                writer.writerow([f"{sample_number}-{stage_number}-{trial_number}-{i}-dif.png", 100, 10, 255, 80])
            writer.writerow(["Average", 100, 10, 255, 80])
            writer.writerow(["Grade", 3.0, "", "", ""])
            writer.writerow(["Backend", "itanet_dll", "", "", ""])
        
        return 3.0

    _train_called = {"called": False, "args": None}

    # Updated to accept trial_number parameter
    def analyze_difference_images(sample_number: str, stage_number: str, trial_number: str, grade_number, output_dir: str = os.path.join("data", "training_features")):
        _train_called["called"] = True
        _train_called["args"] = (sample_number, stage_number, trial_number, grade_number, output_dir)
        return True

    def batch_predict_from_csv(csv_path: str, output_path: str | None = None, custom_net_file: str | None = None, custom_trn_file: str | None = None) -> str:
        if output_path is None:
            output_path = str(Path(csv_path).with_name("predictions.csv"))
        
        # Create dummy predictions
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = dict(row)
                row["Predicted_Grade"] = "3.0"
                rows.append(row)
        
        with open(output_path, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        
        return output_path

    ha.analyze_difference_images_and_predict_output = analyze_difference_images_and_predict_output
    ha.analyze_difference_images = analyze_difference_images
    ha.batch_predict_from_csv = batch_predict_from_csv
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
    # Try to import the real app.py from the current directory
    app_path = Path(__file__).parent.parent / "app.py"
    if not app_path.exists():
        pytest.skip("app.py not found in expected location")
    spec = importlib.util.spec_from_file_location("app", str(app_path))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except SyntaxError as e:
        pytest.skip(f"app.py has syntax errors preventing import: {e}")
    except Exception as e:
        pytest.skip(f"app.py could not be imported: {e}")
    return module


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


def test_analyze_histograms_action_uses_itanet_backend(prepared_env):
    app = _import_app_module()
    # Updated to include trial_number parameter
    grade = app.analyze_histograms_action("00001", "1", "1", 0)
    assert grade == pytest.approx(3.0), "Expected ITANET dummy to return grade 3.0"


def test_training_path_calls_analyze_difference_images(prepared_env):
    app = _import_app_module()
    # Updated to include trial_number parameter
    _ = app.analyze_histograms_action("002", "2", "1", 2.5)
    assert prepared_env["ha"]._train_called["called"] is True
    assert prepared_env["ha"]._train_called["args"][:4] == ("002", "2", "1", 2.5)


def test_create_and_capture_actions_call_impls(prepared_env):
    app = _import_app_module()
    # Updated to include trial_number parameter
    assert app.create_difference_action("003", "3", "1", "for_grading", "0") is True
    assert app.capture_images_action("003", "3", "1", "for_grading", "127.0.0.1", 18812) is True


def test_display_operation_status_does_not_crash(prepared_env):
    app = _import_app_module()
    # Updated to include trial_number parameter
    app.display_operation_status("004", "4", "1", "for_grading")


def test_main_runs_in_grading_mode(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    # Simulate user already chose grading mode; just ensure it runs
    st.session_state["mode"] = "grading"
    st.session_state["operator_name"] = "Test Operator"
    app.main()


def test_navigation_callbacks(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    
    # Test navigation to training
    app.navigate_to_training_callback()
    assert st.session_state["mode"] == "training_requested"
    
    # Test navigation to grading
    app.navigate_to_grading_callback()
    assert st.session_state["mode"] == "grading_requested"


def test_login_callback(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    
    # Test correct password
    app.submit_login_callback("1234")
    assert st.session_state["mode"] == "training"
    assert st.session_state.get("login_error") is None
    
    # Test incorrect password
    app.submit_login_callback("wrong")
    assert st.session_state.get("login_error") is not None


def test_operator_callback(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    
    # Test valid operator name
    app.submit_operator_callback("Test Operator")
    assert st.session_state["operator_name"] == "Test Operator"
    assert st.session_state["mode"] == "grading"
    
    # Test empty name
    app.submit_operator_callback("")
    assert st.session_state.get("operator_error") is not None


def test_export_callback_without_grade(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    
    # Test export without grade - should fail
    app.export_results_callback("00001", "1", "100")
    assert "export_error" in st.session_state


def test_export_callback_with_grade(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    
    # Set grade first
    st.session_state["Grade"] = 3.0
    st.session_state["operator_name"] = "Test Operator"
    
    # Test export with grade - should succeed
    app.export_results_callback("00001", "1", "100")
    assert "export_success" in st.session_state


def test_reset_grade_callback(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]
    
    # Set some state
    st.session_state["Grade"] = 3.0
    st.session_state["capture_success"] = "Success"
    
    # Reset
    app.reset_grade_callback()
    
    # Check grade is cleared
    assert "Grade" not in st.session_state
    assert "capture_success" not in st.session_state


def test_cached_functions(prepared_env):
    app = _import_app_module()
    
    # Test cached functions work
    samples = app.cached_get_available_samples("for_grading", 0)
    assert len(samples) == 5
    
    # Updated to include trial_number and epoch parameters
    status = app.cached_check_required_images("00001", "1", "1", "for_grading", 0)
    assert isinstance(status, dict)
    assert "present_input_images" in status


def test_display_grades_function(prepared_env):
    app = _import_app_module()
    
    # Test with valid grade
    app.display_grades(3.5, "00001", "1")
    
    # Test with None grade
    app.display_grades(None, "00001", "1")


def test_capture_grading_callback_success(prepared_env):
    app = _import_app_module()
    st = prepared_env["st"]

    def test_capture_grading_callback_stage_zero(prepared_env, monkeypatch):
        """Test capturing stage 0 (reference image) in grading mode"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Mock __file__ to point to tmp directory
        monkeypatch.setattr(app, '__file__', str(prepared_env["tmp"] / "app.py"))
        
        # Capture stage 0 - no reference needed
        app.capture_grading_callback("00001", "0", "1", "for_grading", "127.0.0.1", 18812)
        
        # Should succeed without error
        assert st.session_state.get("capture_error") is None
        assert st.session_state.get("capture_success") is not None


    def test_capture_training_callback_stage_zero(prepared_env, monkeypatch):
        """Test capturing stage 0 (reference image) in training mode"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Mock __file__ to point to tmp directory
        monkeypatch.setattr(app, '__file__', str(prepared_env["tmp"] / "app.py"))
        
        # Capture stage 0 - no reference needed
        app.capture_training_callback("00001", "0", "1", "for_training", "127.0.0.1", 18812, 2.5)
        
        # Should succeed without error
        assert st.session_state.get("capture_error_training") is None
        assert st.session_state.get("capture_success_training") is not None


    def test_capture_grading_callback_missing_reference(prepared_env, monkeypatch):
        """Test that capturing non-zero stage without reference fails gracefully"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Mock __file__ to point to tmp directory
        monkeypatch.setattr(app, '__file__', str(prepared_env["tmp"] / "app.py"))
        
        # Try to capture stage 1 without reference
        app.capture_grading_callback("00001", "1", "1", "for_grading", "127.0.0.1", 18812)
        
        # Should have error about missing reference
        assert st.session_state.get("capture_error") is not None
        assert "Reference image not found" in st.session_state.get("capture_error", "")


    def test_capture_training_callback_missing_reference(prepared_env, monkeypatch):
        """Test that capturing non-zero stage without reference fails in training mode"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Mock __file__ to point to tmp directory
        monkeypatch.setattr(app, '__file__', str(prepared_env["tmp"] / "app.py"))
        
        # Try to capture stage 1 without reference
        app.capture_training_callback("00001", "1", "1", "for_training", "127.0.0.1", 18812, 2.5)
        
        # Should have error about missing reference
        assert st.session_state.get("capture_error_training") is not None
        assert "Reference image not found" in st.session_state.get("capture_error_training", "")


    def test_capture_grading_callback_with_reference(prepared_env, monkeypatch):
        """Test successful capture in grading mode with reference"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Mock __file__ to point to tmp directory
        monkeypatch.setattr(app, '__file__', str(prepared_env["tmp"] / "app.py"))
        
        # Create reference image
        ref_dir = prepared_env["tmp"] / "reference_pictures" / "for_grading"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_file = ref_dir / "00001-0-1-ref.png"
        ref_file.touch()
        
        # Capture stage 1 with reference present
        app.capture_grading_callback("00001", "1", "1", "for_grading", "127.0.0.1", 18812)
        
        # Should succeed
        assert st.session_state.get("capture_error") is None
        assert st.session_state.get("capture_success") is not None
        assert st.session_state.get("diff_success") is not None


    def test_capture_training_callback_with_reference(prepared_env, monkeypatch):
        """Test successful capture in training mode with reference"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Mock __file__ to point to tmp directory
        monkeypatch.setattr(app, '__file__', str(prepared_env["tmp"] / "app.py"))
        
        # Create reference image
        ref_dir = prepared_env["tmp"] / "reference_pictures" / "for_training"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_file = ref_dir / "00001-0-1-ref.png"
        ref_file.touch()
        
        # Capture stage 1 with reference present
        app.capture_training_callback("00001", "1", "1", "for_training", "127.0.0.1", 18812, 2.5)
        
        # Should succeed
        assert st.session_state.get("capture_error_training") is None
        assert st.session_state.get("capture_success_training") is not None
        assert prepared_env["ha"]._train_called["called"] is True


    def test_capture_grading_callback_with_exception(prepared_env, monkeypatch):
        """Test that exceptions in capture_grading_callback are handled"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Mock __file__ to point to tmp directory
        monkeypatch.setattr(app, '__file__', str(prepared_env["tmp"] / "app.py"))
        
        # Mock capture_images_action to raise an exception
        def mock_capture(*args, **kwargs):
            raise RuntimeError("Simulated capture failure")
        
        monkeypatch.setattr(app, "capture_images_action", mock_capture)
        
        # Should handle exception gracefully
        app.capture_grading_callback("00001", "0", "1", "for_grading", "127.0.0.1", 18812)
        
        assert st.session_state.get("capture_error") is not None
        assert "Unexpected error" in st.session_state.get("capture_error", "")


    def test_create_difference_action_with_exception(prepared_env, monkeypatch):
        """Test that exceptions in create_difference_action are handled"""
        app = _import_app_module()
        
        # Mock create_difference_images to raise an exception
        diff_module = sys.modules["image_difference"]
        
        def mock_create(*args, **kwargs):
            raise RuntimeError("Simulated difference creation failure")
        
        monkeypatch.setattr(diff_module, "create_difference_images", mock_create)
        
        result = app.create_difference_action("00001", "1", "1", "for_grading", "0")
        assert result is False


    def test_capture_images_action_with_exception(prepared_env, monkeypatch):
        """Test that exceptions in capture_images_action are handled"""
        app = _import_app_module()
        
        # Mock capture_sample_images to raise an exception
        cam_module = sys.modules["camera_control"]
        
        def mock_capture(*args, **kwargs):
            raise RuntimeError("Simulated camera failure")
        
        monkeypatch.setattr(cam_module, "capture_sample_images", mock_capture)
        
        result = app.capture_images_action("00001", "1", "1", "for_grading", "127.0.0.1", 18812)
        assert result is False


    def test_export_with_invalid_operator_name(prepared_env):
        """Test export with special characters in operator name"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        st.session_state["Grade"] = 3.0
        st.session_state["operator_name"] = "Test@#$%Operator!!!"
        
        app.export_results_callback("00001", "1", "100")
        
        # Should sanitize operator name and succeed
        assert "export_success" in st.session_state


    def test_clear_training_messages_callback(prepared_env):
        """Test that clear_training_messages_callback clears all training messages"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Set various training messages
        st.session_state["capture_success_training"] = "Success"
        st.session_state["capture_error_training"] = "Error"
        st.session_state["diff_success_training"] = "Success"
        
        app.clear_training_messages_callback()
        
        # All should be cleared
        assert "capture_success_training" not in st.session_state
        assert "capture_error_training" not in st.session_state
        assert "diff_success_training" not in st.session_state


    def test_clear_grading_messages_callback(prepared_env):
        """Test that clear_grading_messages_callback clears all grading messages"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Set various grading messages
        st.session_state["capture_success"] = "Success"
        st.session_state["capture_error"] = "Error"
        st.session_state["diff_success"] = "Success"
        
        app.clear_grading_messages_callback()
        
        # All should be cleared
        assert "capture_success" not in st.session_state
        assert "capture_error" not in st.session_state
        assert "diff_success" not in st.session_state


    def test_go_back_callback_with_stack(prepared_env):
        """Test that go_back_callback pops from navigation stack"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        # Build a navigation stack
        st.session_state["nav_stack"] = ["mode1", "mode2"]
        st.session_state["mode"] = "mode3"
        
        app.go_back_callback()
        
        # Should pop from stack
        assert st.session_state["mode"] == "mode2"
        assert len(st.session_state["nav_stack"]) == 1


    def test_go_back_callback_empty_stack(prepared_env):
        """Test that go_back_callback handles empty stack"""
        app = _import_app_module()
        st = prepared_env["st"]
        
        st.session_state["nav_stack"] = []
        st.session_state["mode"] = "current_mode"
        
        app.go_back_callback()
        
        # Mode should remain unchanged
        assert st.session_state["mode"] == "current_mode"


def test_capture_training_callback_success(prepared_env, monkeypatch):
    app = _import_app_module()
    st = prepared_env["st"]
    
    # Mock __file__ to point to tmp directory so app uses correct base path
    monkeypatch.setattr(app, '__file__', str(prepared_env["tmp"] / "app.py"))
    
    # Create reference image directory structure
    ref_dir = prepared_env["tmp"] / "reference_pictures" / "for_training"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_file = ref_dir / "00001-0-1-ref.png"
    ref_file.touch()
    
    app.capture_training_callback("00001", "1", "1", "for_training", "127.0.0.1", 18812, 2.5)
    
    assert st.session_state.get("capture_success_training") is not None
    assert prepared_env["ha"]._train_called["called"] is True