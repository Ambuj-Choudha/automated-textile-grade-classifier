# Import custom libraries
import streamlit as st
import numpy as np
import os
from datetime import datetime
import sys
import base64
from PIL import Image

# Import custom modules
from app.capture.camera_control import capture_sample_images
from app.capture.image_difference import create_difference_images
from app.analysis.histogram_analysis import analyze_difference_images_and_predict_output, analyze_difference_images
from app.helpers.utils import validate_input_number, validate_grade, get_available_samples, check_required_images
from app.reporting.results import export_results
from app.reporting.translations import t
from app.settings import config as cfg

# ------------------------------------------------------------------
# Dev / runtime flags
AUTO_CLEAR_CACHE_ON_START = True   # Set False in production to keep caches between restarts
# ------------------------------------------------------------------

# Session-state key constants — single authoritative source for every key string.
# Grading and training share the same status keys; only one mode is active at a time.
_KEY_GRADE = "grade"
_KEY_CAPTURE_PROG = "capture_in_progress"
_KEY_PENDING_CAPTURE = "pending_capture"
_KEY_CAPTURE_ERROR = "capture_error"
_KEY_CAPTURE_SUCCESS = "capture_success"
_KEY_DIFF_SUCCESS = "diff_success"
_KEY_DIFF_ERROR = "diff_error"
_KEY_HIST_SUCCESS = "histogram_success"
_KEY_HIST_ERROR = "histogram_error"
_KEY_EXPORT_SUCCESS = "export_success"
_KEY_EXPORT_ERROR = "export_error"
_KEY_EXPORT_PDF = "export_pdf_path"
_KEY_LOGIN_ERROR = "login_error"
_KEY_OPERATOR_ERROR = "operator_error"

# Pipeline status messages — cleared as a group between runs
_STATUS_KEYS = (
    _KEY_CAPTURE_ERROR, _KEY_CAPTURE_SUCCESS,
    _KEY_DIFF_ERROR, _KEY_DIFF_SUCCESS,
    _KEY_HIST_ERROR, _KEY_HIST_SUCCESS,
)

# Streamlit page configuration: title, layout, and sidebar state
st.set_page_config(
    page_title="TexIQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _inject_css(path: str = "static/style.css") -> None:
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_inject_css()

# One-time startup hook (per server process)
if "bootstrapped" not in st.session_state:
    if AUTO_CLEAR_CACHE_ON_START:
        try:
            st.cache_data.clear()
            st.session_state["_cache_cleared_on_boot"] = True
        except Exception:
            pass
    else:
        st.session_state["_cache_cleared_on_boot"] = False
    st.session_state.bootstrapped = True

# --- Caching helpers to avoid repeated disk scans and image decoding ---
def _file_sig(path: str):
    """
    Lightweight signature for a file; changes if content likely changed.
    Uses high‑resolution mtime + size (cheap, good enough single-user).
    """
    try:
        stt = os.stat(path)
        return (stt.st_mtime_ns, stt.st_size)
    except FileNotFoundError:
        return (-1, -1)

@st.cache_data(show_spinner=False)
def cached_get_available_samples(mode: str, epoch: int):
    """
    Cache directory listing per mode.
    epoch (fs_epoch) busts cache after writes.
    """
    return get_available_samples(mode)

@st.cache_data(show_spinner=False)
def cached_check_required_images(sample_number: str, stage_number: str, trial_number: str, suffix: str, epoch: int):
    """
    Cache presence/absence analysis of required images.
    """
    return check_required_images(sample_number, stage_number, trial_number, suffix)

@st.cache_data(show_spinner=False, ttl=300)
def load_image_bytes(path: str, sig) -> bytes:
    """
    Cache image bytes keyed by (path, file signature).
    sig = (mtime_ns, size) so overwrites are detected.
    """
    with open(path, "rb") as f:
        return f.read()
# --- end caching helpers ---

# Set default language
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# ------------------------------------------------------------------
# Callback Functions for State Management
# ------------------------------------------------------------------

def restart_app_callback():
    """Callback to restart the app"""
    preserved = {
        "lang": st.session_state.get("lang", "en")
    }
    # Always clear caches on restart
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # Wipe everything else
    for k in list(st.session_state.keys()):
        if k not in preserved:
            del st.session_state[k]
    # Restore preserved keys + fresh counters
    st.session_state.update(preserved)
    st.session_state.fs_epoch = 0
    st.session_state.nav_stack = []
    st.session_state.mode = None

def _mode_suffix(mode: str) -> str:
    return cfg.SUFFIX_GRADING if mode == "grading" else cfg.SUFFIX_TRAINING


def run_capture_pipeline(mode, sample_number, stage_number, trial_number,
                         motor_ip, motor_port, grades_input=None):
    """Shared capture → diff → analyze pipeline for both grading and training.

    mode:          "grading" or "training"
    grades_input:  label dict — required for training, ignored for grading

    Reports progress via st.session_state status keys. Returns True on
    success, False if any step failed.
    """
    suffix = _mode_suffix(mode)

    if stage_number != "0":
        stage0_path = os.path.join(
            cfg.get_input_dir(suffix),
            cfg.make_input_filename(sample_number, "0", trial_number, 1),
        )
        if not os.path.exists(stage0_path):
            st.session_state[_KEY_CAPTURE_ERROR] = (
                f"Stage 0 images not found for trial {trial_number}. "
                "Please capture stage 0 first."
            )
            return False

    for k in _STATUS_KEYS:
        st.session_state[k] = None

    if not capture_images_action(sample_number, stage_number, trial_number,
                                 suffix, motor_ip, motor_port):
        st.session_state[_KEY_CAPTURE_ERROR] = t("image_capture_fail")
        return False
    st.session_state[_KEY_CAPTURE_SUCCESS] = t("image_capture_success")
    st.session_state.fs_epoch += 1  # bust directory-related caches

    if stage_number == "0":
        return True

    if not create_difference_action(sample_number, stage_number, trial_number,
                                    suffix, reference_stage="0"):
        st.session_state[_KEY_DIFF_ERROR] = t("diff_create_fail")
        return False
    st.session_state[_KEY_DIFF_SUCCESS] = t("diff_create_success")
    st.session_state.fs_epoch += 1

    if mode == "grading":
        grades = run_grading_analysis(
            sample_number, stage_number, trial_number,
            selected_grades=st.session_state.get("selected_grades", list(cfg.GRADES)),
        )
        if grades is None:
            st.session_state[_KEY_HIST_ERROR] = t("histogram_fail")
            return False
        st.session_state[_KEY_GRADE] = grades
        st.session_state[_KEY_HIST_SUCCESS] = t("histogram_success")
    else:
        run_training_analysis(sample_number, stage_number, trial_number, grades_input)
        st.session_state[_KEY_HIST_SUCCESS] = t("histogram_success")

    return True


def capture_callback(mode, params):
    """Unified callback for both grading and training capture buttons.

    params: dict passed straight to run_capture_pipeline (sample_number,
            stage_number, trial_number, motor_ip, motor_port, and grades_input
            for training).
    """
    try:
        st.session_state[_KEY_CAPTURE_PROG] = True
        run_capture_pipeline(mode=mode, **params)
    except Exception as e:
        st.session_state[_KEY_CAPTURE_ERROR] = f"Unexpected error: {str(e)}"
        print(f"Error in capture_callback ({mode}): {e}")
    finally:
        st.session_state[_KEY_CAPTURE_PROG] = False

def export_results_callback(sample_number, stage_number, load_weight, trial_number="1"):
    """Callback for export results button"""
    st.session_state.pop(_KEY_EXPORT_SUCCESS, None)
    st.session_state.pop(_KEY_EXPORT_ERROR, None)

    operator = st.session_state.get('operator_name', '')
    clean_operator_name = "".join(c for c in operator if c.isalnum() or c in ('-', '_')).strip() or "Unknown"
    pdf_filepath = os.path.join("reports", f"{sample_number}-{clean_operator_name}-report.pdf")

    try:
        grades = st.session_state.get(_KEY_GRADE)
        if not grades:
            st.session_state[_KEY_EXPORT_ERROR] = "Cannot export: No analysis results available. Please complete the grading process first."
            return

        success = export_results(sample_number, stage_number, load_weight, operator,
                                 trial_number=trial_number, grades=grades)

        if success:
            st.session_state[_KEY_EXPORT_SUCCESS] = f"PDF report saved to reports/{sample_number}-{clean_operator_name}-report.pdf!"
            st.session_state[_KEY_EXPORT_PDF] = pdf_filepath
        else:
            dirpath = os.path.join("output", "grading_results")
            matches = []
            if os.path.isdir(dirpath):
                matches = [f for f in os.listdir(dirpath) if f.startswith(f"{sample_number}-") and f.endswith("-analysis.csv")]
            if not matches:
                st.session_state[_KEY_EXPORT_ERROR] = f"Cannot export: No analysis files for sample '{sample_number}' found. Please complete image capture and analysis first."
            else:
                st.session_state[_KEY_EXPORT_ERROR] = "Export failed: Error processing analysis data."

    except Exception as e:
        st.session_state[_KEY_EXPORT_ERROR] = f"Export failed: {str(e)}"
        print(f"Export error details: {e}")

def open_pdf_callback():
    """Open the exported PDF in the default system viewer."""
    try:
        path = st.session_state.get(_KEY_EXPORT_PDF)
        if not path:
            st.session_state[_KEY_EXPORT_ERROR] = "No PDF available to open."
            return
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            st.session_state[_KEY_EXPORT_ERROR] = "PDF not found on disk."
            return

        if sys.platform.startswith("win"):
            os.startfile(abs_path)  # Windows default viewer
        elif sys.platform.startswith("linux"):
            import subprocess
            subprocess.run(["xdg-open", abs_path], check=False)
        else:
            st.session_state[_KEY_EXPORT_ERROR] = "Open PDF is supported only on Windows and Linux."
    except Exception as e:
        st.session_state[_KEY_EXPORT_ERROR] = f"Failed to open PDF: {e}"

def reset_grade_callback():
    """Callback for reset grade button"""
    for key in [_KEY_GRADE, _KEY_CAPTURE_SUCCESS, _KEY_CAPTURE_ERROR, _KEY_DIFF_SUCCESS,
                _KEY_DIFF_ERROR, _KEY_HIST_SUCCESS, _KEY_HIST_ERROR,
                _KEY_EXPORT_SUCCESS, _KEY_EXPORT_ERROR, _KEY_EXPORT_PDF]:
        st.session_state.pop(key, None)

def navigate_to_training_callback():
    """Callback for training mode button"""
    navigate_to("training_requested")

def navigate_to_grading_callback():
    """Callback for grading mode button"""
    navigate_to("grading_requested")

def submit_login_callback(password):
    """Callback for login submit button"""
    if password == cfg.ADMIN_PASSWORD:
        navigate_to("training")
        st.session_state[_KEY_LOGIN_ERROR] = None
    else:
        st.session_state[_KEY_LOGIN_ERROR] = t("login_error")

def submit_operator_callback(name):
    """Callback for operator submit button"""
    if name.strip():
        st.session_state.operator_name = name.strip()
        navigate_to("grading")
        st.session_state[_KEY_OPERATOR_ERROR] = None
    else:
        st.session_state[_KEY_OPERATOR_ERROR] = t("name_required")

def go_back_callback():
    """Callback for back button"""
    stack = st.session_state.get("nav_stack", [])
    if stack:
        st.session_state.mode = stack.pop()

def clear_messages_callback():
    """Callback to clear all pipeline status messages (both modes share these)."""
    for key in _STATUS_KEYS:
        st.session_state.pop(key, None)

# ------------------------------------------------------------------
# End Callback Functions
# ------------------------------------------------------------------

# 🌐 Language switcher in sidebar
st.sidebar.markdown("### 🌐 Language")
# Mapping display names to language codes
language_options = {
    "English": "en",
    "Deutsch": "de",
    "Français": "fr",
    "Español": "es"
}
# Get reverse mapping for display in dropdown
language_display = {v: k for k, v in language_options.items()}

# Default language (fallback if not yet set)
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Dropdown for language selection
selected_language = st.sidebar.selectbox(
    "Select language:",
    options=list(language_options.keys()),
    index=list(language_options.values()).index(st.session_state.lang),
    key="lang_select"  # stable key to avoid remounts on rerun
)
# Update session state
st.session_state.lang = language_options[selected_language]

# Sidebar: App control / restart section
with st.sidebar:
    st.markdown("---")
    st.markdown("#### ⚙️ App Control")
    st.button("🔄 Restart App", key="btn_restart_app", on_click=restart_app_callback)

# Function: Sidebar Settings
def setup_sidebar():
    with st.sidebar:
        st.header(t("settings"))

        st.subheader(t("select_grades"))
        all_grades = list(cfg.GRADES)
        default_grades = st.session_state.get("selected_grades", all_grades)
        selected_grades = st.multiselect(
            "",
            options=all_grades,
            default=default_grades,
            key="k_selected_grades",
            format_func=str.capitalize,
        )
        if not selected_grades:
            st.warning(t("grades_required"))
            selected_grades = list(st.session_state.get("selected_grades", all_grades))
        st.session_state.selected_grades = selected_grades

        st.subheader(t("motor_settings"))
        
        # Get last used values from session state (with fallbacks)
        default_motor_ip = st.session_state.get("last_motor_ip", "")
        default_motor_port = st.session_state.get("last_motor_port", 18812)
        
        # Create inputs with persistent defaults
        motor_ip = st.text_input(t("motor_ip"), value=default_motor_ip, key="k_motor_ip")
        motor_port = int(st.number_input(t("motor_port"), value=default_motor_port, key="k_motor_port"))
        
        # Save current values back to session state
        st.session_state.last_motor_ip = motor_ip
        st.session_state.last_motor_port = motor_port

        st.subheader(t("directory_settings"))
        
        # Optional: also persist base_dir if needed
        default_base_dir = st.session_state.get("last_base_dir", os.getcwd())
        base_dir = st.text_input(t("base_directory"), value=default_base_dir, key="k_base_dir")
        st.session_state.last_base_dir = base_dir
        
        st.markdown("---")

    return motor_ip, motor_port, base_dir

# Function: Image Viewer Expander
def show_image_viewer():
    valid_folders = [
        'data/input_pictures/for_training', 'data/difference_pictures/for_training',
        'data/input_pictures/for_grading', 'data/difference_pictures/for_grading',
    ]

    with st.expander(t("view_images")):
        col1, col2 = st.columns(2)
        with col1:
            selected_folder = st.selectbox(t("select_folder"), valid_folders, key="k_viewer_folder")
        with col2:
            # Help now shows examples without .png
            image_filename = st.text_input(t("enter_filename"), key="k_viewer_file", help="e.g., 001-1-1, 001-1-1-dif, 001-1-ref")

        # Normalize: if no extension provided, assume .png
        normalized_filename = (image_filename or "").strip()
        if normalized_filename and not os.path.splitext(normalized_filename)[1]:
            normalized_filename = f"{normalized_filename}.png"

        if selected_folder and normalized_filename:
            image_path = os.path.join(selected_folder, normalized_filename)
            if os.path.exists(image_path):
                sig = _file_sig(image_path)
                # Use actual basename for caption to avoid double .png
                st.image(load_image_bytes(image_path, sig), caption=os.path.basename(image_path), use_column_width=True)
            else:
                st.error(t("invalid_picture_name"))

def _render_logo():
    """Render the centered TexIQ logo."""
    _, col, _ = st.columns([2, 2, 2])
    with col:
        st.image(os.path.join("logos", "Logo_TexIQ_v1.0.jpg"), width=200)


# Per-mode UI descriptors used by _render_capture_column.
# Everything the two flows differ on lives here, so the renderer stays generic.
_TRAINING_GRADE_INPUTS = (
    # (grade_name, label_key, error_key, last_key, widget_key)
    ("pilling", "grade_number",         "invalid_grade",         "last_grade_number_t",         "k_grade_number_t"),
    ("matting", "matting_grade_number", "invalid_matting_grade", "last_matting_grade_number_t", "k_matting_grade_number_t"),
    ("fuzzing", "fuzzing_grade_number", "invalid_fuzzing_grade", "last_fuzzing_grade_number_t", "k_fuzzing_grade_number_t"),
)


def _render_capture_column(col, mode, motor_ip, motor_port, selected_grades=None):
    """Render the input + capture column for either grading or training.

    Returns (sample_number, stage_number, trial_number, extra, inputs_valid, all_valid).
    `extra` is the load_weight string (grading) or a grades_input dict (training).
    `inputs_valid` covers sample/stage/trial only — used to gate the status view.
    `all_valid`    also covers the mode-specific extra input(s).
    """
    is_grading = mode == "grading"
    suffix     = _mode_suffix(mode)
    tag        = "g" if is_grading else "t"

    with col:
        available = cached_get_available_samples(suffix, st.session_state.fs_epoch)
        st.info(f"{t('available_samples')}: {len(available)}")

        sample_number = st.text_input(t("sample_number"), value=st.session_state.get(f"last_sample_number_{tag}", "00000"), key=f"k_sample_number_{tag}", help="e.g., 00001, 00002, etc.")
        stage_number  = st.text_input(t("stage_number"),  value=st.session_state.get(f"last_stage_number_{tag}",  "0"),     key=f"k_stage_number_{tag}",  help="e.g., 125, 500, 1000, etc.")
        trial_number  = st.text_input(t("trial_number"),  value=st.session_state.get(f"last_trial_number_{tag}",  "1"),     key=f"k_trial_number_{tag}",  help="e.g., 1, 2, 3, etc.")
        st.session_state[f"last_sample_number_{tag}"] = sample_number
        st.session_state[f"last_stage_number_{tag}"]  = stage_number
        st.session_state[f"last_trial_number_{tag}"]  = trial_number

        # Mode-specific inputs
        load_weight = None
        grades_input: dict = {}
        extra_valid = True

        if is_grading:
            load_weight = st.text_input(t("load_weight"), value=st.session_state.get("last_load_weight_g", "0"), key="k_load_weight_g", help="e.g., 155, 415, etc.")
            st.session_state.last_load_weight_g = load_weight
            if not validate_input_number(load_weight):
                st.error(t("invalid_load"))
                extra_valid = False
        else:
            for name, label_key, err_key, last_key, widget_key in _TRAINING_GRADE_INPUTS:
                if name not in (selected_grades or ()):
                    continue
                val = st.text_input(t(label_key), value=st.session_state.get(last_key, "1"), key=widget_key, help="e.g., 1.5, 2, 2.5, etc.")
                st.session_state[last_key] = val
                if not validate_grade(val):
                    st.error(t(err_key))
                    extra_valid = False
                else:
                    grades_input[name] = float(val)

        sample_valid = validate_input_number(sample_number)
        stage_valid  = validate_input_number(stage_number)
        trial_valid  = validate_input_number(trial_number)

        if not sample_valid: st.error(t("invalid_sample"))
        if not stage_valid:  st.error(t("invalid_stage"))
        if not trial_valid:  st.error(t("invalid_trial"))

        inputs_valid = sample_valid and stage_valid and trial_valid
        all_valid    = inputs_valid and extra_valid

        if all_valid:
            st.success(t("valid_input"))
            image_status = cached_check_required_images(sample_number, stage_number, trial_number, suffix, st.session_state.fs_epoch)
            if image_status['all_input_present']:
                st.info(f"📁 {len(image_status['present_input_images'])} {t('available_samples').lower()}")

        st.markdown("---")

        button_label = t("capture_button" if is_grading else "capture_button_training")
        button_key   = "btn_capture_grading" if is_grading else "btn_capture_training"
        spinner_slot = st.empty()
        clicked = st.button(
            button_label,
            key=button_key,
            disabled=(not inputs_valid) or st.session_state.get(_KEY_CAPTURE_PROG, False),
        )
        if clicked and not st.session_state.get(_KEY_CAPTURE_PROG, False):
            st.session_state[_KEY_CAPTURE_PROG] = True
            params = {
                "sample_number": sample_number, "stage_number": stage_number,
                "trial_number": trial_number,
                "motor_ip": motor_ip, "motor_port": motor_port,
            }
            if not is_grading:
                params["grades_input"] = grades_input
            st.session_state[_KEY_PENDING_CAPTURE] = {"mode": mode, "params": params}
            st.rerun()

        if st.session_state.get(_KEY_PENDING_CAPTURE):
            with spinner_slot:
                with st.spinner(t("image_capture_spinner")):
                    p = st.session_state[_KEY_PENDING_CAPTURE]
                    capture_callback(p["mode"], p["params"])
            st.session_state[_KEY_PENDING_CAPTURE] = None
            st.rerun()

        if st.session_state.get(_KEY_CAPTURE_ERROR):   st.error(st.session_state[_KEY_CAPTURE_ERROR])
        if st.session_state.get(_KEY_CAPTURE_SUCCESS): st.success(st.session_state[_KEY_CAPTURE_SUCCESS])
        if st.session_state.get(_KEY_DIFF_SUCCESS):    st.success(st.session_state[_KEY_DIFF_SUCCESS])
        if st.session_state.get(_KEY_DIFF_ERROR):      st.error(st.session_state[_KEY_DIFF_ERROR])
        if st.session_state.get(_KEY_HIST_SUCCESS):    st.success(st.session_state[_KEY_HIST_SUCCESS])
        if st.session_state.get(_KEY_HIST_ERROR):      st.error(st.session_state[_KEY_HIST_ERROR])

    extra = load_weight if is_grading else grades_input
    return sample_number, stage_number, trial_number, extra, inputs_valid, all_valid


def _render_grading_results_column(col, sample_number, stage_number, trial_number, load_weight, inputs_valid):
    """Render results + export + status column for grading mode."""
    with col:
        st.subheader(t("results"))
        if st.session_state.get(_KEY_GRADE) is not None:
            display_grades(st.session_state[_KEY_GRADE], sample_number, stage_number)

        st.button(t("export_results"), key="btn_export_results",
                  on_click=export_results_callback, args=(sample_number, stage_number, load_weight, trial_number))

        if st.session_state.get(_KEY_EXPORT_SUCCESS):
            msg_col, btn_col = st.columns([0.7, 0.3])
            with msg_col:
                st.success(st.session_state[_KEY_EXPORT_SUCCESS])
            with btn_col:
                st.button("Open PDF", key="btn_open_pdf", on_click=open_pdf_callback)
        if st.session_state.get(_KEY_EXPORT_ERROR):
            st.error(st.session_state[_KEY_EXPORT_ERROR])

        st.button(t("reset_button"), key="btn_reset_grade", on_click=reset_grade_callback)

        st.markdown("---")
        st.subheader(t("status"))
        if inputs_valid:
            display_operation_status(sample_number, stage_number, trial_number, cfg.SUFFIX_GRADING)
        st.markdown("---")


def _render_training_status_column(col, sample_number, stage_number, trial_number, inputs_valid):
    """Render status column for training mode."""
    with col:
        st.subheader(t("status"))
        if inputs_valid:
            display_operation_status(sample_number, stage_number, trial_number, cfg.SUFFIX_TRAINING)
        st.markdown("---")


def _render_mode_screen(mode, title_key, back_key):
    """Shared skeleton for both grading and training screens."""
    _render_logo()
    st.markdown("<h1 style='text-align: center;'>" + t(title_key) + "</h1>", unsafe_allow_html=True)
    if can_go_back():
        spec = [3, 3, 1] if mode == "grading" else [1, 1, 1]
        _, _, col = st.columns(spec)
        with col:
            st.button("← Back", key=back_key, on_click=go_back_callback)
    st.markdown("---")

    motor_ip, motor_port, _base_dir = setup_sidebar()
    show_image_viewer()
    st.markdown("---")

    selected_grades = None if mode == "grading" else st.session_state.get("selected_grades", list(cfg.GRADES))
    col1, _, col2 = st.columns([2, 0.5, 2])
    sample, stage, trial, extra, inputs_valid, _ = _render_capture_column(col1, mode, motor_ip, motor_port, selected_grades)
    if mode == "grading":
        _render_grading_results_column(col2, sample, stage, trial, extra, inputs_valid)
    else:
        _render_training_status_column(col2, sample, stage, trial, inputs_valid)


def show_grading_mode():
    _render_mode_screen("grading", "title_grading", "btn_back_grading")


def show_training_mode():
    _render_mode_screen("training", "title_training", "btn_back_training")


def capture_images_action(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port):
    try:
        print(f"Starting image capture for sample {sample_number}, stage {stage_number}, trial {trial_number}")
        return capture_sample_images(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port)
    except Exception as e:
        print(f"Error in capture_images_action: {e}")
        return False


def create_difference_action(sample_number, stage_number, trial_number, suffix, reference_stage):
    try:
        print(f"Creating difference images for sample {sample_number}, stage {stage_number}, trial {trial_number} vs reference {reference_stage}")
        return create_difference_images(sample_number, stage_number, trial_number, suffix, reference_stage)
    except Exception as e:
        print(f"Error in create_difference_action: {e}")
        return False


def run_grading_analysis(sample_number, stage_number, trial_number, selected_grades=None):
    """Run grade prediction from difference images. Returns grades dict or None."""
    try:
        print(f"Analyzing histograms for sample {sample_number}, stage {stage_number}, trial {trial_number}")
        grades = analyze_difference_images_and_predict_output(
            sample_number, stage_number, trial_number,
            selected_grades=selected_grades,
        )
        if grades:
            print(f"Grade prediction completed: {grades}")
            return grades
        print("Grade prediction returned no results.")
        return None
    except Exception as e:
        print(f"Error in run_grading_analysis: {e}")
        return None


def run_training_analysis(sample_number, stage_number, trial_number, grades_dict):
    """Write training CSVs for the provided grades dict."""
    try:
        print(f"Writing training features for sample {sample_number}, stage {stage_number}, trial {trial_number}")
        analyze_difference_images(sample_number, stage_number, trial_number, grades=grades_dict)
    except Exception as e:
        print(f"Error in run_training_analysis: {e}")


def display_operation_status(sample_number, stage_number, trial_number, suffix):
    st.subheader(f"{t('sample_number')} {sample_number}, {t('stage_number')} {stage_number}, {t('trial_number')} {trial_number}")

    image_status = cached_check_required_images(sample_number, stage_number, trial_number, suffix, st.session_state.fs_epoch)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric(t("available_samples"), f"{len(image_status['present_input_images'])}/8")
        if image_status['all_input_present']:
            st.success(t("valid_input").replace("✅ Input parameters valid", "✅ All input images present"))

    with col2:
        st.metric(t("results"), f"{len(image_status['present_difference_images'])}/8")
        if image_status['all_difference_present']:
            st.success(t("valid_input").replace("✅ Input parameters valid", "✅ All difference images present"))

    # Display 8 input images in 2 rows (4 per row)
    st.markdown(f"#### {t('view_images')}")
    input_images = image_status['present_input_images']
    input_dir = cfg.get_input_dir(suffix)
    for i in range(0, len(input_images), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(input_images):
                image_path = os.path.join(input_dir, input_images[i + j])
                if os.path.exists(image_path):
                    sig = _file_sig(image_path)
                    with cols[j]:
                        st.image(load_image_bytes(image_path, sig), caption=input_images[i + j], use_column_width=True)


def display_grades(grades, sample_number, stage_number):
    if not grades:
        st.warning("No analysis results to display")
        return

    selected = st.session_state.get("selected_grades", list(cfg.GRADES))
    active = [g for g in cfg.GRADES if g in selected and g in grades]
    if not active:
        st.warning("No grade predictions available for the selected grades.")
        return

    grade_label_keys = {
        "pilling": "pilling_grade_result",
        "matting": "matting_grade_result",
        "fuzzing": "fuzzing_grade_result",
    }
    st.markdown(f"**{t('sample_number')} {sample_number} — {t('stage_number')} {stage_number}**")
    cols = st.columns(len(active))
    for col, grade in zip(cols, active):
        with col:
            st.metric(t(grade_label_keys[grade]), grades.get(grade, "—"))


# ---- Navigation helpers ----
def _ensure_nav_stack():
    if "nav_stack" not in st.session_state:
        st.session_state.nav_stack = []

def navigate_to(mode):
    _ensure_nav_stack()
    # push current mode to stack (even None, so Back returns to selector)
    current = st.session_state.get("mode", None)
    st.session_state.nav_stack.append(current)
    st.session_state.mode = mode

def can_go_back():
    return bool(st.session_state.get("nav_stack"))

def render_back_button():
    if can_go_back():
        st.button("← Back", key="btn_back", on_click=go_back_callback)
# ---- end helpers ----

def show_mode_selector():
    _render_logo()
    st.title(t("mode_selector"))
    col1, col2 = st.columns(2)
    with col1:
        st.button(t("training_mode"), key="btn_training_mode", on_click=navigate_to_training_callback)
    with col2:
        st.button(t("grading_mode"), key="btn_grading_mode", on_click=navigate_to_grading_callback)

def handle_training_login():
    _render_logo()
    st.subheader(t("login_required"))
    if can_go_back():
        _, _, col = st.columns([3, 3, 1])
        with col:
            st.button("← Back", key="btn_back_login", on_click=go_back_callback)
    password = st.text_input(t("password_prompt"), type="password", key="k_admin_pwd")
    st.button(t("submit"), key="btn_submit_login", on_click=submit_login_callback, args=(password,))
    if st.session_state.get(_KEY_LOGIN_ERROR):
        st.error(st.session_state[_KEY_LOGIN_ERROR])

def handle_operator_name():
    _render_logo()
    if can_go_back():
        _, _, col = st.columns([3, 3, 1])
        with col:
            st.button("← Back", key="btn_back_operator", on_click=go_back_callback)
    st.subheader(t("operator_prompt"))
    name = st.text_input(t("operator_prompt"), key="k_operator_name")
    st.button(t("submit"), key="btn_submit_operator", on_click=submit_operator_callback, args=(name,))
    if st.session_state.get(_KEY_OPERATOR_ERROR):
        st.warning(st.session_state[_KEY_OPERATOR_ERROR])

# ---------------------- Main ---------------------- #

def main():
    if "mode" not in st.session_state:
        st.session_state.mode = None
    if "nav_stack" not in st.session_state:
        st.session_state.nav_stack = []
    if _KEY_GRADE not in st.session_state:
        st.session_state[_KEY_GRADE] = None
    if "fs_epoch" not in st.session_state:
        st.session_state.fs_epoch = 0  # cache-busting counter for filesystem changes
    # Ensure progress flag exists (default False) to safely use in UI disabled conditions
    if _KEY_CAPTURE_PROG not in st.session_state:
        st.session_state[_KEY_CAPTURE_PROG] = False

    # # Global Back button (shown if there is a previous screen)
    # render_back_button()

    if st.session_state.mode is None:
        show_mode_selector()

    elif st.session_state.mode == "training_requested":
        handle_training_login()

    elif st.session_state.mode == "grading_requested":
        handle_operator_name()

    elif st.session_state.mode == "training":
        show_training_mode()

    elif st.session_state.mode == "grading":
        show_grading_mode()

if __name__ == "__main__":
    main()
