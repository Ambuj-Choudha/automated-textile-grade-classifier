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
from app.helpers.utils import validate_input_number, validate_grade, get_available_samples, check_required_images, export_results
from app.reporting.translations import translations
from app.settings import config as cfg

# ------------------------------------------------------------------
# Dev / runtime flags
AUTO_CLEAR_CACHE_ON_START = True   # Set False in production to keep caches between restarts
# ------------------------------------------------------------------

# Streamlit page configuration: title, layout, and sidebar state
st.set_page_config(
    page_title="TexIQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS Styling
st.markdown("""
<style>
  :root{
    --bg: #001f3f;          /* background */
    --fg: #f7f9fc;          /* light text */
    --card: #ffffff;

    --primary: #00B8F0;     /* TexIQ cyan */
    --primary-600: #05A3D6; /* hover */
    --primary-700: #0689B5; /* pressed */

    --accent: #FFD33D;      /* TexIQ yellow */
    --danger: #E23B3B;      /* TexIQ red */
    --muted: #6b7280;

    --sidebar-bg: #2b2f33;  /* charcoal */
    --sidebar-fg: #f7f9fc;  /* light text */
  }

  /* 1. Base */
  body, section.main {
    background-color: var(--bg);
    color: var(--fg);
    font-family: 'Segoe UI', sans-serif;
    padding: 2rem;
    border-radius: 12px;
  }

  /* ENFORCE CONSISTENT HEADING COLORS ACROSS THEMES */
  h1, h2, h3, h4, h5, h6,
  .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
  .block-container h1, .block-container h2, .block-container h3 {
    color: var(--fg) !important;
  }

  /* Make Streamlit header/top bar match app background */
  [data-testid="stHeader"]{
    background: var(--bg) !important;
    color: var(--fg) !important;
    box-shadow: none !important;
  }
  [data-testid="stHeader"] *{
    color: var(--fg) !important;
  }

  /* 2. Sidebar */
  .css-1d391kg, [data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
  }
  .css-1d391kg .css-qbe2hs, [data-testid="stSidebar"] * {
    color: var(--sidebar-fg) !important;
  }

  /* 3. Buttons */
  .stButton > button {
    width: 100%;
    margin: 8px 0;
    background: var(--primary);
    color: #0b2536;
    font-weight: 700;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    border: none;
    box-shadow: 0 2px 10px rgba(0, 184, 240, 0.2);
    transition: background 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
  }
  .stButton > button:hover {
    background: var(--primary-600);
    color: #ffffff;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(0, 184, 240, 0.28);
  }
  .stButton > button:active {
    background: var(--primary-700);
    transform: translateY(0);
  }

  /* 4. Custom Alert Boxes (mapped to logo colors) */
  .success-box, .error-box, .info-box {
    padding: 14px 20px;
    border-radius: 10px;
    margin: 16px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    color: var(--fg);
    opacity: 0;
    transform: translateY(10px);
    animation: fadeInUp 0.6s ease forwards;
    background: var(--card);
  }
  .success-box {
    background: rgba(0, 184, 240, 0.08);
    border-left: 6px solid var(--primary);
  }
  .error-box {
    background: rgba(226, 59, 59, 0.10);
    border-left: 6px solid var(--danger);
  }
  .info-box {
    background: rgba(255, 211, 61, 0.12);
    border-left: 6px solid var(--accent);
  }
  .success-box:hover, .error-box:hover, .info-box:hover {
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    transform: translateY(5px);
    transition: all 0.3s ease;
  }

  /* 5. Animations */
  @keyframes fadeInUp { to { opacity: 1; transform: translateY(0); } }

  /* 6. Images */
  .stImage > img {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }

  /* 7. Links and small accents */
  a, .stMarkdown a { color: var(--primary); }
  hr, .stMarkdown hr { border-color: rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)

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

# Translation function
def t(key):
    return translations[st.session_state.lang].get(key, key)

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

def capture_grading_callback(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port):
    """Callback for capture button in grading mode"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ref_dir = os.path.join(script_dir, cfg.get_reference_dir(suffix))
        
        # Check for this trial's reference image (captured at stage 0)
        ref_filename = cfg.make_reference_filename(sample_number, "0", trial_number)
        ref_filepath = os.path.join(ref_dir, ref_filename)

        if stage_number != "0" and not os.path.exists(ref_filepath):
            st.session_state.capture_error = f"Reference image not found: {ref_filename}. Please capture stage 0 for trial {trial_number} first."
            return

        # Clear any previous error messages at the start
        st.session_state.capture_error = None
        st.session_state.capture_success = None
        st.session_state.diff_success = None
        st.session_state.diff_error = None
        st.session_state.histogram_success = None
        st.session_state.histogram_error = None

        # Mark in progress (UI disables button before this via pending flag)
        st.session_state.capture_in_progress = True

        # Perform capture (spinner shown in UI)
        capture_success = capture_images_action(sample_number, stage_number, trial_number, "for_grading", motor_ip, motor_port)

        if not capture_success:
            st.session_state.capture_error = t("image_capture_fail")
            return
        else:
            st.session_state.capture_success = t("image_capture_success")
            st.session_state.fs_epoch += 1  # bust directory-related caches

        if stage_number != "0":
            diff_success = create_difference_action(sample_number, stage_number, trial_number, "for_grading", reference_stage="0")
            if diff_success:
                st.session_state.diff_success = t("diff_create_success")
                st.session_state.fs_epoch += 1  # new diff files -> bust again
            else:
                st.session_state.diff_error = t("diff_create_fail")
                return

            grade = analyze_histograms_action(sample_number, stage_number, trial_number, 0)
            if grade is not None:
                st.session_state['Grade'] = grade
                st.session_state.histogram_success = f"{t('histogram_success')} {grade:.2f}"
            else:
                st.session_state.histogram_error = t("histogram_fail")

    except Exception as e:
        st.session_state.capture_error = f"Unexpected error: {str(e)}"
        print(f"Error in capture_grading_callback: {e}")
    finally:
        # Always clear in-progress flag (covers early returns and exceptions)
        st.session_state.capture_in_progress = False

def capture_training_callback(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port, grade_number):
    """Callback for capture button in training mode"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ref_dir = os.path.join(script_dir, cfg.get_reference_dir(suffix))
        
        # Check for this trial's reference image (captured at stage 0)
        ref_filename = cfg.make_reference_filename(sample_number, "0", trial_number)
        ref_filepath = os.path.join(ref_dir, ref_filename)

        if stage_number != "0" and not os.path.exists(ref_filepath):
            st.session_state.capture_error_training = f"Reference image not found: {ref_filename}. Please capture stage 0 for trial {trial_number} first."
            return

        # Clear any previous errors
        st.session_state.capture_error_training = None

        # Mark in progress (UI disables button before this via pending flag)
        st.session_state.capture_in_progress_training = True

        # Perform capture (spinner shown in UI)
        capture_success = capture_images_action(sample_number, stage_number, trial_number, "for_training", motor_ip, motor_port)

        if not capture_success:
            st.session_state.capture_error_training = t("image_capture_fail")
            return
        else:
            st.session_state.capture_success_training = t("image_capture_success")
            st.session_state.fs_epoch += 1

        if stage_number != "0":
            with st.spinner(t("diff_create_spinner")):
                diff_success = create_difference_action(sample_number, stage_number, trial_number, "for_training", reference_stage="0")
                if diff_success:
                    st.session_state.diff_success_training = t("diff_create_success")
                    st.session_state.fs_epoch += 1
                else:
                    st.session_state.diff_error_training = t("diff_create_fail")
                    return

            with st.spinner(t("histogram_spinner")):
                analyze_histograms_action(sample_number, stage_number, trial_number, grade_number)
                st.session_state.histogram_success_training = t("histogram_success")

    except Exception as e:
        st.session_state.capture_error_training = f"Unexpected error: {str(e)}"
        print(f"Error in capture_training_callback: {e}")
    finally:
        # Always clear in-progress flag
        st.session_state.capture_in_progress_training = False

def export_results_callback(sample_number, stage_number, load_weight, trial_number="1"):
    """Callback for export results button"""
    # Clear previous messages first
    for key in ['export_success', 'export_error']:
        st.session_state.pop(key, None)
    
    operator = st.session_state.get('operator_name', '')
    # Clean operator name for display message
    clean_operator_name = "".join(c for c in operator if c.isalnum() or c in ('-', '_')).strip()
    if not clean_operator_name:
        clean_operator_name = "Unknown"

    # Precompute expected PDF path
    pdf_filepath = os.path.join("reports", f"{sample_number}-{clean_operator_name}-report.pdf")

    try:
        # Check if analysis was completed first
        if 'Grade' not in st.session_state or st.session_state['Grade'] is None:
            st.session_state.export_error = "Cannot export: No analysis results available. Please complete the grading process first."
            return
        
        # Attempt the export with trial_number
        success = export_results(sample_number, stage_number, load_weight, operator, trial_number=trial_number)
        
        if success:
            st.session_state.export_success = f"PDF report saved to reports/{sample_number}-{clean_operator_name}-report.pdf!"
            st.session_state.export_pdf_path = pdf_filepath  # store for Open PDF button
        else:
            # Check if there are any analysis files for this sample
            dirpath = os.path.join("output", "grading_results")
            matches = []
            if os.path.isdir(dirpath):
                matches = [f for f in os.listdir(dirpath) if f.startswith(f"{sample_number}-") and f.endswith("-analysis.csv")]
            if not matches:
                st.session_state.export_error = f"Cannot export: No analysis files for sample '{sample_number}' found. Please complete image capture and analysis first."
            else:
                st.session_state.export_error = "Export failed: Error processing analysis data."
            
    except Exception as e:
        st.session_state.export_error = f"Export failed: {str(e)}"
        print(f"Export error details: {e}")

def open_pdf_callback():
    """Open the exported PDF in the default system viewer."""
    try:
        path = st.session_state.get("export_pdf_path")
        if not path:
            st.session_state.export_error = "No PDF available to open."
            return
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            st.session_state.export_error = "PDF not found on disk."
            return

        if sys.platform.startswith("win"):
            os.startfile(abs_path)  # Windows default viewer
        elif sys.platform.startswith("linux"):
            import subprocess
            subprocess.run(["xdg-open", abs_path], check=False)
        else:
            st.session_state.export_error = "Open PDF is supported only on Windows and Linux."
    except Exception as e:
        st.session_state.export_error = f"Failed to open PDF: {e}"

def reset_grade_callback():
    """Callback for reset grade button"""
    st.session_state.pop('Grade', None)
    # Clear any messages
    for key in ['capture_success', 'capture_error', 'diff_success', 'diff_error', 
                'histogram_success', 'histogram_error', 'export_success', 'export_error', 'export_pdf_path']:
        st.session_state.pop(key, None)

def navigate_to_training_callback():
    """Callback for training mode button"""
    navigate_to("training_requested")

def navigate_to_grading_callback():
    """Callback for grading mode button"""
    navigate_to("grading_requested")

def submit_login_callback(password):
    """Callback for login submit button"""
    if password == ADMIN_PASSWORD:
        navigate_to("training")
        st.session_state.login_error = None
    else:
        st.session_state.login_error = t("login_error")

def submit_operator_callback(name):
    """Callback for operator submit button"""
    if name.strip():
        st.session_state.operator_name = name.strip()
        navigate_to("grading")
        st.session_state.operator_error = None
    else:
        st.session_state.operator_error = t("name_required")

def go_back_callback():
    """Callback for back button"""
    stack = st.session_state.get("nav_stack", [])
    if stack:
        st.session_state.mode = stack.pop()

def clear_training_messages_callback():
    """Callback to clear training messages"""
    for key in ['capture_success_training', 'capture_error_training', 'diff_success_training', 
                'diff_error_training', 'histogram_success_training']:
        st.session_state.pop(key, None)

def clear_grading_messages_callback():
    """Callback to clear grading messages"""
    for key in ['capture_success', 'capture_error', 'diff_success', 'diff_error', 
                'histogram_success', 'histogram_error']:
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
        'data/input_pictures/for_training', 'data/difference_pictures/for_training', 'data/reference_pictures/for_training',
        'data/input_pictures/for_grading', 'data/difference_pictures/for_grading', 'data/reference_pictures/for_grading'
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

def show_grading_mode():
    # Logo at the top, centered
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.image(os.path.join("logos", "Logo_TexIQ_v1.0.jpg"), width=200)
    
    # Centered title
    st.markdown("<h1 style='text-align: center;'>" + t("title_grading") + "</h1>", unsafe_allow_html=True)
    
    # Right-aligned back button below title
    if can_go_back():
        col1, col2, col3 = st.columns([3, 3, 1])
        with col3:
            st.button("← Back", key="btn_back_grading", on_click=go_back_callback)
    
    st.markdown("---")

    # Sidebar
    motor_ip, motor_port, base_dir = setup_sidebar()

    # Image Viewer Section
    show_image_viewer()
    st.markdown("---")

    col1, spacer, col2 = st.columns([2, 0.5, 2])

    # Column 1
    with col1:
        available_samples = cached_get_available_samples("for_prediction", st.session_state.fs_epoch)
        st.info(f"{t('available_samples')}: {len(available_samples)}")

        # Get last used values from session state (with fallbacks)
        default_sample = st.session_state.get("last_sample_number_g", "00000")
        default_stage = st.session_state.get("last_stage_number_g", "0")
        default_trial = st.session_state.get("last_trial_number_g", "1")
        default_load = st.session_state.get("last_load_weight_g", "0")

        # Create inputs with persistent defaults
        sample_number = st.text_input(t("sample_number"), value=default_sample, key="k_sample_number_g", help="e.g., 00001, 00002, etc.")
        stage_number = st.text_input(t("stage_number"), value=default_stage, key="k_stage_number_g", help="e.g., 125, 500, 1000, etc.")
        trial_number = st.text_input(t("trial_number"), value=default_trial, key="k_trial_number_g", help="e.g., 1, 2, 3, etc.")
        load_weight = st.text_input(t("load_weight"), value=default_load, key="k_load_weight_g", help="e.g., 155, 415, etc.")

        # Save current values back to session state
        st.session_state.last_sample_number_g = sample_number
        st.session_state.last_stage_number_g = stage_number
        st.session_state.last_trial_number_g = trial_number
        st.session_state.last_load_weight_g = load_weight

        sample_valid = validate_input_number(sample_number)
        stage_valid = validate_input_number(stage_number)
        trial_valid = validate_input_number(trial_number)
        load_valid = validate_input_number(load_weight)

        if not sample_valid:
            st.error(t("invalid_sample"))
        if not stage_valid:
            st.error(t("invalid_stage"))
        if not trial_valid:
            st.error(t("invalid_trial"))
        if not load_valid:
            st.error(t("invalid_load"))
        if sample_valid and stage_valid and trial_valid and load_valid:
            st.success(t("valid_input"))
            image_status = cached_check_required_images(sample_number, stage_number, trial_number, "for_grading", st.session_state.fs_epoch)
            if image_status['all_input_present']:
                st.info(f"📁 {len(image_status['present_input_images'])} {t('available_samples').lower()}")

        st.markdown("---")

        # Capture button with inline spinner placed near messages
        capture_spinner_placeholder = st.empty()
        clicked = st.button(
            t("capture_button"),
            key="btn_capture_grading",
            disabled=(not (sample_valid and stage_valid and trial_valid)) or st.session_state.get("capture_in_progress", False)
        )
        if clicked and not st.session_state.get("capture_in_progress", False):
            # Arm a pending task and rerun to immediately disable the button
            st.session_state.capture_in_progress = True
            st.session_state.pending_capture_grading = {
                "sample_number": sample_number,
                "stage_number": stage_number,
                "trial_number": trial_number,
                "suffix": "for_grading",
                "motor_ip": motor_ip,
                "motor_port": motor_port,
            }
            st.rerun()

        # If a grading capture is pending, run it under the spinner
        if st.session_state.get("pending_capture_grading"):
            with capture_spinner_placeholder:
                with st.spinner(t("image_capture_spinner")):
                    p = st.session_state.pending_capture_grading
                    capture_grading_callback(
                        p["sample_number"], p["stage_number"], p["trial_number"], p["suffix"], p["motor_ip"], p["motor_port"]
                    )
            # Clear pending and refresh UI (button will re-enable if not in progress)
            st.session_state.pending_capture_grading = None
            st.rerun()

        # Display capture messages
        if st.session_state.get('capture_error'):
            st.error(st.session_state.capture_error)
        if st.session_state.get('capture_success'):
            st.success(st.session_state.capture_success)
        if st.session_state.get('diff_success'):
            st.success(st.session_state.diff_success)
        if st.session_state.get('diff_error'):
            st.error(st.session_state.diff_error)
        if st.session_state.get('histogram_success'):
            st.success(st.session_state.histogram_success)
        if st.session_state.get('histogram_error'):
            st.error(st.session_state.histogram_error)

    # Column 2
    with col2:
        st.subheader(t("results"))
        if 'Grade' in st.session_state and st.session_state['Grade'] is not None:
            display_grades(st.session_state['Grade'], sample_number, stage_number)

        # Export results button with callback
        st.button(
            t("export_results"), 
            key="btn_export_results",
            on_click=export_results_callback,
            args=(sample_number, stage_number, load_weight, trial_number)
        )

        # Display export messages + Open PDF beside success
        if st.session_state.get('export_success'):
            msg_col, btn_col = st.columns([0.7, 0.3])
            with msg_col:
                st.success(st.session_state.export_success)
            with btn_col:
                st.button("Open PDF", key="btn_open_pdf", on_click=open_pdf_callback)
        if st.session_state.get('export_error'):
            st.error(st.session_state.export_error)

        # Reset button with callback
        st.button(
            t("reset_button"), 
            key="btn_reset_grade",
            on_click=reset_grade_callback
        )
        
        st.markdown("---")
        st.subheader(t("status"))
        if sample_valid and stage_valid and trial_valid:
            display_operation_status(sample_number, stage_number, trial_number, "for_grading")
        st.markdown("---")


def show_training_mode():
    # Logo at the top, centered
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.image(os.path.join("logos", "Logo_TexIQ_v1.0.jpg"), width=200)
    
    # Centered title
    st.markdown("<h1 style='text-align: center;'>" + t("title_training") + "</h1>", unsafe_allow_html=True)
    
    # Right-aligned back button below title
    if can_go_back():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            st.button("← Back", key="btn_back_training", on_click=go_back_callback)
    
    st.markdown("---")

    # Sidebar
    motor_ip, motor_port, base_dir = setup_sidebar()

    # Image Viewer Section
    show_image_viewer()
    st.markdown("---")

    col1, spacer, col2 = st.columns([2, 0.5, 2])

    # Column 1
    with col1:
        available_samples = cached_get_available_samples("for_dataset", st.session_state.fs_epoch)
        st.info(f"{t('available_samples')}: {len(available_samples)}")

        # Get last used values from session state (with fallbacks)
        default_sample = st.session_state.get("last_sample_number_t", "00000")
        default_stage = st.session_state.get("last_stage_number_t", "0")
        default_trial = st.session_state.get("last_trial_number_t", "1")
        default_grade = st.session_state.get("last_grade_number_t", "1")

        # Create inputs with persistent defaults
        sample_number = st.text_input(t("sample_number"), value=default_sample, key="k_sample_number_t", help="e.g., 00001, 00002, etc.")
        stage_number = st.text_input(t("stage_number"), value=default_stage, key="k_stage_number_t", help="e.g., 125, 500, 1000, etc.")
        trial_number = st.text_input(t("trial_number"), value=default_trial, key="k_trial_number_t", help="e.g., 1, 2, 3, etc.")
        grade_number = st.text_input(t("grade_number"), value=default_grade, key="k_grade_number_t", help="e.g., 1.5, 2, 2.5, etc.")

        # Save current values back to session state
        st.session_state.last_sample_number_t = sample_number
        st.session_state.last_stage_number_t = stage_number
        st.session_state.last_trial_number_t = trial_number
        st.session_state.last_grade_number_t = grade_number

        sample_valid = validate_input_number(sample_number)
        stage_valid = validate_input_number(stage_number)
        trial_valid = validate_input_number(trial_number)
        grade_valid = validate_grade(grade_number)

        if not sample_valid:
            st.error(t("invalid_sample"))
        if not stage_valid:
            st.error(t("invalid_stage"))
        if not trial_valid:
            st.error(t("invalid_trial"))
        if not grade_valid:
            st.error(t("invalid_grade"))
        if sample_valid and stage_valid and trial_valid and grade_valid:
            st.success(t("valid_input"))
            image_status = cached_check_required_images(sample_number, stage_number, trial_number, "for_training", st.session_state.fs_epoch)
            if image_status['all_input_present']:
                st.info(f"📁 {len(image_status['present_input_images'])} {t('available_samples').lower()}")

        st.markdown("---")

        # Capture button with inline spinner placed near messages
        capture_spinner_placeholder_t = st.empty()
        clicked_t = st.button(
            t("capture_button_training"),
            key="btn_capture_training",
            disabled=(not (sample_valid and stage_valid and trial_valid)) or st.session_state.get("capture_in_progress_training", False)
        )
        if clicked_t and not st.session_state.get("capture_in_progress_training", False):
            st.session_state.capture_in_progress_training = True
            st.session_state.pending_capture_training = {
                "sample_number": sample_number,
                "stage_number": stage_number,
                "trial_number": trial_number,
                "suffix": "for_training",
                "motor_ip": motor_ip,
                "motor_port": motor_port,
                "grade_number": grade_number,
            }
            st.rerun()

        if st.session_state.get("pending_capture_training"):
            with capture_spinner_placeholder_t:
                with st.spinner(t("image_capture_spinner")):
                    p = st.session_state.pending_capture_training
                    capture_training_callback(
                        p["sample_number"], p["stage_number"], p["trial_number"], p["suffix"], p["motor_ip"], p["motor_port"], p["grade_number"]
                    )
            st.session_state.pending_capture_training = None
            st.rerun()

        # Display training messages
        if st.session_state.get('capture_error_training'):
            st.error(st.session_state.capture_error_training)
        if st.session_state.get('capture_success_training'):
            st.success(st.session_state.capture_success_training)

    # Column 2
    with col2:
        st.subheader(t("status"))
        if sample_valid and stage_valid and trial_valid:
            display_operation_status(sample_number, stage_number, trial_number, "for_training")
        st.markdown("---")


def capture_images_action(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port):
    try:
        print(f"Starting image capture for sample {sample_number}, stage {stage_number}, trial {trial_number}")
        success = capture_sample_images(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port)
        return success
    except Exception as e:
        print(f"Error in capture_images_action: {e}")
        st.error(f"{t('status')}: {str(e)}")
        return False


def create_difference_action(sample_number, stage_number, trial_number, suffix, reference_stage):
    try:
        print(f"Creating difference images for sample {sample_number}, stage {stage_number}, trial {trial_number} vs reference {reference_stage}")
        success = create_difference_images(sample_number, stage_number, trial_number, suffix, reference_stage)
        return success
    except Exception as e:
        print(f"Error in create_difference_action: {e}")
        st.error(f"{t('status')}: {str(e)}")
        return False 


def analyze_histograms_action(sample_number, stage_number, trial_number, grade_number):
    try:
        print(f"Analyzing histograms for sample {sample_number}, stage {stage_number}, trial {trial_number}")
        if grade_number == 0:
            grade = analyze_difference_images_and_predict_output(sample_number, stage_number, trial_number)
            if grade:
                print("Histogram analysis completed successfully")
                return grade
            else:
                print("Histogram analysis failed or no data found")
                return None
        else:
            analyze_difference_images(sample_number, stage_number, trial_number, grade_number)

    except Exception as e:
        print(f"Error in analyze_histograms_action: {e}")
        st.error(f"{t('status')}: {str(e)}")
        return None


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

    # Display the reference image (if present)
    ref_images = image_status.get('reference_image', [])
    ref_dir = cfg.get_reference_dir(suffix)
    if ref_images:
        ref_path = os.path.join(ref_dir, ref_images[0])
        if os.path.exists(ref_path):
            st.image(load_image_bytes(ref_path, _file_sig(ref_path)), caption=ref_images[0], use_column_width=False, width=250)

def display_grades(grade, sample_number, stage_number):
    if grade is None:
        st.warning(t("histogram_fail").replace("❌ Histogram analysis failed!", "No analysis results to display"))
        return

    st.metric(
        f"{t('sample_number')} {sample_number}, {t('stage_number')} {stage_number} {t('results')}:",
        f"{grade}"
    )

ADMIN_PASSWORD = "1234"  # Change this to your actual secure password

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
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.image(os.path.join("logos", "Logo_TexIQ_v1.0.jpg"), width=200)
    
    st.title(t("mode_selector"))
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            t("training_mode"), 
            key="btn_training_mode",
            on_click=navigate_to_training_callback
        )
    with col2:
        st.button(
            t("grading_mode"), 
            key="btn_grading_mode",
            on_click=navigate_to_grading_callback
        )

def handle_training_login():
    # Logo at the top, centered
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.image(os.path.join("logos", "Logo_TexIQ_v1.0.jpg"), width=200)
    
    st.subheader(t("login_required"))
    
    # Right-aligned back button
    if can_go_back():
        col1, col2, col3 = st.columns([3, 3, 1])
        with col3:
            st.button("← Back", key="btn_back_login", on_click=go_back_callback)
    
    password = st.text_input(t("password_prompt"), type="password", key="k_admin_pwd")
    st.button(
        t("submit"), 
        key="btn_submit_login",
        on_click=submit_login_callback,
        args=(password,)
    )
    
    # Display login error if any
    if st.session_state.get('login_error'):
        st.error(st.session_state.login_error)

def handle_operator_name():
    # Logo at the top, centered
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.image(os.path.join("logos", "Logo_TexIQ_v1.0.jpg"), width=200)
    
    # Right-aligned back button
    if can_go_back():
        col1, col2, col3 = st.columns([3, 3, 1])
        with col3:
            st.button("← Back", key="btn_back_operator", on_click=go_back_callback)
    st.subheader(t("operator_prompt"))
    name = st.text_input(t("operator_prompt"), key="k_operator_name")
    st.button(
        t("submit"), 
        key="btn_submit_operator",
        on_click=submit_operator_callback,
        args=(name,)
    )
    
    # Display operator error if any
    if st.session_state.get('operator_error'):
        st.warning(st.session_state.operator_error)

# ---------------------- Main ---------------------- #

def main():
    if "mode" not in st.session_state:
        st.session_state.mode = None
    if "nav_stack" not in st.session_state:
        st.session_state.nav_stack = []
    if "Grade" not in st.session_state:
        st.session_state.Grade = None
    if "fs_epoch" not in st.session_state:
        st.session_state.fs_epoch = 0  # cache-busting counter for filesystem changes
    # Ensure progress flags exist (default False) to safely use in UI disabled conditions
    if "capture_in_progress" not in st.session_state:
        st.session_state.capture_in_progress = False
    if "capture_in_progress_training" not in st.session_state:
        st.session_state.capture_in_progress_training = False

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
