"""Small state-mutation callbacks used by buttons and forms across screens.

Grouped by concern:
  - Navigation stack (navigate_to, go_back, can_go_back)
  - Screen submits (login, operator)
  - Result actions (export, open PDF, reset grade, clear messages)
  - App control (restart)
"""
import logging
import os
import sys
import streamlit as st

from app.reporting.results import export_results
from app.reporting.translations import t
from app.settings import config as cfg
from app import state as S

log = logging.getLogger(__name__)


# ---- Navigation stack ----

def _ensure_nav_stack() -> None:
    if S.KEY_NAV_STACK not in st.session_state:
        st.session_state[S.KEY_NAV_STACK] = []


def navigate_to(mode) -> None:
    _ensure_nav_stack()
    current = st.session_state.get(S.KEY_MODE, None)
    st.session_state[S.KEY_NAV_STACK].append(current)
    st.session_state[S.KEY_MODE] = mode


def can_go_back() -> bool:
    return bool(st.session_state.get(S.KEY_NAV_STACK))


def go_back_callback() -> None:
    stack = st.session_state.get(S.KEY_NAV_STACK, [])
    if stack:
        st.session_state[S.KEY_MODE] = stack.pop()


def navigate_to_training_callback() -> None:
    navigate_to("training_requested")


def navigate_to_grading_callback() -> None:
    navigate_to("grading_requested")


# ---- Screen submits ----

def submit_login_callback(password) -> None:
    if password == cfg.ADMIN_PASSWORD:
        navigate_to("training")
        st.session_state[S.KEY_LOGIN_ERROR] = None
    else:
        st.session_state[S.KEY_LOGIN_ERROR] = t("login_error")


def submit_operator_callback(name) -> None:
    if name.strip():
        st.session_state["operator_name"] = name.strip()
        navigate_to("grading")
        st.session_state[S.KEY_OPERATOR_ERROR] = None
    else:
        st.session_state[S.KEY_OPERATOR_ERROR] = t("name_required")


# ---- Result actions ----

def export_results_callback(sample_number, stage_number, load_weight, trial_number="1") -> None:
    st.session_state.pop(S.KEY_EXPORT_SUCCESS, None)
    st.session_state.pop(S.KEY_EXPORT_ERROR, None)

    operator = st.session_state.get("operator_name", "")
    clean_operator_name = "".join(c for c in operator if c.isalnum() or c in ("-", "_")).strip() or "Unknown"
    pdf_filepath = os.path.join("reports", f"{sample_number}-{clean_operator_name}-report.pdf")

    try:
        grades = st.session_state.get(S.KEY_GRADE)
        if not grades:
            st.session_state[S.KEY_EXPORT_ERROR] = (
                "Cannot export: No analysis results available. "
                "Please complete the grading process first."
            )
            return

        success = export_results(sample_number, stage_number, load_weight, operator,
                                 trial_number=trial_number)

        if success:
            st.session_state[S.KEY_EXPORT_SUCCESS] = (
                f"PDF report saved to reports/{sample_number}-{clean_operator_name}-report.pdf!"
            )
            st.session_state[S.KEY_EXPORT_PDF] = pdf_filepath
        else:
            dirpath = os.path.join("output", "grading_results")
            matches = []
            if os.path.isdir(dirpath):
                matches = [f for f in os.listdir(dirpath)
                           if f.startswith(f"{sample_number}-") and f.endswith("-analysis.csv")]
            if not matches:
                st.session_state[S.KEY_EXPORT_ERROR] = (
                    f"Cannot export: No analysis files for sample '{sample_number}' found. "
                    "Please complete image capture and analysis first."
                )
            else:
                st.session_state[S.KEY_EXPORT_ERROR] = "Export failed: Error processing analysis data."
    except Exception as e:
        st.session_state[S.KEY_EXPORT_ERROR] = f"Export failed: {str(e)}"
        log.exception("export_results_callback failed")


def open_pdf_callback() -> None:
    """Open the exported PDF in the default system viewer."""
    try:
        path = st.session_state.get(S.KEY_EXPORT_PDF)
        if not path:
            st.session_state[S.KEY_EXPORT_ERROR] = "No PDF available to open."
            return
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            st.session_state[S.KEY_EXPORT_ERROR] = "PDF not found on disk."
            return
        if sys.platform.startswith("win"):
            os.startfile(abs_path)
        elif sys.platform.startswith("linux"):
            import subprocess
            subprocess.run(["xdg-open", abs_path], check=False)
        else:
            st.session_state[S.KEY_EXPORT_ERROR] = "Open PDF is supported only on Windows and Linux."
    except Exception as e:
        st.session_state[S.KEY_EXPORT_ERROR] = f"Failed to open PDF: {e}"


def reset_grade_callback() -> None:
    for key in (S.KEY_GRADE, S.KEY_CAPTURE_SUCCESS, S.KEY_CAPTURE_ERROR,
                S.KEY_DIFF_SUCCESS, S.KEY_DIFF_ERROR,
                S.KEY_HIST_SUCCESS, S.KEY_HIST_ERROR,
                S.KEY_EXPORT_SUCCESS, S.KEY_EXPORT_ERROR, S.KEY_EXPORT_PDF):
        st.session_state.pop(key, None)


def clear_messages_callback() -> None:
    for key in S.STATUS_KEYS:
        st.session_state.pop(key, None)


# ---- App control ----

def restart_app_callback() -> None:
    preserved = {"lang": st.session_state.get("lang", "en")}
    try:
        st.cache_data.clear()
    except Exception:
        pass
    for k in list(st.session_state.keys()):
        if k not in preserved:
            del st.session_state[k]
    st.session_state.update(preserved)
    st.session_state[S.KEY_FS_EPOCH] = 0
    st.session_state[S.KEY_NAV_STACK] = []
    st.session_state[S.KEY_MODE] = None
