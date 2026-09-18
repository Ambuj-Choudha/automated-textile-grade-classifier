"""Session-state keys and startup helpers for the TexIQ UI.

All session-state key strings used across the app live here — one place to
audit or add to when a new piece of UI state appears.
"""
import streamlit as st


# ---- Session key constants ----

KEY_MODE = "mode"
KEY_NAV_STACK = "nav_stack"
KEY_FS_EPOCH = "fs_epoch"
KEY_GRADE = "grade"
KEY_CAPTURE_PROG = "capture_in_progress"
KEY_PENDING_CAPTURE = "pending_capture"
KEY_CAPTURE_ERROR = "capture_error"
KEY_CAPTURE_SUCCESS = "capture_success"
KEY_DIFF_SUCCESS = "diff_success"
KEY_DIFF_ERROR = "diff_error"
KEY_HIST_SUCCESS = "histogram_success"
KEY_HIST_ERROR = "histogram_error"
KEY_EXPORT_SUCCESS = "export_success"
KEY_EXPORT_ERROR = "export_error"
KEY_EXPORT_PDF = "export_pdf_path"
KEY_LOGIN_ERROR = "login_error"
KEY_OPERATOR_ERROR = "operator_error"

# Pipeline status messages — cleared as a group between runs.
STATUS_KEYS = (
    KEY_CAPTURE_ERROR, KEY_CAPTURE_SUCCESS,
    KEY_DIFF_ERROR, KEY_DIFF_SUCCESS,
    KEY_HIST_ERROR, KEY_HIST_SUCCESS,
)

# Dev/runtime flag. Step 3 will move this behind the admin panel.
AUTO_CLEAR_CACHE_ON_START = True


def init_session_state() -> None:
    """Populate any missing session-state keys with safe defaults."""
    defaults = {
        KEY_MODE: None,
        KEY_NAV_STACK: [],
        KEY_GRADE: None,
        KEY_FS_EPOCH: 0,
        KEY_CAPTURE_PROG: False,
        "lang": "en",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def bootstrap_once() -> None:
    """One-time per-server-process cache reset.

    Why: caches held across a restart can hand out stale directory listings
    when data/*/ has been edited off-app.
    """
    if "bootstrapped" in st.session_state:
        return
    if AUTO_CLEAR_CACHE_ON_START:
        try:
            st.cache_data.clear()
            st.session_state["_cache_cleared_on_boot"] = True
        except Exception:
            pass
    else:
        st.session_state["_cache_cleared_on_boot"] = False
    st.session_state.bootstrapped = True


def inject_css(path: str = "static/style.css") -> None:
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
