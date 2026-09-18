"""TexIQ — Streamlit entry point.

Three responsibilities live here:
  1. Streamlit page config (must be the first Streamlit call).
  2. Boot-time CSS + cache reset.
  3. Screen dispatch based on st.session_state.mode.

Everything else lives under app/ui/. See docs/app/APP.md for the folder map.
"""
import streamlit as st

# Must precede every other Streamlit call, including anything a submodule
# import might trigger.
st.set_page_config(
    page_title="TexIQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app import state
from app.logging_setup import setup_logging
from app.ui.sidebar import render_persistent_sidebar
from app.ui.screens import (
    show_mode_selector,
    handle_training_login,
    handle_operator_name,
    show_grading_mode,
    show_training_mode,
)

# Public surface re-exports — kept stable for app/tests/test_app.py so that
# reorganizing implementation under app/ui/ doesn't force test churn.
from app.ui.caching import (  # noqa: F401
    cached_get_available_samples, cached_check_required_images, load_image_bytes,
)
from app.callbacks import (  # noqa: F401
    export_results_callback, open_pdf_callback, reset_grade_callback,
    navigate_to, navigate_to_training_callback, navigate_to_grading_callback,
    submit_login_callback, submit_operator_callback, go_back_callback,
    clear_messages_callback, restart_app_callback, can_go_back,
)
from app.ui.components import (  # noqa: F401
    display_grades, display_operation_status,
)
from app.pipeline import (  # noqa: F401
    capture_images_action, create_difference_action,
    run_grading_analysis, run_training_analysis,
    run_capture_pipeline, capture_callback,
)

setup_logging()
state.inject_css()
state.bootstrap_once()


def main() -> None:
    state.init_session_state()
    render_persistent_sidebar()

    mode = st.session_state[state.KEY_MODE]
    if mode is None:
        show_mode_selector()
    elif mode == "training_requested":
        handle_training_login()
    elif mode == "grading_requested":
        handle_operator_name()
    elif mode == "training":
        show_training_mode()
    elif mode == "grading":
        show_grading_mode()


if __name__ == "__main__":
    main()
