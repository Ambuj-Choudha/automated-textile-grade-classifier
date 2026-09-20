"""TexIQ — Streamlit entry point.

Three responsibilities live here:
  1. Streamlit page config (must be the first Streamlit call).
  2. Boot-time CSS + cache reset.
  3. Screen dispatch based on st.session_state.mode.

Everything else lives under app/ui/. See docs/APP.md for the folder map.
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
from app.ui.screens import SCREEN_REGISTRY, show_mode_selector

# Re-exports referenced as app.NAME by app/tests/test_app.py.
from app.ui.caching import (  # noqa: F401
    cached_get_available_samples, cached_check_required_images,
)
from app.callbacks import (  # noqa: F401
    export_results_callback, reset_grade_callback,
    navigate_to_training_callback, navigate_to_grading_callback,
    submit_login_callback, submit_operator_callback, go_back_callback,
    clear_messages_callback, restart_app_callback,
)
from app.ui.views import (  # noqa: F401
    display_grades, display_operation_status,
)
from app.pipeline import (  # noqa: F401
    capture_images_action, create_difference_action,
    run_grading_analysis, run_training_analysis,
    capture_callback,
)

setup_logging()
state.inject_css()
state.bootstrap_once()


def main() -> None:
    state.init_session_state()
    render_persistent_sidebar()

    mode = st.session_state[state.KEY_MODE]
    screen_fn = SCREEN_REGISTRY.get(mode, show_mode_selector)
    screen_fn()


if __name__ == "__main__":
    main()
