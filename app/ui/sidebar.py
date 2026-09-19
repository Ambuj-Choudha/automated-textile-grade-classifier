"""Sidebar rendering.

Two entry points:
  - render_persistent_sidebar(): language + App Control. Shown on every screen.
  - setup_sidebar(): grades + motor settings. Shown only on capture screens.
"""
import streamlit as st

from app.reporting.translations import t
from app.settings import config as cfg
from app.callbacks import restart_app_callback
from app import state as S


LANGUAGE_OPTIONS = {
    "English": "en",
    "Deutsch": "de",
    "Français": "fr",
    "Español": "es",
}


def render_persistent_sidebar() -> None:
    """Language selector + App Control — reachable from every screen."""
    st.sidebar.markdown("### 🌐 Language")
    selected_language = st.sidebar.selectbox(
        "Select language:",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=list(LANGUAGE_OPTIONS.values()).index(st.session_state[S.KEY_LANG]),
        key="lang_select",
    )
    st.session_state[S.KEY_LANG] = LANGUAGE_OPTIONS[selected_language]

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### ⚙️ App Control")
        st.button("🔄 Restart App", key="btn_restart_app", on_click=restart_app_callback)


def setup_sidebar(mode: str = "grading"):
    """Settings section shown on capture screens. Returns (motor_ip, motor_port).

    In grading mode, grades without trained weights render as disabled
    checkboxes with an explanatory tooltip — recall would crash otherwise.
    In training mode all grades stay selectable (training is how you get
    them into the trained set).
    """
    with st.sidebar:
        st.header(t("settings"))

        st.subheader(t("select_grades"))
        if S.KEY_SELECTED_GRADES not in st.session_state:
            st.session_state[S.KEY_SELECTED_GRADES] = list(cfg.GRADES)

        if mode == "grading":
            _render_grading_grade_checkboxes()
        else:
            _render_training_grade_multiselect()

        if not st.session_state.get(S.KEY_SELECTED_GRADES):
            st.warning(t("grades_required"))

        st.subheader(t("motor_settings"))
        default_motor_ip = st.session_state.get("last_motor_ip", "")
        default_motor_port = st.session_state.get("last_motor_port", 18812)
        motor_ip = st.text_input(t("motor_ip"), value=default_motor_ip, key="k_motor_ip")
        motor_port = int(st.number_input(t("motor_port"), value=default_motor_port, key="k_motor_port"))
        st.session_state.last_motor_ip = motor_ip
        st.session_state.last_motor_port = motor_port

        st.markdown("---")

    return motor_ip, motor_port


def _render_grading_grade_checkboxes() -> None:
    """One checkbox per grade; disabled when the grade's network isn't trained.

    Writes the selection back into KEY_SELECTED_GRADES so downstream code
    that reads that key (pipeline.run_capture_pipeline) sees the picks.
    """
    trained = set(cfg.get_trained_grades())
    previously_selected = set(st.session_state.get(S.KEY_SELECTED_GRADES, []))
    selected: list = []

    for grade in cfg.GRADES:
        is_trained = grade in trained
        default = grade in previously_selected and is_trained
        checked = st.checkbox(
            grade.capitalize(),
            value=default,
            key=f"grade_cb_{grade}",
            disabled=not is_trained,
            help=None if is_trained else t("grade_not_trained"),
        )
        if checked and is_trained:
            selected.append(grade)

    st.session_state[S.KEY_SELECTED_GRADES] = selected


def _render_training_grade_multiselect() -> None:
    """Multiselect over all three grades — training doesn't gate on trained state."""
    st.multiselect(
        t("select_grades"),
        options=list(cfg.GRADES),
        key=S.KEY_SELECTED_GRADES,
        format_func=str.capitalize,
        label_visibility="collapsed",
    )
