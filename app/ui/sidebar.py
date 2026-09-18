"""Sidebar rendering.

Two entry points:
  - render_persistent_sidebar(): language + App Control. Shown on every screen.
  - setup_sidebar(): grades / motor / base directory. Shown only on capture screens.
"""
import os
import streamlit as st

from app.reporting.translations import t
from app.settings import config as cfg
from app.callbacks import restart_app_callback


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
        index=list(LANGUAGE_OPTIONS.values()).index(st.session_state.lang),
        key="lang_select",
    )
    st.session_state.lang = LANGUAGE_OPTIONS[selected_language]

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### ⚙️ App Control")
        st.button("🔄 Restart App", key="btn_restart_app", on_click=restart_app_callback)


def setup_sidebar():
    """Settings section shown on capture screens. Returns (motor_ip, motor_port, base_dir)."""
    with st.sidebar:
        st.header(t("settings"))

        st.subheader(t("select_grades"))
        if "selected_grades" not in st.session_state:
            st.session_state.selected_grades = list(cfg.GRADES)
        selected_grades = st.multiselect(
            t("select_grades"),
            options=list(cfg.GRADES),
            key="selected_grades",
            format_func=str.capitalize,
            label_visibility="collapsed",
        )
        if not selected_grades:
            st.warning(t("grades_required"))

        st.subheader(t("motor_settings"))
        default_motor_ip = st.session_state.get("last_motor_ip", "")
        default_motor_port = st.session_state.get("last_motor_port", 18812)
        motor_ip = st.text_input(t("motor_ip"), value=default_motor_ip, key="k_motor_ip")
        motor_port = int(st.number_input(t("motor_port"), value=default_motor_port, key="k_motor_port"))
        st.session_state.last_motor_ip = motor_ip
        st.session_state.last_motor_port = motor_port

        st.subheader(t("directory_settings"))
        default_base_dir = st.session_state.get("last_base_dir", os.getcwd())
        base_dir = st.text_input(t("base_directory"), value=default_base_dir, key="k_base_dir")
        st.session_state.last_base_dir = base_dir

        st.markdown("---")

    return motor_ip, motor_port, base_dir
