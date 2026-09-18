import streamlit as st

from app.reporting.translations import t
from app.callbacks import (
    navigate_to_training_callback, navigate_to_grading_callback,
)
from app.ui.components import render_logo


def show_mode_selector() -> None:
    render_logo()
    st.title(t("mode_selector"))
    col1, col2 = st.columns(2)
    with col1:
        st.button(t("training_mode"), key="btn_training_mode", on_click=navigate_to_training_callback)
    with col2:
        st.button(t("grading_mode"), key="btn_grading_mode", on_click=navigate_to_grading_callback)
