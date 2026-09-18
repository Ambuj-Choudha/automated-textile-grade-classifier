import streamlit as st

from app.reporting.translations import t
from app import state as S
from app.callbacks import can_go_back, go_back_callback, submit_operator_callback
from app.ui.components import render_logo


def handle_operator_name() -> None:
    render_logo()
    if can_go_back():
        _, _, col = st.columns([3, 3, 1])
        with col:
            st.button("← Back", key="btn_back_operator", on_click=go_back_callback)
    st.subheader(t("operator_prompt"))
    name = st.text_input(t("operator_prompt"), key="k_operator_name")
    st.button(t("submit"), key="btn_submit_operator", on_click=submit_operator_callback, args=(name,))
    if st.session_state.get(S.KEY_OPERATOR_ERROR):
        st.warning(st.session_state[S.KEY_OPERATOR_ERROR])
