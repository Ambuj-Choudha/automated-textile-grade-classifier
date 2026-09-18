import streamlit as st

from app.reporting.translations import t
from app import state as S
from app.callbacks import can_go_back, go_back_callback, submit_login_callback
from app.ui.components import render_logo


def handle_training_login() -> None:
    render_logo()
    st.subheader(t("login_required"))
    if can_go_back():
        _, _, col = st.columns([3, 3, 1])
        with col:
            st.button("← Back", key="btn_back_login", on_click=go_back_callback)
    password = st.text_input(t("password_prompt"), type="password", key="k_admin_pwd")
    st.button(t("submit"), key="btn_submit_login", on_click=submit_login_callback, args=(password,))
    if st.session_state.get(S.KEY_LOGIN_ERROR):
        st.error(st.session_state[S.KEY_LOGIN_ERROR])
