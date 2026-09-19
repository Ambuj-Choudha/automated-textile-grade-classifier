"""The capture screen — shared skeleton for grading and training modes.

Both modes render:
  - Logo, title, optional Back button
  - Sidebar settings
  - Image viewer expander
  - Two-column body: input & capture (left), results/status (right)

They differ only in the mode-specific inputs (load_weight vs grade labels)
and in what the right-hand column shows (grading has results/export;
training shows only status).
"""
import streamlit as st

from app.helpers.utils import validate_input_number, validate_grade
from app.reporting.translations import t
from app.settings import config as cfg
from app import state as S
from app.ui.caching import cached_get_available_samples, cached_check_required_images
from app.callbacks import (
    can_go_back, go_back_callback,
    export_results_callback, open_pdf_callback, reset_grade_callback,
)
from app.ui.views import (
    render_logo, show_image_viewer, display_grades, display_operation_status,
)
from app.pipeline import capture_callback
from app.ui.sidebar import setup_sidebar


def _mode_suffix(mode: str) -> str:
    return cfg.SUFFIX_GRADING if mode == "grading" else cfg.SUFFIX_TRAINING


# Per-grade descriptor rows for the training input column.
# Each row: (grade_name, label_key, error_key, last_key, widget_key)
_TRAINING_GRADE_INPUTS = (
    ("pilling", "pilling_grade_number", "invalid_pilling_grade", "last_pilling_grade_number_t", "k_pilling_grade_number_t"),
    ("matting", "matting_grade_number", "invalid_matting_grade", "last_matting_grade_number_t", "k_matting_grade_number_t"),
    ("fuzzing", "fuzzing_grade_number", "invalid_fuzzing_grade", "last_fuzzing_grade_number_t", "k_fuzzing_grade_number_t"),
)


def _render_capture_button_and_flow(mode: str, params: dict, inputs_valid: bool) -> None:
    """Render the capture button + the two-phase click flow.

    Streamlit runs on_click callbacks BEFORE render, so a plain on_click can't
    paint a busy indicator during a multi-minute capture. Instead: the click
    stashes params + reruns; the next render paints the spinner and only then
    invokes capture_callback. Do not "simplify" back to a single on_click
    without validating the busy-indicator UX.
    """
    is_grading = mode == "grading"
    button_label = t("capture_button" if is_grading else "capture_button_training")
    button_key = "btn_capture_grading" if is_grading else "btn_capture_training"
    spinner_slot = st.empty()

    clicked = st.button(
        button_label, key=button_key,
        disabled=(not inputs_valid) or st.session_state.get(S.KEY_CAPTURE_PROG, False),
    )
    if clicked and not st.session_state.get(S.KEY_CAPTURE_PROG, False):
        st.session_state[S.KEY_CAPTURE_PROG] = True
        st.session_state[S.KEY_PENDING_CAPTURE] = {"mode": mode, "params": params}
        st.rerun()

    if st.session_state.get(S.KEY_PENDING_CAPTURE):
        with spinner_slot:
            with st.spinner(t("image_capture_spinner")):
                p = st.session_state[S.KEY_PENDING_CAPTURE]
                capture_callback(p["mode"], p["params"])
        st.session_state[S.KEY_PENDING_CAPTURE] = None
        st.rerun()


def _render_status_messages() -> None:
    """Render the six pipeline-status messages stashed in session_state."""
    if st.session_state.get(S.KEY_CAPTURE_ERROR):   st.error(st.session_state[S.KEY_CAPTURE_ERROR])
    if st.session_state.get(S.KEY_CAPTURE_SUCCESS): st.success(st.session_state[S.KEY_CAPTURE_SUCCESS])
    if st.session_state.get(S.KEY_DIFF_SUCCESS):    st.success(st.session_state[S.KEY_DIFF_SUCCESS])
    if st.session_state.get(S.KEY_DIFF_ERROR):      st.error(st.session_state[S.KEY_DIFF_ERROR])
    if st.session_state.get(S.KEY_HIST_SUCCESS):    st.success(st.session_state[S.KEY_HIST_SUCCESS])
    if st.session_state.get(S.KEY_HIST_ERROR):      st.error(st.session_state[S.KEY_HIST_ERROR])


def _render_capture_column(col, mode, motor_ip, motor_port, selected_grades=None):
    """Render the input + capture column for either grading or training.

    Returns (sample_number, stage_number, trial_number, extra, inputs_valid, all_valid).
    `extra` is the load_weight string (grading) or a grades_input dict (training).
    """
    is_grading = mode == "grading"
    suffix     = _mode_suffix(mode)
    tag        = "g" if is_grading else "t"

    with col:
        available = cached_get_available_samples(suffix, st.session_state[S.KEY_FS_EPOCH])
        st.info(f"{t('available_samples')}: {len(available)}")

        sample_number = st.text_input(t("sample_number"), value=st.session_state.get(f"last_sample_number_{tag}", "00000"), key=f"k_sample_number_{tag}", help="e.g., 00001, 00002, etc.") or ""
        stage_number  = st.text_input(t("stage_number"),  value=st.session_state.get(f"last_stage_number_{tag}",  "0"),     key=f"k_stage_number_{tag}",  help="e.g., 125, 500, 1000, etc.") or ""
        trial_number  = st.text_input(t("trial_number"),  value=st.session_state.get(f"last_trial_number_{tag}",  "1"),     key=f"k_trial_number_{tag}",  help="e.g., 1, 2, 3, etc.") or ""
        st.session_state[f"last_sample_number_{tag}"] = sample_number
        st.session_state[f"last_stage_number_{tag}"]  = stage_number
        st.session_state[f"last_trial_number_{tag}"]  = trial_number

        load_weight = None
        grades_input: dict = {}
        extra_valid = True

        if is_grading:
            load_weight = st.text_input(t("load_weight"), value=st.session_state.get("last_load_weight_g", "0"), key="k_load_weight_g", help="e.g., 155, 415, etc.") or ""
            st.session_state.last_load_weight_g = load_weight
            if not validate_input_number(load_weight):
                st.error(t("invalid_load"))
                extra_valid = False
        else:
            for name, label_key, err_key, last_key, widget_key in _TRAINING_GRADE_INPUTS:
                if name not in (selected_grades or ()):
                    continue
                val = st.text_input(t(label_key), value=st.session_state.get(last_key, "1"), key=widget_key, help="e.g., 1.5, 2, 2.5, etc.") or ""
                st.session_state[last_key] = val
                if not validate_grade(val):
                    st.error(t(err_key))
                    extra_valid = False
                else:
                    grades_input[name] = float(val)

        sample_valid = validate_input_number(sample_number)
        stage_valid  = validate_input_number(stage_number)
        trial_valid  = validate_input_number(trial_number)

        if not sample_valid: st.error(t("invalid_sample"))
        if not stage_valid:  st.error(t("invalid_stage"))
        if not trial_valid:  st.error(t("invalid_trial"))

        inputs_valid = sample_valid and stage_valid and trial_valid
        all_valid    = inputs_valid and extra_valid

        if all_valid:
            st.success(t("valid_input"))
            image_status = cached_check_required_images(sample_number, stage_number, trial_number, suffix, st.session_state[S.KEY_FS_EPOCH])
            if image_status['all_input_present']:
                st.info(f"📁 {len(image_status['present_input_images'])} {t('available_samples').lower()}")

        st.markdown("---")

        params = {
            "sample_number": sample_number, "stage_number": stage_number,
            "trial_number": trial_number,
            "motor_ip": motor_ip, "motor_port": motor_port,
        }
        if not is_grading:
            params["grades_input"] = grades_input
        _render_capture_button_and_flow(mode, params, inputs_valid)
        _render_status_messages()

    extra = load_weight if is_grading else grades_input
    return sample_number, stage_number, trial_number, extra, inputs_valid, all_valid


def _render_grading_results_column(col, sample_number, stage_number, trial_number, load_weight, inputs_valid) -> None:
    with col:
        st.subheader(t("results"))
        if st.session_state.get(S.KEY_GRADE) is not None:
            display_grades(st.session_state[S.KEY_GRADE], sample_number, stage_number)

        st.button(t("export_results"), key="btn_export_results",
                  on_click=export_results_callback,
                  args=(sample_number, stage_number, load_weight, trial_number))

        if st.session_state.get(S.KEY_EXPORT_SUCCESS):
            msg_col, btn_col = st.columns([0.7, 0.3])
            with msg_col:
                st.success(st.session_state[S.KEY_EXPORT_SUCCESS])
            with btn_col:
                st.button("Open PDF", key="btn_open_pdf", on_click=open_pdf_callback)
        if st.session_state.get(S.KEY_EXPORT_ERROR):
            st.error(st.session_state[S.KEY_EXPORT_ERROR])

        st.button(t("reset_button"), key="btn_reset_grade", on_click=reset_grade_callback)

        st.markdown("---")
        st.subheader(t("status"))
        if inputs_valid:
            display_operation_status(sample_number, stage_number, trial_number, cfg.SUFFIX_GRADING)
        st.markdown("---")


def _render_training_status_column(col, sample_number, stage_number, trial_number, inputs_valid) -> None:
    with col:
        st.subheader(t("status"))
        if inputs_valid:
            display_operation_status(sample_number, stage_number, trial_number, cfg.SUFFIX_TRAINING)
        st.markdown("---")


def _render_mode_screen(mode: str, title_key: str, back_key: str) -> None:
    render_logo()
    st.markdown("<h1 style='text-align: center;'>" + t(title_key) + "</h1>", unsafe_allow_html=True)
    if can_go_back():
        spec = [3, 3, 1] if mode == "grading" else [1, 1, 1]
        _, _, col = st.columns(spec)
        with col:
            st.button("← Back", key=back_key, on_click=go_back_callback)
    st.markdown("---")

    motor_ip, motor_port = setup_sidebar()
    show_image_viewer()
    st.markdown("---")

    selected_grades = None if mode == "grading" else st.session_state.get(S.KEY_SELECTED_GRADES, list(cfg.GRADES))
    col1, _, col2 = st.columns([2, 0.5, 2])
    sample, stage, trial, extra, inputs_valid, _ = _render_capture_column(col1, mode, motor_ip, motor_port, selected_grades)
    if mode == "grading":
        _render_grading_results_column(col2, sample, stage, trial, extra, inputs_valid)
    else:
        _render_training_status_column(col2, sample, stage, trial, inputs_valid)


def show_grading_mode() -> None:
    _render_mode_screen("grading", "title_grading", "btn_back_grading")


def show_training_mode() -> None:
    _render_mode_screen("training", "title_training", "btn_back_training")
