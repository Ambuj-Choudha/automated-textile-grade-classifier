"""Shared capture → diff → analyze pipeline for grading and training.

The two modes only differ in the analysis step (predict vs write training
features) and in which extra inputs the user provides. Everything else —
staged capture, image differencing, error reporting — is the same.
"""
import logging
import os
import streamlit as st

from app.capture.camera_control import capture_sample_images
from app.capture.image_difference import create_difference_images
from app.analysis.histogram_analysis import (
    analyze_difference_images_and_predict_output,
    analyze_difference_images,
)
from app.reporting.translations import t
from app.settings import config as cfg
from app import state as S

log = logging.getLogger(__name__)


def _mode_suffix(mode: str) -> str:
    return cfg.SUFFIX_GRADING if mode == "grading" else cfg.SUFFIX_TRAINING


def capture_images_action(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port):
    try:
        log.info("Starting image capture for sample %s, stage %s, trial %s", sample_number, stage_number, trial_number)
        return capture_sample_images(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port)
    except Exception:
        log.exception("capture_images_action failed")
        return False


def create_difference_action(sample_number, stage_number, trial_number, suffix, reference_stage):
    try:
        log.info("Creating difference images for sample %s, stage %s, trial %s vs reference %s",
                 sample_number, stage_number, trial_number, reference_stage)
        return create_difference_images(sample_number, stage_number, trial_number, suffix, reference_stage)
    except Exception:
        log.exception("create_difference_action failed")
        return False


def run_grading_analysis(sample_number, stage_number, trial_number, selected_grades=None):
    """Run grade prediction from difference images. Returns grades dict or None."""
    try:
        log.info("Analyzing histograms for sample %s, stage %s, trial %s", sample_number, stage_number, trial_number)
        grades = analyze_difference_images_and_predict_output(
            sample_number, stage_number, trial_number, selected_grades=selected_grades,
        )
        if grades:
            log.info("Grade prediction completed: %s", grades)
            return grades
        log.warning("Grade prediction returned no results")
        return None
    except Exception:
        log.exception("run_grading_analysis failed")
        return None


def run_training_analysis(sample_number, stage_number, trial_number, grades_dict):
    """Append training features CSV rows for the provided grades dict."""
    try:
        log.info("Writing training features for sample %s, stage %s, trial %s", sample_number, stage_number, trial_number)
        analyze_difference_images(sample_number, stage_number, trial_number, grades=grades_dict)
    except Exception:
        log.exception("run_training_analysis failed")


def run_capture_pipeline(mode, sample_number, stage_number, trial_number,
                         motor_ip, motor_port, grades_input=None):
    """Shared capture → diff → analyze pipeline for both grading and training.

    Reports progress via st.session_state status keys. Returns True on
    success, False if any step failed.
    """
    suffix = _mode_suffix(mode)

    if stage_number != "0":
        stage0_path = os.path.join(
            cfg.get_input_dir(suffix),
            cfg.make_input_filename(sample_number, "0", trial_number, 1),
        )
        if not os.path.exists(stage0_path):
            st.session_state[S.KEY_CAPTURE_ERROR] = (
                f"Stage 0 images not found for trial {trial_number}. "
                "Please capture stage 0 first."
            )
            return False

    for k in S.STATUS_KEYS:
        st.session_state[k] = None

    if not capture_images_action(sample_number, stage_number, trial_number,
                                 suffix, motor_ip, motor_port):
        st.session_state[S.KEY_CAPTURE_ERROR] = t("image_capture_fail")
        return False
    st.session_state[S.KEY_CAPTURE_SUCCESS] = t("image_capture_success")
    st.session_state[S.KEY_FS_EPOCH] += 1

    if stage_number == "0":
        return True

    if not create_difference_action(sample_number, stage_number, trial_number,
                                    suffix, reference_stage="0"):
        st.session_state[S.KEY_DIFF_ERROR] = t("diff_create_fail")
        return False
    st.session_state[S.KEY_DIFF_SUCCESS] = t("diff_create_success")
    st.session_state[S.KEY_FS_EPOCH] += 1

    if mode == "grading":
        grades = run_grading_analysis(
            sample_number, stage_number, trial_number,
            selected_grades=st.session_state.get("selected_grades", list(cfg.GRADES)),
        )
        if grades is None:
            st.session_state[S.KEY_HIST_ERROR] = t("histogram_fail")
            return False
        st.session_state[S.KEY_GRADE] = grades
        st.session_state[S.KEY_HIST_SUCCESS] = t("histogram_success")
    else:
        run_training_analysis(sample_number, stage_number, trial_number, grades_input)
        st.session_state[S.KEY_HIST_SUCCESS] = t("histogram_success")

    return True


def capture_callback(mode, params):
    """Unified callback for both grading and training capture buttons."""
    try:
        st.session_state[S.KEY_CAPTURE_PROG] = True
        run_capture_pipeline(mode=mode, **params)
    except Exception as e:
        st.session_state[S.KEY_CAPTURE_ERROR] = f"Unexpected error: {str(e)}"
        log.exception("capture_callback (mode=%s) failed", mode)
    finally:
        st.session_state[S.KEY_CAPTURE_PROG] = False
