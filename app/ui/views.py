"""Reusable rendered views composed by screens: logo, image viewer, and the
result / status blocks on capture screens."""
import os
import streamlit as st

from app.reporting.translations import t
from app.settings import config as cfg
from app import state as S
from app.ui.caching import (
    cached_check_required_images, file_sig, load_image_bytes,
)


def render_logo() -> None:
    """Render the centered TexIQ logo."""
    _, col, _ = st.columns([2, 2, 2])
    with col:
        st.image(os.path.join("logos", "Logo_TexIQ_v1.0.jpg"), width=200)


def show_image_viewer() -> None:
    """Expander that lets a user pick a folder + filename and view any PNG."""
    valid_folders = [
        "data/input_pictures/for_training", "data/difference_pictures/for_training",
        "data/input_pictures/for_grading",  "data/difference_pictures/for_grading",
    ]

    with st.expander(t("view_images")):
        col1, col2 = st.columns(2)
        with col1:
            selected_folder = st.selectbox(t("select_folder"), valid_folders, key="k_viewer_folder")
        with col2:
            image_filename = st.text_input(t("enter_filename"), key="k_viewer_file",
                                           help="e.g., 001-1-1, 001-1-1-dif, 001-1-ref")

        normalized_filename = (image_filename or "").strip()
        if normalized_filename and not os.path.splitext(normalized_filename)[1]:
            normalized_filename = f"{normalized_filename}.png"

        if selected_folder and normalized_filename:
            image_path = os.path.join(selected_folder, normalized_filename)
            if os.path.exists(image_path):
                sig = file_sig(image_path)
                st.image(load_image_bytes(image_path, sig),
                         caption=os.path.basename(image_path), use_column_width=True)
            else:
                st.error(t("invalid_picture_name"))


def display_operation_status(sample_number, stage_number, trial_number, suffix) -> None:
    st.subheader(f"{t('sample_number')} {sample_number}, {t('stage_number')} {stage_number}, {t('trial_number')} {trial_number}")

    image_status = cached_check_required_images(sample_number, stage_number, trial_number, suffix,
                                                 st.session_state[S.KEY_FS_EPOCH])
    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric(t("available_samples"), f"{len(image_status['present_input_images'])}/8")
        if image_status['all_input_present']:
            st.success(t("valid_input").replace("✅ Input parameters valid", "✅ All input images present"))

    with col2:
        st.metric(t("results"), f"{len(image_status['present_difference_images'])}/8")
        if image_status['all_difference_present']:
            st.success(t("valid_input").replace("✅ Input parameters valid", "✅ All difference images present"))

    st.markdown(f"#### {t('view_images')}")
    input_images = image_status['present_input_images']
    input_dir = cfg.get_input_dir(suffix)
    for i in range(0, len(input_images), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(input_images):
                image_path = os.path.join(input_dir, input_images[i + j])
                if os.path.exists(image_path):
                    sig = file_sig(image_path)
                    with cols[j]:
                        st.image(load_image_bytes(image_path, sig),
                                 caption=input_images[i + j], use_column_width=True)


def display_grades(grades, sample_number, stage_number) -> None:
    if not grades:
        st.warning("No analysis results to display")
        return

    selected = st.session_state.get(S.KEY_SELECTED_GRADES, list(cfg.GRADES))
    active = [g for g in cfg.GRADES if g in selected and g in grades]
    if not active:
        st.warning("No grade predictions available for the selected grades.")
        return

    grade_label_keys = {
        "pilling": "pilling_grade_result",
        "matting": "matting_grade_result",
        "fuzzing": "fuzzing_grade_result",
    }
    st.markdown(f"**{t('sample_number')} {sample_number} — {t('stage_number')} {stage_number}**")
    cols = st.columns(len(active))
    for col, grade in zip(cols, active):
        with col:
            st.metric(t(grade_label_keys[grade]), grades.get(grade, "—"))
