"""Streamlit-cached wrappers over the disk-scanning helpers.

The `epoch` parameter is st.session_state.fs_epoch — bumping it invalidates
these caches after the app writes new images.
"""
import os
import streamlit as st

from app.helpers.utils import get_available_samples, check_required_images


def file_sig(path: str):
    """(mtime_ns, size) — cheap change-detection key for image bytes."""
    try:
        stt = os.stat(path)
        return (stt.st_mtime_ns, stt.st_size)
    except FileNotFoundError:
        return (-1, -1)


@st.cache_data(show_spinner=False)
def cached_get_available_samples(mode: str, epoch: int):
    return get_available_samples(mode)


@st.cache_data(show_spinner=False)
def cached_check_required_images(sample_number: str, stage_number: str,
                                 trial_number: str, suffix: str, epoch: int):
    return check_required_images(sample_number, stage_number, trial_number, suffix)


@st.cache_data(show_spinner=False, ttl=300)
def load_image_bytes(path: str, sig) -> bytes:
    with open(path, "rb") as f:
        return f.read()
