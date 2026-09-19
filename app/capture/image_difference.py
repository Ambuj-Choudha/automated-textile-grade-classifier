import logging
import os
from PIL import Image, ImageChops
from app.helpers.utils import ensure_directory
from app.settings import config as cfg

log = logging.getLogger(__name__)


def _validate_image_paths(current_path: str, reference_path: str, position: int) -> bool:
    if not os.path.exists(current_path):
        log.warning("Current image not found: %s", current_path)
        return False
    if not os.path.exists(reference_path):
        log.warning("Reference image not found: %s", reference_path)
        return False
    return True


def _process_image_pair(current_path: str, reference_path: str, output_path: str, position: int) -> bool:
    try:
        with Image.open(current_path) as current_img, Image.open(reference_path) as reference_img:
            if current_img.size != reference_img.size:
                log.warning("Image sizes don't match for position %d, resizing", position)
                reference_img = reference_img.resize(current_img.size)
            diff_img = ImageChops.difference(current_img, reference_img)
            diff_img.save(output_path)
        log.debug("Created difference image: %s", os.path.basename(output_path))
        return True
    except Exception:
        log.exception("Error creating difference image for position %d", position)
        return False


def create_difference_images(sample_number, stage_number, trial_number, suffix, reference_stage="0"):
    try:
        input_dir = cfg.get_input_dir(suffix)
        diff_dir = cfg.get_difference_dir(suffix)
        ensure_directory(diff_dir)

        for i in range(1, 9):
            current_path = os.path.join(input_dir, cfg.make_input_filename(sample_number, stage_number, trial_number, i))
            reference_path = os.path.join(input_dir, cfg.make_input_filename(sample_number, reference_stage, trial_number, i))
            diff_path = os.path.join(diff_dir, cfg.make_difference_filename(sample_number, stage_number, trial_number, i))

            if not _validate_image_paths(current_path, reference_path, i):
                continue
            _process_image_pair(current_path, reference_path, diff_path, i)

        return True

    except Exception:
        log.exception("create_difference_images failed")
        return False
