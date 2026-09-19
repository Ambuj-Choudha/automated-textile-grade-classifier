import logging
import os
import pathlib
from typing import List
from app.settings import config as cfg

log = logging.getLogger(__name__)

# Ensures that the specified directory exists; creates it if it doesn't.
def ensure_directory(target_folder: str | os.PathLike, recursive: bool = True) -> bool:
    try:
        target_path = pathlib.Path(target_folder)
        if (target_path.exists()):
            return True

        target_path.mkdir(parents=recursive, exist_ok=True)
        log.debug("Created directory: %s", target_folder)
        return True
    except OSError as e:
        log.error("Failed to create directory %s: %s", target_folder, e)
        return False
    except Exception:
        log.exception("Unexpected error creating directory %s", target_folder)
        return False

# Validates that the input sample and stage number is alphanumeric.
def validate_input_number(input_number: str) -> bool:
    return bool(input_number and input_number.strip().isalnum())

# Validates that the input grade is correct.
def validate_grade(input_number: str) -> bool:
    if not input_number:
        return False
    try:
        num = float(input_number)
        return num in {1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}
    except ValueError:
        return False

# Returns a sorted list of unique sample numbers found in the input directory.
def get_available_samples(suffix: str) -> List[str]:
    input_directory = cfg.get_input_dir(suffix)
    try:
        if not os.path.exists(input_directory):
            return []
        
        samples = set()
        for filename in os.listdir(input_directory):
            if filename.endswith('.png') and '-' in filename:
                sample = filename.split('-')[0]
                if sample:
                    samples.add(sample)
        return sorted(samples)
    except Exception:
        log.exception("get_available_samples failed")
        return []

# Helper function to check if image exists and add to status
def _check_image_exists(image_path: str, image_name: str, image_list: List[str]) -> bool:
    """Helper function to check if an image exists and add it to the appropriate list."""
    if os.path.exists(image_path):
        image_list.append(image_name)
        return True
    return False

# Checks for the presence of all required image files (8 rotational) for a given sample-stage pair.
def check_required_images(sample_number: str, stage_number: str, trial_number: str, suffix: str) -> dict:
    status = {
        'all_input_present': True,
        'all_difference_present': True,
        'present_input_images': [],
        'present_difference_images': [],
    }

    try:
        for i in range(1, 9):
            image_name = cfg.make_input_filename(sample_number, stage_number, trial_number, i)
            image_path = os.path.join(cfg.get_input_dir(suffix), image_name)
            if not _check_image_exists(image_path, image_name, status['present_input_images']):
                status['all_input_present'] = False

        for i in range(1, 9):
            image_name = cfg.make_difference_filename(sample_number, stage_number, trial_number, i)
            image_path = os.path.join(cfg.get_difference_dir(suffix), image_name)
            if not _check_image_exists(image_path, image_name, status['present_difference_images']):
                status['all_difference_present'] = False

        return status

    except Exception:
        log.exception("check_required_images failed")
        return {
            'all_input_present': False,
            'all_difference_present': False,
            'present_input_images': [],
            'present_difference_images': [],
        }

