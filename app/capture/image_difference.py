import os
from PIL import Image, ImageChops
from app.helpers.utils import ensure_directory
from app.settings import config as cfg


def _validate_image_paths(current_path: str, reference_path: str, position: int) -> bool:
    if not os.path.exists(current_path):
        print(f"Current image not found: {current_path}")
        return False
    if not os.path.exists(reference_path):
        print(f"Reference image not found: {reference_path}")
        return False
    return True


def _process_image_pair(current_path: str, reference_path: str, output_path: str, position: int) -> bool:
    try:
        with Image.open(current_path) as current_img, Image.open(reference_path) as reference_img:
            if current_img.size != reference_img.size:
                print(f"Image sizes don't match for position {position}, resizing...")
                reference_img = reference_img.resize(current_img.size)
            diff_img = ImageChops.difference(current_img, reference_img)
            diff_img.save(output_path)
        print(f"Created difference image: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"Error creating difference image for position {position}: {e}")
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

    except Exception as e:
        print(f"Error in creating difference images: {e}")
        return False
