import pathlib
from PIL import Image, ImageChops
import os
import cv2
import datetime
from app.helpers.utils import ensure_directory

# Helper function to validate image existence
def _validate_image_paths(current_path: str, reference_path: str, position: int) -> bool:
    """Helper function to validate existence of both current and reference images."""
    if not os.path.exists(current_path):
        print(f"Current image not found: {current_path}")
        return False
    if not os.path.exists(reference_path):
        print(f"Reference image not found: {reference_path}")
        return False
    return True

# Helper function to process a single image pair
def _process_image_pair(current_path: str, reference_path: str, output_path: str, position: int) -> bool:
    """Helper function to process a single current-reference image pair and create difference image."""
    try:
        # Open both images
        with Image.open(current_path) as current_img, Image.open(reference_path) as reference_img:
            # Resize reference image if dimensions don't match
            if current_img.size != reference_img.size:
                print(f"Image sizes don't match for position {position}, resizing...")
                reference_img = reference_img.resize(current_img.size)

            # Compute and save difference image
            diff_img = ImageChops.difference(current_img, reference_img)
            diff_img.save(output_path)
            
        print(f"Created difference image: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"Error creating difference image for position {position}: {e}")
        return False

# Helper function to construct file paths
def _construct_paths(script_dir: str, suffix: str, sample_number: str, stage_number: str, trial_number: str, reference_stage: str, position: int) -> tuple:
    """Helper function to construct all necessary file paths for a given position."""
    input_dir = os.path.join(script_dir, "data", "input_pictures", suffix)
    diff_dir = os.path.join(script_dir, "data", "difference_pictures", suffix)
    
    current_image_path = os.path.join(input_dir, f"{sample_number}-{stage_number}-{trial_number}-{position}.png")
    reference_image_path = os.path.join(input_dir, f"{sample_number}-{reference_stage}-{trial_number}-{position}.png")
    diff_filename = f"{sample_number}-{stage_number}-{trial_number}-{position}-dif.png"
    diff_path = os.path.join(diff_dir, diff_filename)
    
    return current_image_path, reference_image_path, diff_path, diff_dir

# Creates difference images by subtracting reference images from current stage images
def create_difference_images(sample_number, stage_number, trial_number, suffix, reference_stage="0"):
    try:
        # Walk up from app/capture/image_difference.py to the project root.
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Process each image position (1-8)
        for i in range(1, 9):
            current_path, reference_path, diff_path, diff_dir = _construct_paths(
                script_dir, suffix, sample_number, stage_number, trial_number, reference_stage, i
            )
            
            # Ensure difference directory exists (only create once per position if needed)
            if i == 1:  # Only check/create directory once
                ensure_directory(diff_dir)
            
            # Validate image existence
            if not _validate_image_paths(current_path, reference_path, i):
                continue
                
            # Process the image pair
            _process_image_pair(current_path, reference_path, diff_path, i)
        
        return True

    except Exception as e:
        print(f"Error in creating difference images: {e}")
        return False