import os
import numpy as np
from PIL import Image
from app.helpers.utils import ensure_directory
from app.settings import config as cfg

try:
    import cv2
    import rpyc
    from pypylon import pylon
    from pypylon import genicam
    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mock capture — generates synthetic PNG images without real hardware
# ---------------------------------------------------------------------------

def _mock_capture(sample_number: str, stage_number: str, trial_number: str, suffix: str) -> bool:
    """
    Produce synthetic images that let the full grading pipeline run without hardware.

    Stage 0 images are "clean" (low mean intensity difference vs reference).
    Later stages get progressively brighter to simulate pilling, so difference
    images carry meaningful stats for ITA-Net / TF analysis.
    """
    input_dir = cfg.get_input_dir(suffix)
    ensure_directory(input_dir)

    H, W = 480, 640
    # Seed from inputs so re-runs for the same sample/stage are identical
    seed = hash((sample_number, trial_number)) % (2 ** 32)
    rng = np.random.default_rng(seed)

    # Stage 0 is the clean baseline; higher stages are progressively "pilled"
    try:
        stage_val = int(stage_number)
    except ValueError:
        stage_val = 0
    pilling_offset = min(stage_val // 100 * 8, 60)  # gentle ramp, cap at +60
    ref_base = 110

    # 8 input images: each position has a slight angular brightness variation
    for i in range(1, 9):
        angle_offset = (i - 4) * 3          # ±12 across 8 positions
        brightness = ref_base + pilling_offset + angle_offset
        img_arr = rng.normal(brightness, 8, (H, W)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L").convert("RGB")
        pic_name = cfg.make_input_filename(sample_number, stage_number, trial_number, i)
        img.save(os.path.join(input_dir, pic_name))
        print(f"[MOCK] position {i} → {pic_name}")

    print(f"[MOCK] capture complete — sample={sample_number} stage={stage_number} trial={trial_number}")
    return True


# ---------------------------------------------------------------------------
# Real capture — requires Basler camera + motor controller
# ---------------------------------------------------------------------------

def _get_image_from_cam(camera, target_path, img, save_file=True, file_name="Test-ref.jpg", user_set="UserSet1"):
    camera.UserSetSelector.SetValue(user_set)
    camera.UserSetLoad.Execute()
    camera.StartGrabbing()
    try:
        with camera.RetrieveResult(2000) as result:
            img.AttachGrabResultBuffer(result)
            temp_bmp_path = os.path.join(target_path, "temp_image.bmp")
            final_path = os.path.join(target_path, file_name)
            img.Save(pylon.ImageFileFormat_Bmp, temp_bmp_path)
            import cv2 as _cv2
            image_cv = _cv2.imread(temp_bmp_path)
            if image_cv is None:
                raise RuntimeError(f"Failed to load temporary BMP file: {temp_bmp_path}")
            if save_file:
                _cv2.imwrite(final_path, image_cv, [int(_cv2.IMWRITE_JPEG_QUALITY), 100])
                print(f"Saved image to: {final_path}")
            os.remove(temp_bmp_path)
            img.Release()
            return image_cv
    finally:
        camera.StopGrabbing()


def _real_capture(sample_number: str, stage_number: str, trial_number: str, suffix: str, motor_ip: str, motor_port: int) -> bool:
    input_dir = cfg.get_input_dir(suffix)
    ensure_directory(input_dir)

    img = pylon.PylonImage()
    tlf = pylon.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    if len(devices) == 0:
        print("No camera devices found!")
        return False

    print(f"Connecting to motor at {motor_ip}:{motor_port}")
    for i in range(1, 9):
        try:
            cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            cam.Open()
            picture_name = cfg.make_input_filename(sample_number, stage_number, trial_number, i)
            _get_image_from_cam(cam, input_dir, img, save_file=True, file_name=picture_name, user_set="UserSet2")
            print(f"Captured image: {picture_name}")
        except genicam.GenericException as e:
            print(f"Camera exception: {e.GetDescription()}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False
        finally:
            if "cam" in locals():
                cam.Close()

        if motor_ip:
            try:
                conn = rpyc.connect(motor_ip, port=motor_port)
                conn.root.run_motor_degrees(20, 45.5)
                conn.close()
                print(f"Motor rotated 45° after image {i}")
            except Exception as e:
                print(f"Motor control failed: {e}")
                return False

    print(f"Captured all images for sample={sample_number} stage={stage_number} trial={trial_number}")
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def capture_sample_images(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port=18812):
    try:
        if cfg.MOCK_HARDWARE:
            return _mock_capture(sample_number, stage_number, trial_number, suffix)

        if not _HW_AVAILABLE:
            print("[ERR] Camera/motor libraries not installed and MOCK_HARDWARE is False.")
            return False

        return _real_capture(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port)

    except Exception as e:
        print(f"Error in capture_sample_images: {e}")
        return False
