import logging
import os
import hashlib
import numpy as np
from PIL import Image
from app.helpers.utils import ensure_directory
from app.settings import config as cfg

log = logging.getLogger(__name__)

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
    seed = int(hashlib.md5(f"{sample_number}{trial_number}".encode()).hexdigest(), 16) % (2 ** 32)
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
        log.debug("mock capture: position %d -> %s", i, pic_name)

    log.info("mock capture complete — sample=%s stage=%s trial=%s", sample_number, stage_number, trial_number)
    return True


# ---------------------------------------------------------------------------
# Motor position tracking — persists step count so an interrupted run can be
# resumed to the origin on the next capture. State is a single integer in
# cfg.MOTOR_STATE_FILE representing steps taken in the current cycle.
# ---------------------------------------------------------------------------

def _read_motor_steps() -> int:
    try:
        with open(cfg.MOTOR_STATE_FILE, "r", encoding="utf-8") as f:
            return max(0, min(cfg.MOTOR_STEPS_PER_CYCLE, int(f.read().strip() or "0")))
    except (OSError, ValueError):
        return 0


def _write_motor_steps(n: int) -> None:
    ensure_directory(os.path.dirname(cfg.MOTOR_STATE_FILE) or ".")
    with open(cfg.MOTOR_STATE_FILE, "w", encoding="utf-8") as f:
        f.write(str(n))


def _rotate_motor(motor_ip: str, motor_port: int, degrees: float) -> None:
    conn = rpyc.connect(motor_ip, port=motor_port)
    try:
        conn.root.run_motor_degrees(cfg.MOTOR_SPEED, degrees)
    finally:
        conn.close()


def _return_motor_to_origin(motor_ip: str, motor_port: int) -> bool:
    """If the previous run was interrupted, complete the remaining rotations
    to bring the motor back to the origin. Idempotent when already at origin.
    """
    steps = _read_motor_steps()
    if steps == 0:
        return True
    remaining = cfg.MOTOR_STEPS_PER_CYCLE - steps
    log.info("Motor was at step %d/%d — returning to origin (%d rotations)",
             steps, cfg.MOTOR_STEPS_PER_CYCLE, remaining)
    try:
        _rotate_motor(motor_ip, motor_port, remaining * cfg.MOTOR_DEGREES_PER_STEP)
        _write_motor_steps(0)
        return True
    except Exception:
        log.exception("Failed to return motor to origin")
        return False


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
            image_cv = cv2.imread(temp_bmp_path)
            if image_cv is None:
                raise RuntimeError(f"Failed to load temporary BMP file: {temp_bmp_path}")
            if save_file:
                cv2.imwrite(final_path, image_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                log.debug("Saved image to: %s", final_path)
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
        log.error("No camera devices found")
        return False

    log.info("Connecting to motor at %s:%s", motor_ip, motor_port)
    if motor_ip and not _return_motor_to_origin(motor_ip, motor_port):
        return False

    for i in range(1, cfg.MOTOR_STEPS_PER_CYCLE + 1):
        cam = None
        try:
            cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            cam.Open()
            picture_name = cfg.make_input_filename(sample_number, stage_number, trial_number, i)
            _get_image_from_cam(cam, input_dir, img, save_file=True, file_name=picture_name, user_set="UserSet2")
            log.info("Captured image: %s", picture_name)
        except genicam.GenericException as e:
            log.error("Camera exception: %s", e.GetDescription())
            return False
        except Exception:
            log.exception("Unexpected error during capture")
            return False
        finally:
            if cam is not None:
                cam.Close()

        if motor_ip:
            try:
                _rotate_motor(motor_ip, motor_port, cfg.MOTOR_DEGREES_PER_STEP)
                _write_motor_steps(i % cfg.MOTOR_STEPS_PER_CYCLE)
                log.debug("Motor rotated %.1f deg after image %d", cfg.MOTOR_DEGREES_PER_STEP, i)
            except Exception:
                log.exception("Motor control failed")
                return False

    log.info("Captured all images for sample=%s stage=%s trial=%s", sample_number, stage_number, trial_number)
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def capture_sample_images(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port=18812):
    try:
        if cfg.MOCK_HARDWARE:
            return _mock_capture(sample_number, stage_number, trial_number, suffix)

        if not _HW_AVAILABLE:
            log.error("Camera/motor libraries not installed and MOCK_HARDWARE is False")
            return False

        return _real_capture(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port)

    except Exception:
        log.exception("capture_sample_images failed")
        return False
