import datetime
import os
import sys
import cv2
import rpyc
import platform
from pypylon import pylon
from pypylon import genicam
from app.helpers.utils import ensure_directory

# Captures a single image from the camera using specified User Set configuration
def get_image_from_cam(camera, target_path, img, save_file=True, file_name="Test-ref.jpg", user_set="UserSet1"):
    # Load specified User Set configuration (contains all camera settings)
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
                print(f"Saved image to: {final_path}")

            os.remove(temp_bmp_path)
            img.Release()
            return image_cv
    finally:
        camera.StopGrabbing()


# Captures a reference image and 8 rotated images for a sample-stage pair, rotating motor between captures
def capture_sample_images(sample_number, stage_number, trial_number, suffix, motor_ip, motor_port=18812):
    try:
        input_dir = os.path.join("data", "input_pictures", suffix)
        reference_dir = os.path.join("data", "reference_pictures", suffix)
        ensure_directory(input_dir)
        ensure_directory(reference_dir)

        img = pylon.PylonImage()
        tlf = pylon.TlFactory.GetInstance()
        devices = tlf.EnumerateDevices()
        if len(devices) == 0:
            print("No camera devices found!")
            return False

        cam = pylon.InstantCamera(tlf.CreateFirstDevice())
        cam.Open()

        # Capture reference image using User Set 1 (format: {sample}-{stage}-{trial}-ref.png)
        reference_name = f"{sample_number}-{stage_number}-{trial_number}-ref.png"
        get_image_from_cam(cam, reference_dir, img, file_name=reference_name, user_set="UserSet1")
        print(f"Captured reference image with User Set 1: {reference_name}")
        cam.Close()

        print(f"Connecting to motor at {motor_ip}:{motor_port}")

        # Capture 8 images at 45° increments using User Set 2 (format: {sample}-{stage}-{trial}-{position}.png)
        for i in range(1, 9):
            try:
                cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                cam.Open()

                picture_name = f"{sample_number}-{stage_number}-{trial_number}-{i}.png"
                get_image_from_cam(cam, input_dir, img, save_file=True, file_name=picture_name, user_set="UserSet2")
                print(f"Captured image with User Set 2: {picture_name}")
            except genicam.GenericException as e:
                print("Camera exception occurred:")
                print(e.GetDescription())
                return False
            except Exception as e:
                print(f"Unexpected error: {e}")
                return False
            finally:
                if 'cam' in locals():
                    cam.Close()

            # Rotate motor by 45 degrees after each image capture
            if motor_ip:
                try:
                    conn = rpyc.connect(motor_ip, port=motor_port)
                    conn.root.run_motor_degrees(20, 45.5)  # Rotate motor
                    conn.close()
                    print(f"Motor rotated 45 degrees after image {i}")
                except Exception as e:
                    print(f"Motor control failed: {e}, motor ip:{motor_ip}")
                    return False

        print(f"Successfully captured all images for sample {sample_number}, stage {stage_number}, trial {trial_number}")
        return True
    except Exception as e:
        print(f"Error in capturing sample images: {e}")
        return False
