"""
Central config — all paths and toggles live here.

ITA-Net model layout (relative to project root):
  models/itanet/
    common/
      itanet.dll               shared runtime — loaded once, CWD is set per-grade
    {pilling,matting,fuzzing}/
      run/   .NET + .TRN + train/recall .dat files  (CWD when DLL is called)
      data/  recall.fls / training.fls
    archive/
      {pilling,matting,fuzzing}/  timestamped .NET + .TRN snapshots

Data layout:
  data/
    input_pictures/{for_grading,for_training}/
    difference_pictures/{for_grading,for_training}/
    training_features/   per-grade CSVs written during training
  output/grading_results/   per-trial analysis CSVs
  reports/                  exported PDFs

Run scripts from the project root.
"""

import os
from typing import Tuple

# ---- ITA-Net paths ----

ITANET_ROOT = os.path.join("models", "itanet")

# The three grade networks — each has its own run/ + data/ under ITANET_ROOT
GRADES = ("pilling", "matting", "fuzzing")

# Shared runtime — one DLL for all grades, CWD is set per-grade at call time
DLL_PATH = os.path.join(ITANET_ROOT, "common", "itanet.dll")


def get_itanet_run_dir(grade: str) -> str:
    return os.path.join(ITANET_ROOT, grade, "run")


def get_itanet_fls_dir(grade: str) -> str:
    return os.path.join(ITANET_ROOT, grade, "data")


def get_itanet_archive_dir(grade: str) -> str:
    return os.path.join(ITANET_ROOT, "archive", grade)


# ---- Backend selection ----

BACKEND = "itanet_dll"  # use "tf" for TensorFlow

# Set True to bypass camera/motor — generates synthetic PNG images instead.
MOCK_HARDWARE = True

# Set True to return synthetic grades without a trained .NET file.
MOCK_PREDICTION = True

# ---- TensorFlow model ----

TF_MODEL_PATH = os.path.join("models", "tf", "my_model.h5")

# ---- Data directories ----

DATA_DIR = "data"
INPUT_PICTURES_DIR = os.path.join(DATA_DIR, "input_pictures")
DIFFERENCE_PICTURES_DIR = os.path.join(DATA_DIR, "difference_pictures")
TRAINING_FEATURES_DIR = os.path.join(DATA_DIR, "training_features")

# ---- Motor settings ----
# Each capture cycle is MOTOR_STEPS_PER_CYCLE rotations of MOTOR_DEGREES_PER_STEP.
# The step count taken in the current cycle is persisted so an interrupted run
# can be resumed to the origin on the next capture.
MOTOR_STEPS_PER_CYCLE = 8
MOTOR_DEGREES_PER_STEP = 45.5
MOTOR_SPEED = 20
MOTOR_STATE_FILE = os.path.join(DATA_DIR, ".motor_state")

# Mode suffixes — the only valid subdirectory names under each picture dir
SUFFIX_GRADING = "for_grading"
SUFFIX_TRAINING = "for_training"


def get_input_dir(mode: str) -> str:
    return os.path.join(INPUT_PICTURES_DIR, mode)


def get_difference_dir(mode: str) -> str:
    return os.path.join(DIFFERENCE_PICTURES_DIR, mode)


def make_input_filename(sample: str, stage: str, trial: str, position: int) -> str:
    return f"{sample}-{stage}-{trial}-{position}.png"


def make_difference_filename(sample: str, stage: str, trial: str, position: int) -> str:
    return f"{sample}-{stage}-{trial}-{position}-dif.png"


# ---- Output directories ----

OUTPUT_DIR = "output"
GRADING_RESULTS_DIR = os.path.join(OUTPUT_DIR, "grading_results")
REPORTS_DIR = "reports"

# ---- Numerical settings ----

FEATURE_COLUMNS = ("Mean", "Std", "Max", "Mode")
DECIMAL = "Point"
ROUND_TO_HALF = True
CLIP_RANGE: Tuple[float, float] = (1.0, 5.0)

# ---- Application settings ----

ADMIN_PASSWORD = "1234"

# ---- Optional custom model overrides ----

CUSTOM_NET = None
CUSTOM_TRN = None
