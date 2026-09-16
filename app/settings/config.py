"""
Central config is at app/settings/config.py. Defaults:
BACKEND = "itanet_dll"
ITANET_ROOT = models/itanet
ITANET_RUN_DIR = models/itanet/run          (CWD when the DLL is called)
ITANET_FLS_DIR = models/itanet/data         (holds recall.fls / training.fls)
DLL_PATH = models/itanet/run/itanet.dll
TF_MODEL_PATH = models/tf/my_model.h5
FEATURE_COLUMNS = ("Mean","Std","Max","Mode")

Folder layout used by the rest of the code (all relative to the project root,
which is the cwd when you run app.py or itanet_training.py):
  models/itanet/
    run/       ITA-Net DLL + .NET/.TRN + train/recall .dat files (CWD for DLL)
    data/      recall.fls and training.fls (DLL reads these as ../data/*.fls)
    archive/   backups of Neuronalesnetz.NET before each training run
  models/tf/       TensorFlow backend model
  data/            input images and training-feature CSVs
  output/          per-trial grading-result CSVs
  reports/         exported PDFs

Note: Run scripts from the project root directory.
"""

import os
from typing import Tuple

# ITA-NET data and DLL paths - relative to current working directory
ITANET_ROOT = os.path.join("models", "itanet")
ITANET_RUN_DIR = os.path.join(ITANET_ROOT, "run")
ITANET_FLS_DIR = os.path.join(ITANET_ROOT, "data")
ITANET_ARCHIVE_DIR = os.path.join(ITANET_ROOT, "archive")
DLL_PATH = os.path.join(ITANET_RUN_DIR, "itanet.dll")

# Legacy alias — some external tools still import ITANET_DATA_DIR; points at
# the run dir now so predict_from_csv keeps working with the old kwarg name.
ITANET_DATA_DIR = ITANET_RUN_DIR

# Backend selection
BACKEND = "itanet_dll" # use "tf" for tensorflow

# TensorFlow model
TF_MODEL_PATH = os.path.join("models", "tf", "my_model.h5")

# Input data folders (under data/)
DATA_DIR = "data"
INPUT_PICTURES_DIR = os.path.join(DATA_DIR, "input_pictures")
REFERENCE_PICTURES_DIR = os.path.join(DATA_DIR, "reference_pictures")
DIFFERENCE_PICTURES_DIR = os.path.join(DATA_DIR, "difference_pictures")
TRAINING_FEATURES_DIR = os.path.join(DATA_DIR, "training_features")

# Mode suffixes — the only two valid subdirectory names under each picture dir
SUFFIX_GRADING = "for_grading"
SUFFIX_TRAINING = "for_training"


# --- Directory path builders ---

def get_input_dir(mode: str) -> str:
    return os.path.join(INPUT_PICTURES_DIR, mode)


def get_reference_dir(mode: str) -> str:
    return os.path.join(REFERENCE_PICTURES_DIR, mode)


def get_difference_dir(mode: str) -> str:
    return os.path.join(DIFFERENCE_PICTURES_DIR, mode)


# --- File name builders ---

def make_input_filename(sample: str, stage: str, trial: str, position: int) -> str:
    return f"{sample}-{stage}-{trial}-{position}.png"


def make_reference_filename(sample: str, stage: str, trial: str) -> str:
    return f"{sample}-{stage}-{trial}-ref.png"


def make_difference_filename(sample: str, stage: str, trial: str, position: int) -> str:
    return f"{sample}-{stage}-{trial}-{position}-dif.png"

# Generated outputs
OUTPUT_DIR = "output"
GRADING_RESULTS_DIR = os.path.join(OUTPUT_DIR, "grading_results")
REPORTS_DIR = "reports"

# Data schema and numerical settings
FEATURE_COLUMNS = ("Mean", "Std", "Max", "Mode")
DECIMAL = "Point"
ROUND_TO_HALF = True
CLIP_RANGE: Tuple[float, float] = (1.0, 5.0)

# Optional custom files
CUSTOM_NET = None
CUSTOM_TRN = None
