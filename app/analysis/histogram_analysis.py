import os
import csv
import tempfile
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2

from app.helpers.utils import ensure_directory
from itanet_recall import predict_from_csv
from app.settings import config as cfg

# Try to import TensorFlow if installed; do not fail if missing
try:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

# --------------------------------- Constants ---------------------------------

FEATURE_HEADERS: Tuple[str, str, str, str, str] = ("Image", "Mean", "Std", "Max", "Mode")
TRAIN_HEADERS: Tuple[str, str, str, str, str, str] = (*FEATURE_HEADERS, "Grade")

# Track last-used backend for diagnostics
LAST_BACKEND_USED: str = "itanet_dll"

# -------- TensorFlow helpers (batch, via CSV) --------

_TF_MODEL = None


def _get_tf_model():
    """Load and cache the TensorFlow model (if BACKEND=tf)."""
    global _TF_MODEL
    if _TF_MODEL is not None:
        return _TF_MODEL
    if load_model is None:
        raise RuntimeError("TensorFlow is not installed, but BACKEND=tf was selected.")
    model_path = cfg.TF_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TF model file not found: {model_path}")
    _TF_MODEL = load_model(model_path)
    return _TF_MODEL


def _tf_batch_predict_csv(csv_path: str, output_path: Optional[str]) -> str:
    """Run a TF model on a CSV (Mean/Std/Max/Mode) and write predictions CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    for col in cfg.FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")
    X = df[list(cfg.FEATURE_COLUMNS)].astype(np.float32).to_numpy()
    model = _get_tf_model()
    preds = model.predict(X, verbose=0).reshape(-1)

    # Round to nearest 0.5 and clip [1.0, 5.0]
    preds = np.clip(np.round(preds * 2) / 2.0, 1.0, 5.0)

    df_out = df.copy()
    df_out["Predicted_Grade"] = preds
    if output_path is None:
        base, _ = os.path.splitext(csv_path)
        output_path = f"{base}_predictions.csv"
    df_out.to_csv(output_path, index=False)
    return output_path


# -------- Image feature extraction --------

def _calculate_image_stats(image_path: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate statistics for a single image using grayscale intensities.
    Returns (mean, std, max, mode) as floats, or None if image cannot be read.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    img = image.astype(np.float32)
    mean_val = float(np.mean(img))
    std_val = float(np.std(img))
    max_val = float(np.max(img))

    # Fast mode via histogram (0..255)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).reshape(-1)
    mode_val = float(int(np.argmax(hist)))

    return mean_val, std_val, max_val, mode_val


def _calculate_feature_averages(all_stats: List[List[Union[str, float]]]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    From rows [filename, mean, std, max, mode], compute (mean_avg, std_avg, max_avg, mode_avg).
    Returns Nones if empty.
    """
    if not all_stats:
        return None, None, None, None
    return tuple(float(np.mean([row[i] for row in all_stats])) for i in range(1, 5))


def _process_difference_images(diff_dir: str, sample_number: str, stage_number: str, trial_number: str, grade_number: Optional[Union[str, float]] = None) -> List[List[Union[str, float]]]:
    """
    Process all difference images and return per-image stats rows.
    Each row is [filename, mean, std, max, mode] (+ grade if provided).
    """
    rows: List[List[Union[str, float]]] = []
    for i in range(1, 9):
        diff_filename = cfg.make_difference_filename(sample_number, stage_number, trial_number, i)
        diff_path = os.path.join(diff_dir, diff_filename)

        stats_result = _calculate_image_stats(diff_path)
        if stats_result is None:
            print(f"[WARN] Could not read image: {diff_path}")
            continue

        mean_val, std_val, max_val, mode_val = stats_result
        if grade_number is not None:
            rows.append([diff_filename, mean_val, std_val, max_val, mode_val, grade_number])
        else:
            rows.append([diff_filename, mean_val, std_val, max_val, mode_val])

    return rows


def _write_csv_with_headers(file_path: str, headers: Sequence[str], data: List[Sequence[Union[str, float]]], append_mode: bool = False) -> None:
    """Write CSV with headers once; append if requested and file non-empty."""
    file_exists = os.path.exists(file_path)
    file_is_empty = (not file_exists) or os.path.getsize(file_path) == 0
    mode = "a" if append_mode and file_exists and not file_is_empty else "w"
    ensure_directory(os.path.dirname(file_path) or ".")
    with open(file_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w" or file_is_empty:
            writer.writerow(list(headers))
        writer.writerows(data)


def _get_diff_dir(suffix: str) -> str:
    return cfg.get_difference_dir(suffix)


def _safe_unlink(path: Optional[str]) -> None:
    """Best-effort file deletion."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


# -------- Backend adapter (CSV-only) --------

def _run_backend_batch(csv_path: str, output_path: Optional[str], custom_net_file: Optional[str], custom_trn_file: Optional[str]) -> str:
    """
    Dispatch prediction to selected backend and return path to predictions CSV.
    custom_net_file/custom_trn_file are accepted for parity but currently used by ITANET DLL only.
    """
    global LAST_BACKEND_USED
    if cfg.BACKEND == "tf":
        LAST_BACKEND_USED = "tf"
        print("[BACKEND] Using TensorFlow model")
        return _tf_batch_predict_csv(csv_path, output_path)

    LAST_BACKEND_USED = "itanet_dll"
    print("[BACKEND] Using ITANET DLL")
    return predict_from_csv(
        csv_path=csv_path,
        data_dir=cfg.ITANET_RUN_DIR,
        dll_path=cfg.DLL_PATH,
        output_path=output_path,
        feature_cols=cfg.FEATURE_COLUMNS,
        clip_range=cfg.CLIP_RANGE,
        fls_dir=cfg.ITANET_FLS_DIR,
    )


# -------- Public functions --------

def analyze_difference_images_and_predict_output(sample_number: str, stage_number: str, trial_number: str, output_dir: str = cfg.GRADING_RESULTS_DIR) -> Optional[float]:
    """
    Analyze per-image stats, write analysis CSV, then perform prediction by:
      - Creating a temporary single-row CSV with averaged features
      - Running backend batch recall on that CSV
      - Reading the predicted grade from the predictions CSV
    Returns the predicted grade (float) or None on error.
    """
    try:
        diff_dir = _get_diff_dir("for_grading")
        analysis_dir = output_dir
        ensure_directory(analysis_dir)

        if not os.path.exists(diff_dir):
            print(f"[ERR] Difference pictures directory not found: {diff_dir}")
            return None

        # Extract per-image stats
        all_stats = _process_difference_images(diff_dir, sample_number, stage_number, trial_number)
        if not all_stats:
            print("[WARN] No valid images found for analysis.")
            return None

        # Compute averages
        mean_avg, std_avg, max_avg, mode_avg = _calculate_feature_averages(all_stats)
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in (mean_avg, std_avg, max_avg, mode_avg)):
            print("[WARN] Invalid averaged features.")
            return None

        # Append "Average" row to analysis table (for final CSV)
        all_stats.append(["Average", mean_avg, std_avg, max_avg, mode_avg])

        temp_single_csv = None
        preds_csv = None

        # Create temporary single-row CSV for prediction
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as temp_f:
            temp_single_csv = temp_f.name
            w = csv.writer(temp_f)
            w.writerow(list(FEATURE_HEADERS))
            w.writerow([f"{sample_number}-{stage_number}-{trial_number}", mean_avg, std_avg, max_avg, mode_avg])

        try:
            # Run backend batch on the temporary single-row CSV
            preds_csv = _run_backend_batch(
                temp_single_csv,
                None,
                custom_net_file=(cfg.CUSTOM_NET or os.environ.get("CUSTOM_NET")),
                custom_trn_file=(cfg.CUSTOM_TRN or os.environ.get("CUSTOM_TRN")),
            )

            # Read predicted grade
            with open(preds_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)

            if not row or "Predicted_Grade" not in row:
                print("[ERR] No predicted grade found in predictions CSV.")
                return None

            grade_str = str(row["Predicted_Grade"]).strip()
            try:
                prediction = float(grade_str)
            except ValueError:
                print(f"[ERR] Invalid predicted grade: {grade_str}")
                prediction = None

        finally:
            _safe_unlink(temp_single_csv)
            _safe_unlink(preds_csv)

        if prediction is None:
            return None

        # Write final analysis CSV with all data and metadata
        analysis_csv = os.path.join(analysis_dir, f"{sample_number}-{stage_number}-{trial_number}-analysis.csv")

        # Add grade and backend info to the stats
        all_stats.append(["Grade", prediction, "", "", ""])
        all_stats.append(["Backend", cfg.BACKEND, "", "", ""])

        _write_csv_with_headers(
            analysis_csv,
            list(FEATURE_HEADERS),
            all_stats,
        )

        print(f"[OK] Analysis saved: {analysis_csv} (backend={cfg.BACKEND})")
        return prediction

    except Exception as e:
        print(f"[ERR] analyze_difference_images_and_predict_output failed: {e}")
        return None


def analyze_difference_images(sample_number: str, stage_number: str, trial_number: str, grade_number: Union[str, float], output_dir: str = cfg.TRAINING_FEATURES_DIR) -> Optional[None]:
    """
    Build CSV rows of (Mean, Std, Max, Mode, Grade) feature vectors for ITA-Net training:
      - per_image_features.csv: one row per difference image (8 per sample)
      - averaged_features.csv:  one averaged row per sample
    Either file can be passed to itanet_training.py as the training input.
    """
    try:
        diff_dir = _get_diff_dir("for_training")
        out_dir = output_dir
        ensure_directory(out_dir)

        if not os.path.exists(diff_dir):
            print(f"[ERR] Difference pictures directory not found: {diff_dir}")
            return None

        # Coerce grade to rounded half-steps as float for dataset consistency
        try:
            g = float(str(grade_number).replace(",", "."))
            g = round(g * 2) / 2.0
        except ValueError:
            print(f"[ERR] Invalid grade_number: {grade_number}")
            return None

        all_stats = _process_difference_images(diff_dir, sample_number, stage_number, trial_number, g)
        if not all_stats:
            print("[WARN] No valid images found for analysis.")
            return None

        mean_avg, std_avg, max_avg, mode_avg = _calculate_feature_averages(all_stats)
        avg_stats = [[f"{sample_number}-{stage_number}-{trial_number}", mean_avg, std_avg, max_avg, mode_avg, g]]

        out_per_image = os.path.join(out_dir, "per_image_features.csv")
        out_averaged = os.path.join(out_dir, "averaged_features.csv")

        _write_csv_with_headers(out_per_image, TRAIN_HEADERS, all_stats, append_mode=True)
        _write_csv_with_headers(out_averaged, TRAIN_HEADERS, avg_stats, append_mode=True)

        print(f"[OK] Training rows appended to {out_per_image} and {out_averaged}")

    except Exception as e:
        print(f"[ERR] analyze_difference_images failed: {e}")
        return None


def batch_predict_from_csv(csv_path: str, output_path: Optional[str] = None, custom_net_file: Optional[str] = None, custom_trn_file: Optional[str] = None) -> str:
    """
    Batch prediction from CSV using the selected backend.
    For itanet_dll, delegates to predict_from_csv.
    For tf, runs a TF model if installed and my_model.h5 is present.
    """
    return _run_backend_batch(csv_path, output_path, custom_net_file, custom_trn_file)