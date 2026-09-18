import logging
import os
import csv
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2

from app.helpers.utils import ensure_directory
from itanet_recall import predict_from_csv
from app.settings import config as cfg

log = logging.getLogger(__name__)

# Try to import TensorFlow if installed; do not fail if missing
try:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

# --------------------------------- Constants ---------------------------------

FEATURE_HEADERS: Tuple[str, str, str, str, str] = ("Image", "Mean", "Std", "Max", "Mode")
TRAIN_HEADERS: Tuple[str, str, str, str, str, str] = (*FEATURE_HEADERS, "Grade")

LAST_BACKEND_USED: str = "itanet_dll"

# -------- TensorFlow helpers --------

_TF_MODEL = None


def _get_tf_model():
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
    import pandas as pd
    df = pd.read_csv(csv_path)
    for col in cfg.FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")
    X = df[list(cfg.FEATURE_COLUMNS)].astype(np.float32).to_numpy()
    model = _get_tf_model()
    preds = model.predict(X, verbose=0).reshape(-1)
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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    img = image.astype(np.float32)
    mean_val = float(np.mean(img))
    std_val = float(np.std(img))
    max_val = float(np.max(img))
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).reshape(-1)
    mode_val = float(int(np.argmax(hist)))
    return mean_val, std_val, max_val, mode_val


def _calculate_feature_averages(all_stats: List[List[Union[str, float]]]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not all_stats:
        return None, None, None, None
    return tuple(float(np.mean([row[i] for row in all_stats])) for i in range(1, 5))


def _process_difference_images(diff_dir: str, sample_number: str, stage_number: str, trial_number: str, grade_number: Optional[Union[str, float]] = None) -> List[List[Union[str, float]]]:
    rows: List[List[Union[str, float]]] = []
    for i in range(1, 9):
        diff_filename = cfg.make_difference_filename(sample_number, stage_number, trial_number, i)
        diff_path = os.path.join(diff_dir, diff_filename)
        stats_result = _calculate_image_stats(diff_path)
        if stats_result is None:
            log.warning("Could not read image: %s", diff_path)
            continue
        mean_val, std_val, max_val, mode_val = stats_result
        if grade_number is not None:
            rows.append([diff_filename, mean_val, std_val, max_val, mode_val, grade_number])
        else:
            rows.append([diff_filename, mean_val, std_val, max_val, mode_val])
    return rows


def _write_csv_with_headers(file_path: str, headers: Sequence[str], data: List[Sequence[Union[str, float]]], append_mode: bool = False) -> None:
    file_exists = os.path.exists(file_path)
    file_is_empty = (not file_exists) or os.path.getsize(file_path) == 0
    mode = "a" if append_mode and file_exists and not file_is_empty else "w"
    ensure_directory(os.path.dirname(file_path) or ".")
    with open(file_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w" or file_is_empty:
            writer.writerow(list(headers))
        writer.writerows(data)


def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


# -------- Grade coercion helpers (training) --------

def _coerce_grade(grade_number: Union[str, float, None], label: str) -> Optional[float]:
    """Parse and round a grade value to the nearest half-step; return None on failure."""
    if grade_number is None:
        return None
    try:
        g = float(str(grade_number).replace(",", "."))
        return round(g * 2) / 2.0
    except ValueError:
        log.error("Invalid %s: %s", label, grade_number)
        return None


def _restamp_grade(rows: List[List], new_grade: float) -> List[List]:
    """Return a copy of rows with the last column (Grade) replaced by new_grade."""
    return [row[:-1] + [new_grade] for row in rows]


# -------- Backend adapters --------

def _run_backend_for_grade(grade: str, csv_path: str, output_path: Optional[str]) -> str:
    """Run prediction for a single grade using that grade's model directory."""
    global LAST_BACKEND_USED
    if cfg.BACKEND == "tf":
        LAST_BACKEND_USED = "tf"
        return _tf_batch_predict_csv(csv_path, output_path)
    LAST_BACKEND_USED = "itanet_dll"
    return predict_from_csv(
        csv_path=csv_path,
        data_dir=cfg.get_itanet_run_dir(grade),
        dll_path=cfg.DLL_PATH,
        output_path=output_path,
        feature_cols=cfg.FEATURE_COLUMNS,
        clip_range=cfg.CLIP_RANGE,
        fls_dir=cfg.get_itanet_fls_dir(grade),
    )


def _predict_grade_from_features(grade: str, sample_key: str, mean_avg: float, std_avg: float, max_avg: float, mode_avg: float) -> Optional[float]:
    """Write a temp CSV, run recall for one grade, return the predicted value."""
    temp_csv = None
    preds_csv = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
            temp_csv = f.name
            w = csv.writer(f)
            w.writerow(list(FEATURE_HEADERS))
            w.writerow([sample_key, mean_avg, std_avg, max_avg, mode_avg])

        preds_csv = _run_backend_for_grade(grade, temp_csv, None)

        with open(preds_csv, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f), None)

        if not row or "Predicted_Grade" not in row:
            log.error("No predicted grade in predictions CSV for %s", grade)
            return None
        return float(row["Predicted_Grade"].strip())

    except Exception:
        log.exception("Prediction failed for grade=%s", grade)
        return None
    finally:
        _safe_unlink(temp_csv)
        _safe_unlink(preds_csv)


# -------- Public functions --------

def analyze_difference_images_and_predict_output(
    sample_number: str,
    stage_number: str,
    trial_number: str,
    selected_grades: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Analyze difference images and predict grades for all three grade types
    (pilling, matting, fuzzing) using independent ITA-Net networks.

    Returns a dict of {grade: float} for each selected grade,
    or None if feature extraction fails.  Individual grade predictions that
    fail are omitted from the dict so partial results are still usable.
    """
    if output_dir is None:
        output_dir = cfg.GRADING_RESULTS_DIR
    active_grades = list(selected_grades) if selected_grades is not None else list(cfg.GRADES)
    try:
        diff_dir = cfg.get_difference_dir(cfg.SUFFIX_GRADING)
        ensure_directory(output_dir)

        if not os.path.exists(diff_dir):
            log.error("Difference pictures directory not found: %s", diff_dir)
            return None

        all_stats = _process_difference_images(diff_dir, sample_number, stage_number, trial_number)
        if not all_stats:
            log.warning("No valid images found for analysis")
            return None

        mean_avg, std_avg, max_avg, mode_avg = _calculate_feature_averages(all_stats)
        if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in (mean_avg, std_avg, max_avg, mode_avg)):
            log.warning("Invalid averaged features")
            return None

        sample_key = f"{sample_number}-{stage_number}-{trial_number}"

        # Mock prediction path — returns synthetic grades without a trained .NET
        if cfg.MOCK_PREDICTION:
            try:
                stage_val = int(stage_number)
            except ValueError:
                stage_val = 0
            mock_base = max(1.0, min(5.0, 1.0 + stage_val / 2000.0 * 4.0))
            all_mock = {
                "pilling": round(mock_base * 2) / 2.0,
                "matting": round(max(1.0, mock_base - 0.5) * 2) / 2.0,
                "fuzzing": round(min(5.0, mock_base + 0.5) * 2) / 2.0,
            }
            grades: Dict[str, Optional[float]] = {g: all_mock[g] for g in active_grades if g in all_mock}
            log.info("mock predicted grades: %s", grades)
        else:
            grades = {}
            for grade in active_grades:
                grades[grade] = _predict_grade_from_features(
                    grade, sample_key, mean_avg, std_avg, max_avg, mode_avg
                )

        # Write analysis CSV — one per grade, appending prediction result
        all_stats_with_avg = all_stats + [["Average", mean_avg, std_avg, max_avg, mode_avg]]
        for grade in active_grades:
            prediction = grades.get(grade)
            if prediction is None:
                continue
            rows_for_csv = all_stats_with_avg + [
                [f"Grade ({grade})", prediction, "", "", ""],
                ["Backend", cfg.BACKEND, "", "", ""],
            ]
            analysis_csv = os.path.join(
                output_dir,
                f"{sample_number}-{stage_number}-{trial_number}-{grade}-analysis.csv",
            )
            _write_csv_with_headers(analysis_csv, list(FEATURE_HEADERS), rows_for_csv)
            log.info("Analysis saved: %s", analysis_csv)

        # Filter out None predictions before returning
        return {g: v for g, v in grades.items() if v is not None} or None

    except Exception:
        log.exception("analyze_difference_images_and_predict_output failed")
        return None


def analyze_difference_images(
    sample_number: str,
    stage_number: str,
    trial_number: str,
    grades: Dict[str, float],
    output_dir: Optional[str] = None,
) -> None:
    """
    Build CSV feature vectors for ITA-Net training.

    ``grades`` maps each active grade name to its already-validated float value,
    e.g. ``{"pilling": 2.5, "matting": 3.0}``.  One pair of CSVs is written per
    entry:
      - {grade}_per_image_features.csv
      - {grade}_averaged_features.csv
    """
    if output_dir is None:
        output_dir = cfg.TRAINING_FEATURES_DIR
    if not grades:
        log.error("No grades provided")
        return
    try:
        diff_dir = cfg.get_difference_dir(cfg.SUFFIX_TRAINING)
        ensure_directory(output_dir)

        if not os.path.exists(diff_dir):
            log.error("Difference pictures directory not found: %s", diff_dir)
            return

        # Feature extraction is the same for all grades — compute once
        first_grade_val = next(iter(grades.values()))
        all_stats = _process_difference_images(diff_dir, sample_number, stage_number, trial_number, first_grade_val)
        if not all_stats:
            log.warning("No valid images found for analysis")
            return

        mean_avg, std_avg, max_avg, mode_avg = _calculate_feature_averages(all_stats)
        sample_key = f"{sample_number}-{stage_number}-{trial_number}"

        def _write_grade_csvs(prefix: str, per_image_rows: List[List], grade_val: float) -> None:
            avg_row = [[sample_key, mean_avg, std_avg, max_avg, mode_avg, grade_val]]
            out_per = os.path.join(output_dir, f"{prefix}_per_image_features.csv")
            out_avg = os.path.join(output_dir, f"{prefix}_averaged_features.csv")
            _write_csv_with_headers(out_per, TRAIN_HEADERS, per_image_rows, append_mode=True)
            _write_csv_with_headers(out_avg, TRAIN_HEADERS, avg_row, append_mode=True)
            log.info("%s training rows appended to %s and %s", prefix, out_per, out_avg)

        for grade_name, grade_val in grades.items():
            rows = _restamp_grade(all_stats, grade_val)
            _write_grade_csvs(grade_name, rows, grade_val)

    except Exception:
        log.exception("analyze_difference_images failed")


def batch_predict_from_csv(csv_path: str, output_path: Optional[str] = None, custom_net_file: Optional[str] = None, custom_trn_file: Optional[str] = None) -> str:
    """Batch prediction from CSV using the pilling model (single-grade utility)."""
    return _run_backend_for_grade("pilling", csv_path, output_path)
