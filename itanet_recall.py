"""
ITA-Net recall — grade a batch of samples via the trained DLL.

Public entry points:
    predict_from_csv(csv_path, data_dir, dll_path, ...) -> str
        End-to-end: read features from CSV, run recall, write predictions CSV.
    ITANetManager(run_dir, dll_path, fls_dir)
        Lower-level manager if you already have a feature matrix in memory.

Companion module: itanet_training.py handles the training path.

DLL contract (ITA-net-repo docs/ITANET.md):
    int run_recall_session(int net_type, int komma_punkt, int shuffle)

The DLL hardcodes `..\\data\\recall.fls`, so the caller must chdir into a
folder (the "run dir") that has `..\\data\\` as a sibling before invoking.
"""

from __future__ import annotations
from typing import Sequence, Literal, Tuple, Optional
import logging
import os
import re
import argparse
import contextlib
import ctypes
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


DecimalMode = Literal["point", "comma"]


# Enum values mirrored from ITA-net-repo INCLUDE/HAUPTDEF.H.
CREATE_NEW_NET = 1
LOAD_FROM_FILE = 2
KOMMA_ZU_PUNKT = 0   # input has comma decimals; DLL converts on read/write
PUNKT_ZU_KOMMA = 1   # input already uses '.'; DLL skips conversion
SHUFFLE_OFF = 0
SHUFFLE_ON = 1


class ITANetFiles:
    NET = "Neuronalesnetz.NET"
    TRN = "Neuronalesnetz.TRN"
    TRAIN_INPUT = "TrainingData.dat"
    TRAIN_TARGET = "TrainingGrades.dat"
    RECALL_INPUT = "RecallData.dat"
    RECALL_OUTPUT = "Recall_output.dat"
    RECALL_FLS = "recall.fls"
    TRAINING_FLS = "training.fls"


@contextlib.contextmanager
def change_directory(path: str):
    """Context manager to temporarily change working directory."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def format_number(value: float, decimal: DecimalMode) -> str:
    # Whole numbers keep a trailing ".0" / ",0" — the DLL's pattern loader rejects
    # bare ints in feature data (manifests as an access violation in run_recall_session).
    if float(value) == int(value):
        formatted = f"{int(value)}.0"
    else:
        formatted = f"{value:g}"
    return formatted.replace('.', ',') if decimal == "comma" else formatted


def decimal_to_komma_punkt(decimal: DecimalMode) -> int:
    """Map the Python-side decimal style to the DLL's komma_punkt flag."""
    return KOMMA_ZU_PUNKT if decimal == "comma" else PUNKT_ZU_KOMMA


# Loaded DLLs are cached by absolute path so a Streamlit session doesn't open .dll everytime to bind the args
_dll_cache: dict[str, ctypes.CDLL] = {}


def load_itanet_dll(dll_path: str) -> ctypes.CDLL:
    """Load itanet.dll and bind argtypes/restype for both session entrypoints.

    Shared by recall (this module) and training (itanet_training.py); both
    exports have the same (int, int, int) -> int signature. Result is cached
    by dll_path.
    """
    key = os.path.abspath(dll_path)
    cached = _dll_cache.get(key)
    if cached is not None:
        return cached

    if not os.path.exists(dll_path):
        raise FileNotFoundError(
            f"ITANet DLL not found at {dll_path} — build it (see "
            f"ITA-net-repo/build.py) and copy to the run dir."
        )
    lib = ctypes.CDLL(dll_path)
    for fn_name in ("run_training_session", "run_recall_session"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_int
    _dll_cache[key] = lib
    return lib


class ITANetFileWriter:
    @staticmethod
    def write_data_file(path: str, matrix: list[list[float]], decimal: DecimalMode):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(f"{len(matrix)}\t{len(matrix[0])}\r\n")
            for row in matrix:
                f.write("\t".join(format_number(val, decimal) for val in row) + "\r\n")

    @staticmethod
    def write_filelist(path: str, files: list[str]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for filename in files:
                f.write(filename.strip() + "\n")


_HEADER_RE = re.compile(r"\d+\s+\d+")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:[.,]\d*)?|\.\d+)")
_NAN_RE = re.compile(r"nan", re.IGNORECASE)


class ITANetFileReader:
    @staticmethod
    def read_recall_output(path: str) -> list[float]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return []

        data_lines = lines[1:] if _HEADER_RE.fullmatch(lines[0]) else lines

        values: list[float] = []
        nan_hits: list[int] = []
        for i, line in enumerate(data_lines):
            if _NAN_RE.search(line):
                nan_hits.append(i)
                continue
            match = _NUMBER_RE.search(line)
            if match:
                try:
                    values.append(float(match.group(0).replace(",", ".")))
                except ValueError:
                    continue

        if nan_hits:
            raise RuntimeError(
                f"Recall output at {path} contains NaN on {len(nan_hits)} of "
                f"{len(data_lines)} lines (first at row {nan_hits[0] + 1}) — "
                f"trained .NET has diverged weights. Re-run training with saner "
                f".TRN hyperparameters or restore an archived .NET from "
                f"models/itanet/archive/."
            )
        return values


class ITANetManager:
    """High-level manager for ITANet recall.

    ``run_dir`` is the folder the DLL cd's into (holds .NET + .dat files for
    the active grade). The DLL binary itself lives at ``dll_path`` and is
    loaded by absolute path. ``fls_dir`` is where recall.fls lives — must be a
    sibling of run_dir so the DLL finds it at ``..\\data\\recall.fls``.
    """

    def __init__(self, run_dir: str, dll_path: str, fls_dir: Optional[str] = None):
        # Fail fast on a bad DLL path so a typo doesn't get past
        # prepare_recall_session (which writes files) before load_itanet_dll
        # raises further down. Also verified by test_missing_dll_raises_before_write.
        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"ITANet DLL not found at {dll_path} — build it (see "
                f"ITA-net-repo/build.py) and copy to the run dir."
            )

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.fls_dir = Path(fls_dir) if fls_dir else self.run_dir.parent / "data"
        self.fls_dir.mkdir(parents=True, exist_ok=True)

        self.dll_path = dll_path

    def prepare_recall_session(self, features_matrix: list[list[float]], decimal: DecimalMode = "comma"):
        recall_input_path = self.run_dir / ITANetFiles.RECALL_INPUT
        ITANetFileWriter.write_data_file(str(recall_input_path), features_matrix, decimal)

        filelist_path = self.fls_dir / ITANetFiles.RECALL_FLS
        files = [ITANetFiles.RECALL_INPUT, ITANetFiles.RECALL_OUTPUT,
                 ITANetFiles.TRAIN_INPUT, ITANetFiles.TRAIN_TARGET, ITANetFiles.NET]
        ITANetFileWriter.write_filelist(str(filelist_path), files)

        log.info("Recall session prepared: %d samples", len(features_matrix))

    def run_recall(
        self,
        *,
        net_type: int = LOAD_FROM_FILE,
        decimal: DecimalMode = "comma",
        shuffle: int = SHUFFLE_OFF,
    ) -> list[float]:
        lib = load_itanet_dll(self.dll_path)
        komma_punkt = decimal_to_komma_punkt(decimal)

        with change_directory(str(self.run_dir)):
            result = int(lib.run_recall_session(net_type, komma_punkt, shuffle))

        if result != 0:
            raise RuntimeError(f"ITANet recall failed with code: {result}")

        output_path = self.run_dir / ITANetFiles.RECALL_OUTPUT
        predictions = ITANetFileReader.read_recall_output(str(output_path))

        log.info("Recall completed: %d predictions", len(predictions))
        return predictions


def _load_recall_csv(
    csv_path: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """Load a recall CSV per the ITA-Net contract.

    - Drops the first column if it's non-numeric (e.g. ``Image`` filename).
    - Ignores a ``Grade`` column if present (case-insensitive).
    - Drops any row that has a non-numeric cell in any remaining column —
      this cleans out the trailing ``Average`` / ``Grade`` / ``Backend``
      summary rows that analysis CSVs append.
    - Surviving numeric columns, in CSV order, are the features.
    """
    df = pd.read_csv(csv_path)

    first_col = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first_col]):
        df = df.drop(columns=[first_col])

    drop_cols = [c for c in df.columns if c.strip().lower() == "grade"]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    coerced = df.apply(pd.to_numeric, errors="coerce")
    keep = coerced.notna().all(axis=1)
    df = coerced[keep].reset_index(drop=True)

    if feature_cols is None:
        feature_cols = list(df.columns)
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {csv_path}")

    feature_df = df[list(feature_cols)].astype(float)
    return feature_df, list(feature_cols)


def predict_from_csv(
    csv_path: str,
    data_dir: str,
    dll_path: str,
    output_path: str = None,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    clip_range: Tuple[float, float] = (1.0, 5.0),
    decimal: DecimalMode = "comma",
    fls_dir: Optional[str] = None,
    net_type: int = LOAD_FROM_FILE,
    shuffle: int = SHUFFLE_OFF,
) -> str:
    """Complete prediction pipeline from a CSV file.

    ``data_dir`` is the DLL's run dir for the active grade (holds .NET + .dat
    files). The DLL binary is at ``dll_path`` (shared across grades, loaded by
    absolute path). ``fls_dir`` is where recall.fls is written; defaults to
    ``<data_dir>/../data`` to match the DLL's hardcoded `..\\data\\recall.fls`
    lookup.
    """
    feature_df, used_cols = _load_recall_csv(csv_path, feature_cols)
    features_matrix = feature_df.values.tolist()

    if not features_matrix:
        raise ValueError(f"No data rows found in {csv_path}")

    manager = ITANetManager(data_dir, dll_path, fls_dir=fls_dir)
    manager.prepare_recall_session(features_matrix, decimal=decimal)
    predictions = manager.run_recall(net_type=net_type, decimal=decimal, shuffle=shuffle)

    if len(predictions) != len(feature_df):
        raise ValueError(
            f"Prediction count mismatch: {len(predictions)} != {len(feature_df)}"
        )

    if clip_range is not None:
        low, high = clip_range
        processed_preds = [max(low, min(high, round(p * 2) / 2.0)) for p in predictions]
    else:
        processed_preds = [round(p * 2) / 2.0 for p in predictions]

    feature_df["Predicted_Grade"] = processed_preds

    if output_path is None:
        base, _ = os.path.splitext(csv_path)
        output_path = f"{base}_predictions.csv"

    feature_df.to_csv(output_path, index=False)
    log.info("Wrote predictions (%s) -> %s", used_cols, output_path)
    return output_path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def default_dll_path() -> str:
    return str(_project_root() / "models" / "itanet" / "common" / "itanet.dll")


def _grade_run_dir(grade: str) -> str:
    return str(_project_root() / "models" / "itanet" / grade / "run")


def _grade_fls_dir(grade: str) -> str:
    return str(_project_root() / "models" / "itanet" / grade / "data")


def main():
    parser = argparse.ArgumentParser(
        description="Run ITA-Net recall on a CSV of features. "
                    "Reads/writes under models/itanet/<grade>/ (run/ + data/).",
    )
    parser.add_argument("csv_path", help="Input CSV (e.g. output/grading_results/*.csv).")
    parser.add_argument("--grade", required=True,
                        choices=["pilling", "matting", "fuzzing"],
                        help="Which grade network to use (required — recall is per-grade).")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--feature-cols", nargs="+", default=None,
                        help="Explicit feature column names (default: auto-detect numeric).")
    parser.add_argument("--no-clip", action="store_true",
                        help="Disable clipping predictions to [1.0, 5.0].")
    parser.add_argument("--decimal", choices=("comma", "point"), default="comma",
                        help="Decimal format written into RecallData.dat "
                             "(sets komma_punkt=0 for 'comma', 1 for 'point').")
    parser.add_argument("--net-type", type=int, default=LOAD_FROM_FILE,
                        choices=[CREATE_NEW_NET, LOAD_FROM_FILE],
                        help="2=LOAD_FROM_FILE (use trained weights — default), "
                             "1=CREATE_NEW_NET (fresh random weights).")
    parser.add_argument("--shuffle", type=int, default=SHUFFLE_OFF,
                        choices=[SHUFFLE_OFF, SHUFFLE_ON],
                        help="0=preserve order (default), 1=shuffle.")
    args = parser.parse_args()

    predict_from_csv(
        csv_path=args.csv_path,
        data_dir=_grade_run_dir(args.grade),
        dll_path=default_dll_path(),
        output_path=args.output,
        feature_cols=args.feature_cols,
        clip_range=None if args.no_clip else (1.0, 5.0),
        decimal=args.decimal,
        fls_dir=_grade_fls_dir(args.grade),
        net_type=args.net_type,
        shuffle=args.shuffle,
    )


if __name__ == "__main__":
    main()
