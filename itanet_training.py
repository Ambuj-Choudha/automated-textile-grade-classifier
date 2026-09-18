"""
ITA-Net training — convert a feature CSV and run the training DLL.

Operations:
    1. Convert the CSV into TrainingData.dat + TrainingGrades.dat.
    2. Archive the existing Neuronalesnetz.NET + .TRN.
    3. Call run_training_session(); the .NET is rewritten in place with the
       trained weights, ready for the next recall.

Usage:
    python itanet_training.py data/training_features/per_image_features.csv
    python itanet_training.py data/training_features/averaged_features.csv --net-type 2

Full workflow reference (DLL contract, byte formats, archive rationale, layout
constraints): docs/itanet/WORKFLOW.md.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from itanet_recall import (
    CREATE_NEW_NET,
    DecimalMode,
    ITANetFiles,
    ITANetFileWriter,
    LOAD_FROM_FILE,
    SHUFFLE_OFF,
    SHUFFLE_ON,
    change_directory,
    decimal_to_komma_punkt,
    default_dll_path,
    default_fls_dir,
    default_run_dir,
    format_number,
    load_itanet_dll,
)


# Size below which a Neuronalesnetz.NET is definitely the pristine topology-only
# file rather than a trained weights-bearing one. Mirrors the check in
# ITA-net-repo/experiment/recall/recall.py.
PRISTINE_NET_MAX_BYTES = 500


def _parse_cell(raw: str) -> float:
    """CSV cells may arrive with either separator; normalize before parsing."""
    return float(str(raw).strip().replace(",", "."))


def _fmt_value(raw: str, decimal: DecimalMode) -> str:
    return format_number(_parse_cell(raw), decimal)


def _fmt_grade(raw: str, decimal: DecimalMode) -> str:
    f = _parse_cell(raw)
    if f == int(f):
        return str(int(f))
    formatted = f"{f:g}"
    return formatted.replace(".", ",") if decimal == "comma" else formatted


def _read_csv(csv_path: Path) -> Tuple[List[str], List[List[str]]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{csv_path}: empty file")
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{csv_path}: no data rows")
    return header, rows


def _pick_columns(
    header: Sequence[str], sample_row: Sequence[str]
) -> Tuple[List[int], int]:
    grade_idx = next(
        (i for i, h in enumerate(header) if h.strip().lower() == "grade"), None
    )
    if grade_idx is None:
        raise ValueError("CSV is missing the required 'Grade' column for training")

    feature_idx: List[int] = []
    for i, v in enumerate(sample_row):
        if i == grade_idx:
            continue
        try:
            float(str(v).strip().replace(",", "."))
        except ValueError:
            continue
        feature_idx.append(i)

    if not feature_idx:
        raise ValueError("CSV has no numeric feature columns")
    return feature_idx, grade_idx


def _write_dat(path: Path, rows: List[List[str]], col_idx: Sequence[int],
               decimal: DecimalMode) -> None:
    n, ncols = len(rows), len(col_idx)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{n}\t{ncols}\r\n")
        for row in rows:
            f.write("\t".join(_fmt_value(row[i], decimal) for i in col_idx) + "\r\n")


def _write_grades(path: Path, rows: List[List[str]], grade_idx: int,
                  decimal: DecimalMode) -> None:
    n = len(rows)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{n}\t1\r\n")
        for row in rows:
            f.write(_fmt_grade(row[grade_idx], decimal) + "\r\n")


def _archive_existing_net(run_dir: Path, archive_dir: Path) -> Optional[Path]:
    """Snapshot .NET + .TRN into <archive_dir>/<UTC>/ if the .NET exists.

    Training rewrites the .NET in place with weights, so archiving is the
    only way back to the previous trained state. The .TRN goes with it
    because the hyperparameters (learning rate, epoch counts, etc.) are
    needed to reproduce the run — a .NET alone is not enough context.

    Returns the archive folder path (containing both files), or None if
    there was no .NET to archive.
    """
    net = run_dir / ITANetFiles.NET
    if not net.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_dir = archive_dir / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(net, dest_dir / net.name)

    trn = run_dir / ITANetFiles.TRN
    if trn.exists():
        shutil.copy2(trn, dest_dir / trn.name)
    else:
        print(f"[train] warning: no {ITANetFiles.TRN} alongside {net.name} to archive")

    return dest_dir


def _guard_net_matches_net_type(net_path: Path, net_type: int) -> None:
    """Fail fast if the .NET on disk is the wrong shape for `net_type`.

    CREATE_NEW_NET expects a pristine topology-only .NET (typically < 500
    bytes). LOAD_FROM_FILE expects a trained weights-bearing .NET. Passing
    the wrong one silently produces garbage because new_net / load_net read
    different on-disk formats.
    """
    size = net_path.stat().st_size
    if net_type == CREATE_NEW_NET and size > PRISTINE_NET_MAX_BYTES:
        raise RuntimeError(
            f"{net_path} is {size} bytes — looks weights-bearing, not pristine. "
            f"CREATE_NEW_NET requires the topology-only form. Restore a pristine "
            f".NET from models/itanet/archive/ or use --net-type 2 to continue "
            f"training from the current weights."
        )
    if net_type == LOAD_FROM_FILE and size <= PRISTINE_NET_MAX_BYTES:
        raise RuntimeError(
            f"{net_path} is only {size} bytes — looks pristine (topology only), "
            f"not weights-bearing. LOAD_FROM_FILE expects the format written by "
            f"a previous training run. Use --net-type 1 to train fresh weights, "
            f"or restore a trained .NET from models/itanet/archive/."
        )


def csv_to_training_dats(
    csv_path: Path,
    data_out: Path,
    grades_out: Path,
    decimal: DecimalMode = "comma",
) -> Tuple[int, int]:
    """Write TrainingData.dat + TrainingGrades.dat. Returns (n_samples, n_features)."""
    header, rows = _read_csv(csv_path)
    feature_idx, grade_idx = _pick_columns(header, rows[0])
    _write_dat(data_out, rows, feature_idx, decimal)
    _write_grades(grades_out, rows, grade_idx, decimal)
    return len(rows), len(feature_idx)


def train_from_csv(
    csv_path: str,
    *,
    grade: str = "pilling",
    net_type: int = CREATE_NEW_NET,
    decimal: DecimalMode = "comma",
    shuffle: int = SHUFFLE_ON,
    backup: bool = True,
) -> Path:
    """End-to-end: CSV -> .dat files -> run training -> return trained .NET path.

    ``grade`` selects which ITA-Net network to train (pilling / matting / fuzzing).
    Each grade has its own run/ + data/ directory under models/itanet/<grade>/.

    ``net_type=CREATE_NEW_NET`` (default) requires the .NET to still be in
    pristine topology-only form. ``net_type=LOAD_FROM_FILE`` continues
    training from the current weights.

    When ``backup`` is True (default), the existing .NET + .TRN are
    snapshotted to ``models/itanet/archive/<grade>/<UTC>/`` first.
    """
    from app.settings import config as cfg
    run_dir_p = Path(cfg.get_itanet_run_dir(grade))
    fls_dir_p = Path(cfg.get_itanet_fls_dir(grade))
    dll_path_s = cfg.DLL_PATH
    archive_dir_p = Path(cfg.get_itanet_archive_dir(grade))
    csv_path_p = Path(csv_path).resolve()

    if not csv_path_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path_p}")

    net_path = run_dir_p / ITANetFiles.NET
    trn_path = run_dir_p / ITANetFiles.TRN
    if not net_path.exists():
        raise FileNotFoundError(f"Missing network structure file: {net_path}")
    if not trn_path.exists():
        raise FileNotFoundError(f"Missing training schedule file: {trn_path}")

    _guard_net_matches_net_type(net_path, net_type)

    data_out = run_dir_p / ITANetFiles.TRAIN_INPUT
    grades_out = run_dir_p / ITANetFiles.TRAIN_TARGET
    n, k = csv_to_training_dats(csv_path_p, data_out, grades_out, decimal=decimal)
    print(
        f"[train] converted {csv_path_p.name}: {n} samples x {k} features -> "
        f"{data_out.name}, {grades_out.name}"
    )

    fls_dir_p.mkdir(parents=True, exist_ok=True)
    training_fls = fls_dir_p / ITANetFiles.TRAINING_FLS
    ITANetFileWriter.write_filelist(
        str(training_fls),
        [
            ITANetFiles.TRAIN_INPUT,
            ITANetFiles.TRAIN_TARGET,
            ITANetFiles.NET,
            ITANetFiles.TRN,
        ],
    )

    if backup:
        archived = _archive_existing_net(run_dir_p, archive_dir_p)
        if archived is not None:
            print(f"[train] archived previous .NET + .TRN to {archived}")
        else:
            print("[train] no existing .NET to archive")

    lib = load_itanet_dll(dll_path_s)
    komma_punkt = decimal_to_komma_punkt(decimal)

    print(f"[train] running run_training_session(net_type={net_type}, "
          f"komma_punkt={komma_punkt}, shuffle={shuffle}) ...")
    with change_directory(str(run_dir_p)):
        rc = int(lib.run_training_session(net_type, komma_punkt, shuffle))

    if rc != 0:
        raise RuntimeError(f"ITANet training failed with code: {rc}")

    if not net_path.exists():
        raise RuntimeError(f"Training returned 0 but {net_path} was not written")

    print(f"[train] wrote {net_path}")
    return net_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ITA-Net on a feature CSV (with 'Grade' column). "
                    "Reads/writes under models/itanet/ (run/ + data/ + archive/).",
    )
    parser.add_argument(
        "csv_path",
        help="Feature CSV with Mean/Std/Max/Mode + Grade columns "
             "(e.g. data/training_features/pilling_per_image_features.csv).",
    )
    parser.add_argument(
        "--grade", default="pilling",
        choices=["pilling", "matting", "fuzzing"],
        help="Which grade network to train (default: pilling).",
    )
    parser.add_argument(
        "--net-type", type=int, default=CREATE_NEW_NET,
        choices=[CREATE_NEW_NET, LOAD_FROM_FILE],
        help="1=CREATE_NEW_NET (fresh weights, requires pristine .NET), "
             "2=LOAD_FROM_FILE (continue training from current weights).",
    )
    parser.add_argument(
        "--decimal", choices=("comma", "point"), default="comma",
        help="Decimal format written into TrainingData.dat "
             "(sets komma_punkt=0 for 'comma', 1 for 'point').",
    )
    parser.add_argument(
        "--shuffle", type=int, default=SHUFFLE_ON,
        choices=[SHUFFLE_OFF, SHUFFLE_ON],
        help="1=shuffle patterns (default), 0=preserve order.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip archiving the existing Neuronalesnetz.NET + .TRN before training.",
    )
    args = parser.parse_args()

    train_from_csv(
        csv_path=args.csv_path,
        grade=args.grade,
        net_type=args.net_type,
        decimal=args.decimal,
        shuffle=args.shuffle,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
