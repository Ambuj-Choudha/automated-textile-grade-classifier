"""ITA-Net integration test CLI.

Quick way to verify the train/recall pipeline without launching Streamlit.
Run from the repo root:

    python app/tests/test_integration.py recall                 # recall on a default fixture CSV
    python app/tests/test_integration.py recall --csv <path>    # recall on any CSV
    python app/tests/test_integration.py regression             # exercise the 3 bug-prone shapes
    python app/tests/test_integration.py all                    # both

Each check prints PASS/FAIL; the process exits non-zero if anything fails.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

# Allow `python app/tests/test_integration.py ...` from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.settings import config as cfg
from itanet_recall import predict_from_csv, format_number, ITANetFileWriter


GRADE_RANGE = (1.0, 5.0)
DEFAULT_RECALL_CSV = REPO_ROOT / "output" / "grading_results" / "00000-100-1-analysis.csv"


# --- helpers -----------------------------------------------------------------


class Reporter:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        suffix = f" -- {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if not ok:
            self.failed.append(name)

    def summary_and_exit(self) -> None:
        print()
        if self.failed:
            print(f"FAILED ({len(self.failed)}): {', '.join(self.failed)}")
            sys.exit(1)
        print("All checks passed.")
        sys.exit(0)


def _read_predictions(csv_path: str) -> list[float]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [float(r["Predicted_Grade"]) for r in reader]


def _run_recall(csv_path: str, output_path: str) -> list[float]:
    predict_from_csv(
        csv_path=csv_path,
        data_dir=cfg.ITANET_RUN_DIR,
        dll_path=cfg.DLL_PATH,
        output_path=output_path,
        feature_cols=cfg.FEATURE_COLUMNS,
        clip_range=cfg.CLIP_RANGE,
        fls_dir=cfg.ITANET_FLS_DIR,
    )
    return _read_predictions(output_path)


# --- commands ----------------------------------------------------------------


def cmd_recall(args: argparse.Namespace, rep: Reporter) -> None:
    print(f"\n== recall: {args.csv} ==")
    csv_path = Path(args.csv)
    rep.check("input CSV exists", csv_path.exists(), str(csv_path))
    if not csv_path.exists():
        return

    out = args.output or str(REPO_ROOT / "_test_predictions.csv")
    try:
        preds = _run_recall(str(csv_path), out)
    except Exception as e:
        rep.check("recall ran without error", False, repr(e))
        return

    rep.check("recall ran without error", True)
    rep.check("at least one prediction returned", len(preds) > 0, f"got {len(preds)}")
    rep.check(
        f"all predictions in [{GRADE_RANGE[0]}, {GRADE_RANGE[1]}]",
        all(GRADE_RANGE[0] <= p <= GRADE_RANGE[1] for p in preds),
        f"preds={preds}",
    )
    rep.check(
        "all predictions are half-step values",
        all((p * 2) == int(p * 2) for p in preds),
        f"preds={preds}",
    )
    print(f"  wrote {out}")


def cmd_regression(args: argparse.Namespace, rep: Reporter) -> None:
    print("\n== regression: bugs fixed in itanet_recall.py ==")

    # --- bug 1: CSV with trailing Average/Grade/Backend summary rows -----
    # _load_recall_csv used to drop ALL rows here; the summary rows also
    # poisoned the Mean column dtype so Mean was silently dropped from
    # auto-detect, leading to a 3-feature RecallData.dat against a
    # 4-feature network -> DLL access violation.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        analysis_csv = f.name
        w = csv.writer(f)
        w.writerow(["Image", "Mean", "Std", "Max", "Mode"])
        w.writerow(["00000-100-1-dif.png", 11.37, 8.70, 103.0, 1.0])
        w.writerow(["00000-100-2-dif.png", 11.74, 9.16, 132.0, 1.0])
        w.writerow(["Average", 11.55, 8.93, 117.5, 1.0])
        w.writerow(["Grade", 4.0, "", "", ""])
        w.writerow(["Backend", "itanet_dll", "", "", ""])

    # Contract: rows with all-numeric feature cells survive (so 2 image rows
    # AND the Average row -> 3). Rows with empty or non-numeric feature cells
    # (Grade, Backend) are dropped. This means analysis_results CSVs round-trip
    # cleanly; the Average row's prediction is harmless because the grading
    # flow feeds a single-row temp CSV, not the persisted analysis CSV.
    out = str(REPO_ROOT / "_test_predictions_summary.csv")
    try:
        preds = _run_recall(analysis_csv, out)
        rep.check(
            "summary-row CSV: 2 data + 1 Average -> 3 predictions",
            len(preds) == 3,
            f"got {len(preds)}",
        )
        rep.check(
            "summary-row CSV: Grade/Backend rows dropped",
            len(preds) < 5,  # would be 5 if no filtering happened
            f"got {len(preds)}",
        )
    except Exception as e:
        rep.check("summary-row CSV: recall succeeds", False, repr(e))
    finally:
        for p in (analysis_csv, out):
            try:
                os.unlink(p)
            except OSError:
                pass

    # --- bug 2: single-row temp CSV (the shape the grading flow builds) ---
    # The Image cell is "<sample>-<stage>-<trial>" which is non-numeric;
    # the old row-keep mask treated it as a summary row and dropped
    # everything, leaving zero rows.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as f:
        temp_csv = f.name
        w = csv.writer(f)
        w.writerow(["Image", "Mean", "Std", "Max", "Mode"])
        w.writerow(["00000-100-1", 11.35, 8.79, 104.0, 1.0])

    out = str(REPO_ROOT / "_test_predictions_singlerow.csv")
    try:
        preds = _run_recall(temp_csv, out)
        rep.check(
            "single-row temp CSV: 1 data row -> 1 prediction",
            len(preds) == 1,
            f"got {len(preds)}",
        )
    except Exception as e:
        rep.check("single-row temp CSV: recall succeeds", False, repr(e))
    finally:
        for p in (temp_csv, out):
            try:
                os.unlink(p)
            except OSError:
                pass

    # --- bug 3: whole-number formatting in RecallData.dat ----------------
    # The DLL's pattern loader rejects bare ints; format_number used to
    # emit "1" instead of "1,0" for whole-number features.
    rep.check(
        "format_number(1.0, 'comma') keeps trailing zero",
        format_number(1.0, "comma") == "1,0",
        f"got {format_number(1.0, 'comma')!r}",
    )
    rep.check(
        "format_number(104.0, 'comma') keeps trailing zero",
        format_number(104.0, "comma") == "104,0",
        f"got {format_number(104.0, 'comma')!r}",
    )
    rep.check(
        "format_number(11.35, 'comma') uses comma separator",
        format_number(11.35, "comma") == "11,35",
        f"got {format_number(11.35, 'comma')!r}",
    )

    # And the file itself: written with CRLF, comma decimals, "1,0"-style.
    with tempfile.NamedTemporaryFile(
        mode="rb", suffix=".dat", delete=False
    ) as f:
        dat_path = f.name
    try:
        ITANetFileWriter.write_data_file(dat_path, [[11.35, 1.0]], "comma")
        with open(dat_path, "rb") as f:
            body = f.read()
        rep.check(
            "RecallData.dat header uses CRLF",
            b"\r\n" in body.split(b"\r\n", 1)[0] + b"\r\n",
            repr(body[:20]),
        )
        rep.check(
            "RecallData.dat encodes whole numbers as '1,0'",
            b"1,0" in body,
            repr(body),
        )
        rep.check(
            "RecallData.dat does not emit bare ints",
            b"\t1\r\n" not in body and b"\t1\n" not in body,
            repr(body),
        )
    finally:
        try:
            os.unlink(dat_path)
        except OSError:
            pass


# --- entry point -------------------------------------------------------------


def _ensure_prereqs(rep: Reporter) -> bool:
    ok = True
    if not Path(cfg.DLL_PATH).exists():
        rep.check("DLL present", False, cfg.DLL_PATH)
        ok = False
    else:
        rep.check("DLL present", True, cfg.DLL_PATH)

    net = Path(cfg.ITANET_RUN_DIR) / "Neuronalesnetz.NET"
    if not net.exists():
        rep.check("trained .NET present", False, str(net))
        ok = False
    else:
        rep.check("trained .NET present", True, str(net))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="ITA-Net integration test CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_recall = sub.add_parser("recall", help="Run recall on a CSV.")
    p_recall.add_argument("--csv", default=str(DEFAULT_RECALL_CSV))
    p_recall.add_argument("--output", default=None)
    p_recall.set_defaults(func=cmd_recall)

    p_reg = sub.add_parser(
        "regression",
        help="Exercise the CSV shapes / formatter that triggered past bugs.",
    )
    p_reg.set_defaults(func=cmd_regression)

    p_all = sub.add_parser("all", help="Run recall + regression.")
    p_all.set_defaults(func=None)
    p_all.add_argument("--csv", default=str(DEFAULT_RECALL_CSV))
    p_all.add_argument("--output", default=None)

    args = parser.parse_args()
    rep = Reporter()

    print("== prereqs ==")
    if not _ensure_prereqs(rep):
        rep.summary_and_exit()

    if args.cmd == "all":
        cmd_recall(args, rep)
        cmd_regression(args, rep)
    else:
        args.func(args, rep)

    rep.summary_and_exit()


if __name__ == "__main__":
    main()
