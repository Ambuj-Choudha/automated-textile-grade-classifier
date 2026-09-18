import logging
import os
import re

import pandas as pd

from app.helpers.utils import ensure_directory
from app.reporting.pdf_report import generate_pilling_report
from app.settings import config as cfg

log = logging.getLogger(__name__)


def _process_csv_data(df_full: pd.DataFrame) -> tuple:
    grade_row_index = None
    grade_number = None
    for idx, row in df_full.iterrows():
        if str(row[0]).strip().lower() == "grade":
            grade_row_index = idx
            grade_number = row[1]
            break
    if grade_row_index is None:
        return None, None, False
    df = df_full.iloc[:grade_row_index]
    if len(df) > 0:
        df.columns = df.iloc[0]
        df = df[1:]
    return df, grade_number, True


def _scan_stage_results_for_sample(sample_number: str, grade: str) -> dict[int, list[str]]:
    """Scan output/grading_results for analysis CSVs matching sample+grade and
    return {stage_rubs: int -> [trial_grades]}.

    Pattern: {sample}-{stage}-{trial}-{grade}-analysis.csv
    """
    results: dict[int, list[str]] = {}
    try:
        dirpath = cfg.GRADING_RESULTS_DIR
        if not os.path.isdir(dirpath):
            return results
        pattern = re.compile(
            rf"^{re.escape(sample_number)}-(\d+)-(\d+)-{re.escape(grade)}-analysis\.csv$"
        )
        for fname in os.listdir(dirpath):
            m = pattern.match(fname)
            if not m:
                continue
            stage_rubs = int(m.group(1))
            fpath = os.path.join(dirpath, fname)
            try:
                df_full = pd.read_csv(fpath, header=None)
                _, grade_number, success = _process_csv_data(df_full)
                if success and grade_number is not None and str(grade_number).strip():
                    results.setdefault(stage_rubs, []).append(str(grade_number).strip())
            except Exception:
                continue
        for stage_rubs in results:
            results[stage_rubs].sort()
        return dict(sorted(results.items()))
    except Exception:
        log.exception("Error scanning stage results for sample=%s grade=%s",
                      sample_number, grade)
        return {}


def _build_by_rubs(stage_trials_map: dict[int, list[str]]) -> dict[int, list[str]]:
    """Turn {rub: [grades]} into the 4-column {rub: [r1, r2, r3, avg]} shape
    consumed by the PDF table."""
    by_rubs: dict[int, list[str]] = {}
    for rub, grade_list in stage_trials_map.items():
        n = len(grade_list)
        if n == 1:
            by_rubs[rub] = [grade_list[0], "NA", "NA", grade_list[0]]
        elif n == 2:
            avg = sum(float(g) for g in grade_list) / n
            by_rubs[rub] = [grade_list[0], grade_list[1], "NA", f"{avg:.2f}"]
        else:
            avg = sum(float(g) for g in grade_list[:3]) / 3
            by_rubs[rub] = [grade_list[0], grade_list[1], grade_list[2], f"{avg:.2f}"]
    return by_rubs


def export_results(
    sample_number: str,
    stage_number: str,
    load_weight: str | float | int,
    operator_name: str,
    trial_number: str = "1",
) -> bool:
    """Export a combined PDF for a sample, one column-group per grade with data.

    Each grade (pilling / matting / fuzzing) is populated from its own set of
    analysis CSVs on disk. A grade with no CSVs is omitted from the report.
    Returns True on success; False if no grade has any results.
    """
    try:
        per_grade_maps: dict[str, dict[int, list[str]]] = {}
        for grade in cfg.GRADES:
            stage_map = _scan_stage_results_for_sample(sample_number, grade)
            if stage_map:
                per_grade_maps[grade] = _build_by_rubs(stage_map)

        if not per_grade_maps:
            log.warning("No analysis CSVs found for sample %s in %s",
                        sample_number, cfg.GRADING_RESULTS_DIR)
            return False

        # Pilling drives the primary results_by_rubs; the other grades are
        # optional columns. If pilling has no data, fall back to whichever
        # grade did — union the rub levels so no row is dropped.
        all_rubs = sorted({rub for m in per_grade_maps.values() for rub in m})
        results_by_rubs = per_grade_maps.get("pilling") or {}
        for rub in all_rubs:
            results_by_rubs.setdefault(rub, ["NA", "NA", "NA", "NA"])

        reports_dir = cfg.REPORTS_DIR
        ensure_directory(reports_dir)
        clean_name = "".join(c for c in operator_name if c.isalnum() or c in ("-", "_")).strip() or "Unknown"
        pdf_filepath = os.path.join(reports_dir, f"{sample_number}-{clean_name}-report.pdf")

        ok = generate_pilling_report(
            sample_number=sample_number,
            stage_number=stage_number,
            load_weight_g=load_weight,
            operator_name=operator_name,
            results_by_rubs=results_by_rubs,
            matting_by_rubs=per_grade_maps.get("matting"),
            fuzzing_by_rubs=per_grade_maps.get("fuzzing"),
            abradant="Similar Fabric",
            output_path=pdf_filepath,
            rub_levels=all_rubs,
            trial_number=trial_number,
        )

        if ok:
            log.info("PDF report saved to: %s", pdf_filepath)
            return True
        log.error("PDF export: generate_pilling_report returned False")
        return False

    except Exception:
        log.exception("Error during PDF export")
        return False
