import os
import re
import pandas as pd
from app.reporting.pdf_report import generate_pilling_report
from app.helpers.utils import ensure_directory


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


def _scan_stage_results_for_sample(sample_number: str) -> dict[int, list[str]]:
    """
    Scan output/grading_results for this sample and return
    {stage_rubs: int -> [trial_grades]}.
    Pattern: {sample}-{stage}-{trial}-analysis.csv
    """
    results: dict[int, list[str]] = {}
    try:
        dirpath = os.path.join("output", "grading_results")
        if not os.path.isdir(dirpath):
            return results
        pattern = re.compile(rf"^{re.escape(sample_number)}-(\d+)-(\d+)-analysis\.csv$")
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
    except Exception as e:
        print(f"Error scanning stage results for sample {sample_number}: {e}")
        return {}


def export_results(
    sample_number: str,
    stage_number: str,
    load_weight: str | float | int,
    operator_name: str,
    trial_number: str = "1",
    grades: dict | None = None,
) -> bool:
    """
    Export results to a single combined PDF per sample.
    Builds the pilling table from all available grading-result CSVs, then
    appends matting/fuzzing columns if grades are provided.
    """
    try:
        stage_trials_map = _scan_stage_results_for_sample(sample_number)
        if not stage_trials_map:
            print(f"No analysis CSVs found for sample {sample_number} in output/grading_results/")
            return False

        results_by_rubs = {}
        for rub, grade_list in stage_trials_map.items():
            n = len(grade_list)
            if n == 1:
                results_by_rubs[rub] = [grade_list[0], "NA", "NA", grade_list[0]]
            elif n == 2:
                avg = sum(float(g) for g in grade_list) / n
                results_by_rubs[rub] = [grade_list[0], grade_list[1], "NA", f"{avg:.2f}"]
            else:
                avg = sum(float(g) for g in grade_list) / n
                results_by_rubs[rub] = [grade_list[0], grade_list[1], grade_list[2], f"{avg:.2f}"]

        reports_dir = "reports"
        ensure_directory(reports_dir)
        clean_name = "".join(c for c in operator_name if c.isalnum() or c in ('-', '_')).strip() or "Unknown"
        pdf_filepath = os.path.join(reports_dir, f"{sample_number}-{clean_name}-report.pdf")

        matting_by_rubs = None
        fuzzing_by_rubs = None
        if grades:
            if (v := grades.get("matting")) is not None:
                matting_by_rubs = {rub: [str(v), "NA", "NA", str(v)] for rub in results_by_rubs}
            if (v := grades.get("fuzzing")) is not None:
                fuzzing_by_rubs = {rub: [str(v), "NA", "NA", str(v)] for rub in results_by_rubs}

        ok = generate_pilling_report(
            sample_number=sample_number,
            stage_number=stage_number,
            load_weight_g=load_weight,
            operator_name=operator_name,
            results_by_rubs=results_by_rubs,
            matting_by_rubs=matting_by_rubs,
            fuzzing_by_rubs=fuzzing_by_rubs,
            abradant="Similar Fabric",
            output_path=pdf_filepath,
            rub_levels=list(results_by_rubs.keys()),
            trial_number=trial_number,
        )

        if ok:
            print(f"PDF report saved to: {pdf_filepath}")
            return True
        print("Error during PDF export: failed to generate PDF.")
        return False

    except Exception as e:
        print(f"Error during PDF export: {e}")
        return False
