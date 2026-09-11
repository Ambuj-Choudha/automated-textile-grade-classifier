import os
import pathlib
from typing import Union, List
import pandas as pd
from fpdf import FPDF
import streamlit as st
from app.reporting.pdf_report import generate_pilling_report
import re

# Ensures that the specified directory exists; creates it if it doesn't.
def ensure_directory(target_folder: str | os.PathLike, recursive: bool = True) -> bool:
    try:
        target_path = pathlib.Path(target_folder)
        if (target_path.exists()):
            return True
        
        target_path.mkdir(parents=recursive, exist_ok=True)
        print(f"Created directory: {target_folder}")
        return True
    except OSError as e:
        print(f"Failed to create directory {target_folder}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error creating directory {target_folder}: {e}")
        return False

# Validates that the input sample and stage number is alphanumeric.
def validate_input_number(input_number: str) -> bool:
    return bool(input_number and input_number.strip().isalnum())

# Validates that the input grade is correct.
def validate_grade(input_number: str) -> bool:
    if not input_number:
        return False
    try:
        num = float(input_number)
        return num in {1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}
    except ValueError:
        return False

# Returns a sorted list of unique sample numbers found in the input directory.
def get_available_samples(suffix: str) -> List[str]:
    input_directory = os.path.join("data", "input_pictures", suffix)
    try:
        if not os.path.exists(input_directory):
            return []
        
        samples = set()
        for filename in os.listdir(input_directory):
            if filename.endswith('.png') and '-' in filename:
                sample = filename.split('-')[0]
                if sample:
                    samples.add(sample)
        return sorted(samples)
    except Exception as e:
        print(f"Error getting available samples: {e}")
        return []

# Helper function to check if image exists and add to status
def _check_image_exists(image_path: str, image_name: str, image_list: List[str]) -> bool:
    """Helper function to check if an image exists and add it to the appropriate list."""
    if os.path.exists(image_path):
        image_list.append(image_name)
        return True
    return False

@st.cache_data(show_spinner=False)
def cached_check_required_images(sample_number: str, stage_number: str, trial_number: str, suffix: str, epoch: int):
    """
    Cache presence/absence analysis of required images.
    """
    return check_required_images(sample_number, stage_number, trial_number, suffix)

# Checks for the presence of all required image files (8 rotational + 1 reference) for a given sample-stage pair.
def check_required_images(sample_number: str, stage_number: str, trial_number: str, suffix: str) -> dict:
    status = {
        'all_input_present': True,
        'all_difference_present': True,
        'present_input_images': [],
        'present_difference_images': [],
        'reference_present': True,
        'reference_image': []
    }
    
    try:
        # Check for 8 input (rotational) images with trial number
        for i in range(1, 9):
            image_name = f"{sample_number}-{stage_number}-{trial_number}-{i}.png"
            image_path = os.path.join("data", "input_pictures", suffix, image_name)
            if not _check_image_exists(image_path, image_name, status['present_input_images']):
                status['all_input_present'] = False

        # Check for 8 difference images with trial number
        for i in range(1, 9):
            image_name = f"{sample_number}-{stage_number}-{trial_number}-{i}-dif.png"
            image_path = os.path.join("data", "difference_pictures", suffix, image_name)
            if not _check_image_exists(image_path, image_name, status['present_difference_images']):
                status['all_difference_present'] = False
        
        # Reference image format: {sample}-{stage}-{trial}-ref.png (each trial has its own reference)
        ref_name = f"{sample_number}-{stage_number}-{trial_number}-ref.png"
        ref_path = os.path.join("data", "reference_pictures", suffix, ref_name)
        if not _check_image_exists(ref_path, ref_name, status['reference_image']):
            status['reference_present'] = False
        
        return status

    except Exception as e:
        print(f"Error checking required images: {e}")
        return {
            'all_input_present': False,
            'all_difference_present': False,
            'present_input_images': [],
            'present_difference_images': [],
            'reference_present': False,
            'reference_image': []
        }

class CustomPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        pass

# Helper function to process CSV data
def _process_csv_data(df_full: pd.DataFrame) -> tuple:
    """Helper function to process CSV data and extract grade information."""
    grade_row_index = None
    grade_number = None
    
    # Search for the "Grade" row
    for idx, row in df_full.iterrows():
        if str(row[0]).strip().lower() == "grade":
            grade_row_index = idx
            grade_number = row[1]  # Extract grade from the same row, column 1
            break
    
    # If no grade found, return failure
    if grade_row_index is None:
        return None, None, False
    
    # Extract data rows (everything before the grade row)
    df = df_full.iloc[:grade_row_index]
    
    # Set up column headers (assuming first row contains headers)
    if len(df) > 0:
        df.columns = df.iloc[0]
        df = df[1:]  # Remove header row from data
    
    return df, grade_number, True

# Helper function to calculate column widths
def _calculate_column_widths(col_names: List[str]) -> List[int]:
    """Helper function to calculate column widths based on column names."""
    return [60 if str(col).lower() == "image" else 30 for col in col_names]

# Helper function to add PDF header section
def _add_pdf_header(pdf: CustomPDF, sample_number: str, stage_number: str, load_weight: str, name: str):
    """Helper function to add header section to PDF."""
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Pilling Grade Analysis Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Operator Name: {name}", ln=True)
    pdf.cell(0, 10, f"Sample Number: {sample_number}", ln=True)
    pdf.cell(0, 10, f"Assessment Stage: {stage_number}", ln=True)
    pdf.cell(0, 10, f"Loading Weight (g): {load_weight}", ln=True)
    pdf.ln(5)

# Helper function to add PDF table
def _add_pdf_table(pdf: CustomPDF, df: pd.DataFrame, col_names: List[str], col_widths: List[int]):
    """Helper function to add table to PDF."""
    # Header row
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(200, 200, 200)
    pdf.set_text_color(0, 0, 0)
    for i, col in enumerate(col_names):
        pdf.cell(col_widths[i], 10, str(col), border=1, fill=True)
    pdf.ln()

    # Table rows
    pdf.set_font("Arial", size=10)
    for _, row in df.iterrows():
        for i, col in enumerate(col_names):
            cell_text = str(row[col])
            # Truncate long file names
            if col.lower() == "image" and len(cell_text) > 25:
                cell_text = cell_text[:22] + "..."
            pdf.cell(col_widths[i], 10, cell_text, border=1)
        pdf.ln()

def _scan_stage_results_for_sample(sample_number: str) -> dict[int, list[str]]:
    """
    Scan output/grading_results for this sample and return {stage_rubs:int -> [trial_grades]}.
    Pattern: {sample}-{stage}-{trial}-analysis.csv
    """
    results: dict[int, list[str]] = {}
    try:
        dirpath = os.path.join("output", "grading_results")
        if not os.path.isdir(dirpath):
            return results

        # Updated pattern to capture trial number
        pattern = re.compile(rf"^{re.escape(sample_number)}-(\d+)-(\d+)-analysis\.csv$")
        for fname in os.listdir(dirpath):
            m = pattern.match(fname)
            if not m:
                continue
            stage_rubs = int(m.group(1))
            trial_num = int(m.group(2))
            
            fpath = os.path.join(dirpath, fname)
            try:
                df_full = pd.read_csv(fpath, header=None)
                _, grade_number, success = _process_csv_data(df_full)
                if success and grade_number is not None and str(grade_number).strip():
                    grade_str = str(grade_number).strip()
                    if stage_rubs not in results:
                        results[stage_rubs] = []
                    results[stage_rubs].append(grade_str)
            except Exception:
                continue
        
        # Sort results for each stage to maintain consistent order
        for stage_rubs in results:
            results[stage_rubs].sort()
        
        return dict(sorted(results.items()))
    except Exception as e:
        print(f"Error scanning stage results for sample {sample_number}: {e}")
        return {}

def export_results(sample_number: str, stage_number: str, load_weight: str | float | int, operator_name: str, trial_number: str = "1") -> bool:
    """
    Export results to a single combined PDF per sample.
    - Dynamically builds table rows from all available stages for the sample.
    - Handles multiple trials per stage: shows individual results + average.
    """
    try:
        # Aggregate all available stage results for this sample
        stage_trials_map = _scan_stage_results_for_sample(sample_number)
        if not stage_trials_map:
            print(f"No analysis CSVs found for sample {sample_number} in output/grading_results/")
            return False

        # Build results table with dynamic trial handling
        results_by_rubs = {}
        for rub, grade_list in stage_trials_map.items():
            num_trials = len(grade_list)
            
            if num_trials == 1:
                # Single trial: Result 1 = grade, Result 2 & 3 = NA, Average = grade
                results_by_rubs[rub] = [grade_list[0], "NA", "NA", grade_list[0]]
            elif num_trials == 2:
                # Two trials: Result 1 & 2 = grades, Result 3 = NA, Average = mean
                avg = sum(float(g) for g in grade_list) / num_trials
                results_by_rubs[rub] = [grade_list[0], grade_list[1], "NA", f"{avg:.2f}"]
            elif num_trials >= 3:
                # Three or more trials: Result 1-3 = first 3 grades, Average = mean of all
                avg = sum(float(g) for g in grade_list) / num_trials
                results_by_rubs[rub] = [grade_list[0], grade_list[1], grade_list[2], f"{avg:.2f}"]
            else:
                # No trials (shouldn't happen, but handle gracefully)
                results_by_rubs[rub] = ["NA", "NA", "NA", "NA"]

        # Prepare output path
        reports_dir = "reports"
        ensure_directory(reports_dir)
        clean_operator_name = "".join(c for c in operator_name if c.isalnum() or c in ('-', '_')).strip() or "Unknown"
        pdf_filename = f"{sample_number}-{clean_operator_name}-report.pdf"
        pdf_filepath = os.path.join(reports_dir, pdf_filename)

        # Generate PDF (dynamic rub levels from scanned files)
        ok = generate_pilling_report(
            sample_number=sample_number,
            stage_number=stage_number,
            load_weight_g=load_weight,
            operator_name=operator_name,
            results_by_rubs=results_by_rubs,
            abradant="Similar Fabric",
            output_path=pdf_filepath,
            rub_levels=list(results_by_rubs.keys()),
            trial_number=trial_number
        )

        if ok:
            print(f"PDF report saved to: {pdf_filepath}")
            return True
        else:
            print("Error during PDF export: failed to generate PDF.")
            return False

    except Exception as e:
        print(f"Error during PDF export: {e}")
        return False
