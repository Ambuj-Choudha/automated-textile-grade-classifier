import os
from datetime import datetime
from typing import Dict, List, Sequence, Optional
from fpdf import FPDF
import traceback

# Public default rub levels used in the table
DEFAULT_RUB_LEVELS: Sequence[int] = (125, 500, 1000, 2000, 5000, 7000)


class PillingReportPDF(FPDF):
    def header(self):
        # Logo at the top, centered
        logo_path = os.path.join("logos", "Logo_TexIQ_v1.0.jpg")
        if os.path.exists(logo_path):
            # Calculate center position for logo
            logo_width = 40  # Width in mm
            x_center = (self.w - logo_width) / 2
            self.image(logo_path, x=x_center, y=10, w=logo_width)
            self.ln(25)  # Space after logo
            
        # Title: centered, bold
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Pilling Test Report", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

def _add_statement(pdf: PillingReportPDF):
    pdf.ln(2)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, "Pilling test has been conducted according to ISO-12945-2 standards")
    pdf.ln(2)

def _add_details(
    pdf: PillingReportPDF,
    *,
    date_str: str,
    sample_number: str,
    operator_name: str,
    load_weight_g: str | float | int,
    abradant: str = "Similar Fabric",
):
    pdf.set_font("Arial", "", 11)

    # Prepare left/right content
    left_lines = [
        f"Sample Number: {sample_number}",
        f"Abradant used: {abradant}",
        f"Operator Name: {operator_name}",
        f"Loading Weight (gms): {load_weight_g}",
    ]
    right_text = f"Date: {date_str}"

    # Layout metrics
    eff_w = pdf.w - pdf.l_margin - pdf.r_margin
    gutter = 6
    row_h = 8
    date_w = max(50, pdf.get_string_width(right_text) + 6)
    left_w = max(60, eff_w - date_w - gutter)

    # Anchor top-left Y for the block
    x_left = pdf.l_margin
    y_top = pdf.get_y()

    # Draw right column (date) aligned to the right area
    pdf.set_xy(x_left + left_w + gutter, y_top)
    pdf.cell(date_w, row_h, right_text, border=0, ln=0, align="R")

    # Draw left column lines
    for i, line in enumerate(left_lines):
        pdf.set_xy(x_left, y_top + i * row_h)
        pdf.cell(left_w, row_h, line, border=0, ln=0, align="L")

    # Move cursor below the block
    pdf.set_y(y_top + len(left_lines) * row_h + 2)

def _add_results_table(
    pdf: PillingReportPDF,
    results_by_rubs: Dict[int, List[str | float | int]],
    rub_levels: Sequence[int] = DEFAULT_RUB_LEVELS,
):
    # Layout
    pdf.set_font("Arial", "", 11)
    left_w = 70
    res_w = 30
    row_h = 8
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(left_w, row_h * 2, "Number of pilling rubs", border=1, align="C", fill=True)
    pdf.cell(res_w * 4, row_h, "Pilling", border=1, align="C", fill=True)
    pdf.ln(row_h)
    pdf.set_x(pdf.l_margin + left_w)
    for sub in ("Result 1", "Result 2", "Result 3", "Average"):
        pdf.cell(res_w, row_h, sub, border=1, align="C", fill=True)
    pdf.ln(row_h)
    pdf.set_font("Arial", "", 10)

    for rub in rub_levels:
        pdf.cell(left_w, row_h, f"{rub} rev.", border=1, align="L")
        values = list(results_by_rubs.get(rub, []))
        while len(values) < 4:
            values.append("")
        for val in values[:4]:
            pdf.cell(res_w, row_h, f"{val}", border=1, align="C")
        pdf.ln(row_h)

def _add_reference_image(
    pdf: PillingReportPDF,
    sample_number: str,
    trial_number: str = "1"):
    
    ref_dir = os.path.join("data", "reference_pictures", "for_grading")
    if not os.path.exists(ref_dir):
        return False
    
    # First, try the specified trial number
    ref_filename = f"{sample_number}-0-{trial_number}-ref.png"
    ref_path = os.path.join(ref_dir, ref_filename)
    
    if os.path.exists(ref_path):
        used_trial = trial_number
    else:
        # Fall back: find any reference image for this sample (stage 0)
        import glob
        pattern = os.path.join(ref_dir, f"{sample_number}-0-*-ref.png")
        matches = glob.glob(pattern)
        if not matches:
            return False
        ref_path = matches[0]  # Use first available
        # Extract trial number from filename for display
        basename = os.path.basename(ref_path)
        parts = basename.replace(".png", "").split("-")
        used_trial = parts[2] if len(parts) >= 4 else "unknown"
    
    pdf.ln(6)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Reference Image (Unpilled sample):", ln=True)
    pdf.ln(2)
    
    # Calculate center position for image
    img_width = 40  # Width in mm
    x_center = (pdf.w - img_width) / 2
    
    # Add image centered
    pdf.image(ref_path, x=x_center, w=img_width)
    pdf.ln(4)
    
    # Add caption
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 6, f"Reference Image for Sample: {sample_number}, Trial: {used_trial}", ln=True, align="C")
    return True

def generate_pilling_report(
    *,
    sample_number: str,
    stage_number: str | None = None,  # kept for future use if needed
    load_weight_g: str | float | int,
    operator_name: str,
    results_by_rubs: Dict[int, List[str | float | int]] | None = None,
    abradant: str = "Similar Fabric",
    output_path: str,
    report_date: Optional[datetime] = None,
    rub_levels: Optional[Sequence[int]] = None,
    trial_number: str = "1",  # Add trial number parameter
) -> bool:
    """
    Create a PDF report with the required structure and save it to output_path.
    Returns True on success.
    """
    try:
        pdf = PillingReportPDF()
        pdf.add_page()

        _add_details(
            pdf,
            date_str=(report_date or datetime.now()).strftime("%d-%m-%Y"),
            sample_number=sample_number,
            operator_name=operator_name,
            load_weight_g=load_weight_g,
            abradant=abradant,
        )

        data = results_by_rubs or {}
        levels = list(rub_levels) if rub_levels else (sorted(data.keys()) if data else list(DEFAULT_RUB_LEVELS))
        _add_results_table(pdf, data, rub_levels=levels)

        # Append ISO statement
        pdf.ln(6)
        _add_statement(pdf)

        # Add reference image at the bottom
        _add_reference_image(pdf, sample_number, trial_number)

        # Ensure directory exists and write file
        _ensure_parent_dir(output_path)
        pdf.output(output_path)
        return True
    except Exception as e:
        print(f"PDF generation error: {e}")
        traceback.print_exc()
        return False

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)