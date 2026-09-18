# App Development Notes

This document is for developers extending the Streamlit app — file map, configuration, hardware setup, testing, and troubleshooting.

For the **end-user workflow** (install → collect data → train → grade), start with the [root README](../../README.md). For **ITA-Net integration internals** (DLL contract, recall data flow), see [docs/itanet/WORKFLOW.md](../itanet/WORKFLOW.md).

---

## Architecture

| Concern | File |
|---|---|
| UI / flow / login | [app.py](../../app.py) |
| Training CLI | [itanet_training.py](../../itanet_training.py) |
| ITA-Net DLL glue (recall) | [itanet_recall.py](../../itanet_recall.py) |
| Configuration | [app/settings/config.py](../../app/settings/config.py) |
| Image capture & motor control | [app/capture/camera_control.py](../../app/capture/camera_control.py) |
| Difference & histogram analysis | [app/capture/image_difference.py](../../app/capture/image_difference.py), [app/analysis/histogram_analysis.py](../../app/analysis/histogram_analysis.py) |
| PDF reporting | [app/reporting/pdf_report.py](../../app/reporting/pdf_report.py) |
| Translations (EN/DE/FR/ES) | [app/reporting/translations.py](../../app/reporting/translations.py) |
| Helpers (dir checks, validators) | [app/helpers/utils.py](../../app/helpers/utils.py) |
| Tests | [app/tests/](../../app/tests/) |

Two execution paths cross this code:

- **Grading** — `app.py` → `analyze_difference_images_and_predict_output` → `predict_from_csv` (ITA-Net DLL) → write `output/grading_results/<sample>-<stage>-<trial>-analysis.csv`.
- **Training** — `app.py` Training Mode appends to `data/training_features/*.csv`; the user separately runs `python itanet_training.py <csv>` to write new weights.

The recall pipeline is documented in detail in [docs/itanet/WORKFLOW.md](../itanet/WORKFLOW.md).

## Configuration

All paths and runtime settings live in [app/settings/config.py](../../app/settings/config.py). Defaults:

```python
BACKEND            = "itanet_dll"                    # or "tf" for the TensorFlow path
ITANET_ROOT        = "models/itanet"
GRADES             = ("pilling", "matting", "fuzzing")
DLL_PATH           = "models/itanet/itanet.dll"      # shared DLL, loaded by absolute path
TF_MODEL_PATH      = "models/tf/my_model.h5"
FEATURE_COLUMNS    = ("Mean", "Std", "Max", "Mode")
DECIMAL            = "Point"
ROUND_TO_HALF      = True
CLIP_RANGE         = (1.0, 5.0)
```

Per-grade paths are resolved via builder functions:
- `get_itanet_run_dir(grade)` → `models/itanet/{grade}/run/`  (CWD when the DLL is called)
- `get_itanet_fls_dir(grade)` → `models/itanet/{grade}/data/`  (recall.fls / training.fls)
- `get_itanet_archive_dir(grade)` → `models/itanet/archive/{grade}/`

UI theme (optional): `.streamlit/config.toml`.

## Folder layout

Auto-created at runtime, all `.gitignored` (see the [root README](../../README.md#folder-layout) for the same table aimed at users):

- `data/input_pictures/`, `data/reference_pictures/`, `data/difference_pictures/` — raw, reference, and computed images
- `data/training_features/` — feature CSVs the training CLI consumes
- `output/grading_results/` — per-trial prediction CSVs
- `reports/` — exported PDFs
- `models/itanet/run/` — DLL, trained `Neuronalesnetz.NET`, `.TRN`, and training/recall `.dat` files (DLL's cwd)
- `models/itanet/data/` — `recall.fls` and `training.fls` (opened by the DLL as `..\data\*.fls`)
- `models/itanet/archive/` — timestamped `.NET` + `.TRN` snapshots taken before each training run
- `models/tf/` — TensorFlow backend model

Data directories are safe to delete and recreate — the app repopulates them.

## Hardware Setup

- **Basler camera** — verify connection and focus with Pylon Viewer before running the app.
- **LEGO EV3 motor** — connect over USB or SSH (PuTTY).

The app falls back gracefully if hardware is absent (just drop images into `data/input_pictures/<suffix>/` and the difference/analysis path still runs).

## Admin Access

- Demo password: `1234`. To change it, edit the login logic in [app.py](../../app.py) (search for `ADMIN_PASSWORD`).

## Testing

### Integration tests — `app/tests/test_integration.py`

Run without launching Streamlit:

```bash
python app/tests/test_integration.py all                       # prereqs + recall + regression
python app/tests/test_integration.py recall                    # recall on the default fixture CSV
python app/tests/test_integration.py recall --csv <path>       # recall on a CSV of your choice
python app/tests/test_integration.py recall --csv <path> --output <out.csv>
python app/tests/test_integration.py regression                # CSV format / formatter contracts
```

The default fixture is `output/grading_results/00000-100-1-analysis.csv`. Pass `--csv <path>` to run recall against any CSV with `Image,Mean,Std,Max,Mode` columns — this is how you sanity-check freshly-trained weights against a held-out labeled test set (see [root README §4](../../README.md#4-check-the-trained-model-against-known-samples-recommended) for that workflow). Predictions go to `_test_predictions.csv` at the repo root unless `--output` redirects them.

The runner first prereq-checks `itanet.dll` and `Neuronalesnetz.NET` under `models/itanet/run/` and bails fast if either is missing. Each check prints `PASS` / `FAIL`; non-zero exit on any failure.

There's also a pure-Python unit suite that doesn't require the DLL:

```bash
python -m pytest app/tests/test_itanet_unit.py -v
```

It pins the HAUPTDEF.H enum values, the on-disk `.dat` byte contract, the `.NET`-shape guard, the NaN detection in recall output, and the run/data folder-split invariant — so drift on any of these fails the test suite before it becomes a mystery crash inside the DLL.

Use these runners after any change to [`itanet_recall.py`](../../itanet_recall.py) or [`itanet_training.py`](../../itanet_training.py). The regression subcommand of `test_integration.py` exercises three byte-format edge cases that have bitten the integration in the past — see [docs/itanet/WORKFLOW.md §Integration tests](../itanet/WORKFLOW.md#integration-tests--apptests-test_integrationpy) for the rationale.

### PDF tests — `app/tests/test_pdf_report.py`

Run with pytest:

```bash
pytest app/tests/test_pdf_report.py -m pdf
```

The same file is also a CLI: `python app/tests/test_pdf_report.py` writes sample PDFs to `reports/visual_tests/` for eyeball verification.

## Common Issues

| Symptom | Fix |
|---|---|
| Camera not detected | Check USB, drivers, and Pylon Viewer. |
| EV3 motor won't connect | Verify SSH credentials and IP in PuTTY. |
| DLL not found | Confirm `models/itanet/run/itanet.dll` exists; or override `DLL_PATH` in [config.py](../../app/settings/config.py). |
| Library install errors | `pip install <package> --upgrade` for the failing package. |
| Recall predicts the same grade every time | The trained weights may be stale — re-run `itanet_training.py` (see [root README §3](../../README.md#3-train-the-ita-net-model)). |

## Change Guide

Where to make changes:

- **Settings (paths, backend, clipping)** → [app/settings/config.py](../../app/settings/config.py)
- **UI / flow / login** → [app.py](../../app.py)
- **Translations** → [app/reporting/translations.py](../../app/reporting/translations.py)
- **ITA-Net file names** → `ITANetFiles` class in [itanet_recall.py](../../itanet_recall.py)
- **TF model path** → `TF_MODEL_PATH` in [config.py](../../app/settings/config.py); file at [models/tf/my_model.h5](../../models/tf/my_model.h5)

Keep DLLs and proprietary datasets out of source control (already `.gitignored`).

## Contact

Issues and feedback: <https://github.com/Akshat-Shandilya/Automated_Pilling_Grade_Classifier>.
