# App Notes for Developers

How the Streamlit app is wired together. For the user workflow (setup, training, grading), see the [root README](../README.md). For the ITA-Net side (how Python talks to the DLL), see [INTEGRATION.md](INTEGRATION.md).

## File map

| Concern | File |
|---|---|
| UI, flow, login | [app.py](../app.py) |
| Config (paths, backend, mocks) | [app/settings/config.py](../app/settings/config.py) |
| Camera + motor | [app/capture/camera_control.py](../app/capture/camera_control.py) |
| Difference images + histogram | [app/capture/image_difference.py](../app/capture/image_difference.py), [app/analysis/histogram_analysis.py](../app/analysis/histogram_analysis.py) |
| PDF report | [app/reporting/pdf_report.py](../app/reporting/pdf_report.py) |
| Translations (EN/DE/FR/ES) | [app/reporting/translations.py](../app/reporting/translations.py) |
| ITA-Net glue | [itanet_recall.py](../itanet_recall.py), [itanet_training.py](../itanet_training.py) |
| Tests | [app/tests/](../app/tests/) |

## Two execution paths

- **Grading** — `app.py` → `analyze_difference_images_and_predict_output` → `predict_from_csv` (ITA-Net DLL) → analysis CSV in `output/grading_results/`. The sequence: capture 8 images → build 8 difference images against the stage-0 reference → extract Mean/Std/Max/Mode per image and average them → write a one-row temp CSV → call the DLL → clip and round-to-half the returned grade → append it (with the per-image features) to the analysis CSV.
- **Training** — the app writes labeled features to `data/training_features/*.csv`. The model owner then runs `scripts/train.py` (or `itanet_training.py` directly), which converts the CSV to the DLL's `.dat` byte format, archives the previous `.NET` + `.TRN`, and calls `run_training_session` to overwrite `.NET` with new weights.

Config lives in one place: [`app/settings/config.py`](../app/settings/config.py). `MOCK_HARDWARE` and `MOCK_PREDICTION` bypass the camera and the DLL respectively for local development.

## Hardware

- **Basler camera** — check the connection with Pylon Viewer before starting the app.
- **LEGO EV3 motor** — over USB or SSH.

If either is absent, set `MOCK_HARDWARE = True` and drop images into `data/input_pictures/<suffix>/` by hand.

## Tests

- `pytest app/tests/` — pure-Python unit and PDF suites. No DLL required.
- `python app/tests/test_integration.py all --grade <grade>` — end-to-end recall against the real DLL and a trained `.NET`. Requires both.

The unit suite pins the ITA-Net enum values, the `.dat` byte format, and the `.NET`-shape guard, so drift there fails a fast test before it becomes a mystery crash inside the DLL.
