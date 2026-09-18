# Automated Textile Grade Classifier

A prototype for automating textile quality assessment. It captures images on a Basler camera, compares them against a stage-0 reference shot, and predicts one or more grades — **pilling**, **matting**, and **fuzzing** (each 1.0 – 5.0 in 0.5 steps) — using either a trained neural network per grade (ITA-Net) or a TensorFlow-based deep learning model.

This document walks through the full workflow end to end: 
**install → collect labeled samples → train the model → check the trained model → grade new samples**. 
Follow the sections in order the first time around.

> Need more detail later? **[docs/app/APP.md](docs/app/APP.md)** covers app internals for developers. **[docs/itanet/WORKFLOW.md](docs/itanet/WORKFLOW.md)** covers ITA-Net integration internals (DLL contract, file formats, recall data flow).

## 1. Install

You need **Python 3.11** on Windows. Install it from the [official Python site](https://www.python.org/downloads/release/python-3100/) — make sure to tick **"Add Python to PATH"** during setup — or use winget:

```bash
winget install --id Python.Python.3.11 -e
python --version    # should print Python 3.11.x
```

Then, from this project folder:

```bash
py -3.11 -m venv env
env\Scripts\activate                  # Command Prompt
# or:  source env/Scripts/activate     (Git Bash / MINGW)

pip install -r requirements.txt
```

That's it. Verify with `python --version` — you should still see Python 3.11.x while the environment is active.

> If you don't have the Basler camera or the LEGO EV3 motor handy, the app still runs in software-only mode — see [docs/app/APP.md](docs/app/APP.md#hardware-setup) for the hardware details when you're ready to wire them up.

## 2. Collect labeled training samples

Launch the app:

```bash
python -m streamlit run app.py
```

It opens at <http://localhost:8501>.

1. Pick **Training Mode** and log in (default password: `1234`).
2. Enter operator name, sample number, stage (rubs), trial number, and the **known grades** (1.0 – 5.0 in 0.5 steps) for each active grade type (pilling / matting / fuzzing).
3. Click **Capture**. The app captures 8 rotational images, builds difference images against the stage-0 reference, extracts a feature vector (`Mean`, `Std`, `Max`, `Mode`), and appends a row to a pair of CSVs *per active grade* under `data/training_features/`:
   - `{grade}_per_image_features.csv` — one row per difference image (8 rows per sample)
   - `{grade}_averaged_features.csv` — one averaged row per sample
4. Repeat for as many labeled samples as you can. More variety across grades and rub counts means a better model.

You can quit and resume later — the CSVs are appended on every capture.

## 3. Train the ITA-Net model

Once you've collected enough training samples, run training from the command line (with the environment activated). Each grade has its own network — train them one at a time with `--grade`:

```bash
python itanet_training.py data\training_features\pilling_averaged_features.csv --grade pilling
python itanet_training.py data\training_features\matting_averaged_features.csv --grade matting
python itanet_training.py data\training_features\fuzzing_averaged_features.csv --grade fuzzing
```

What happens:

1. The CSV is converted into the binary `.dat` files the DLL expects (`TrainingData.dat`, `TrainingGrades.dat` in `models/itanet/run/`).
2. The existing `Neuronalesnetz.NET` (and its `.TRN`) are archived to `models/itanet/archive/<UTC timestamp>/` (safety backup — copy them back if a run goes bad).
3. The DLL runs training and rewrites `models/itanet/run/Neuronalesnetz.NET` in place with the trained weights (the file was topology-only going in; it's weights-bearing coming out).
4. During grading, the new `.NET` is picked up automatically.

> **Re-training from scratch.** `--net-type 1` (the default) requires a pristine, topology-only `.NET`. If your current `.NET` has already been trained (>500 bytes), the script will stop with an error — restore a pristine copy from `models/itanet/archive/` first, or pass `--net-type 2` to continue training from the current weights.

Options:

| Flag | Default | Effect |
|---|---|---|
| `--net-type N` | `1` | `1` = fresh weights (needs pristine `.NET`), `2` = continue training from current weights. |
| `--decimal MODE` | `comma` | `comma` or `point` — decimal separator written into the `.dat` files; the DLL's `komma_punkt` flag is derived from this. |
| `--shuffle N` | `1` | `1` = shuffle training patterns, `0` = preserve original order. |
| `--no-backup` | off | Skip archiving the existing `.NET` + `.TRN`. |

Paths are per-grade: the DLL, `.NET`, `.TRN`, and `.dat` files live in `models/itanet/{grade}/run/`; the `.fls` filelists in `models/itanet/{grade}/data/`; archives in `models/itanet/archive/{grade}/`. The shared DLL sits at `models/itanet/itanet.dll`. Override these in [`app/settings/config.py`](app/settings/config.py) if you need a non-standard layout.

If you want to check the trained model against known samples (recall)

### Recall without Streamlit

Run recall on any analysis CSV directly

```bash
python itanet_recall.py output\grading_results\<sample>-<stage>-<trial>-<grade>-analysis.csv
```

Summary rows (`Average`, `Grade`, `Backend`) at the bottom are filtered out automatically.

## 4. Grade new samples interactively using Streamlit UI 

In the app, pick **Grading Mode**:

1. Enter operator name, sample number, stage, trial, and load weight.
2. Pick the active grades in the sidebar (any subset of pilling / matting / fuzzing).
3. Click **Capture** — the app takes 8 images, generates the features (Mean/Std/Max/Mode), calls each selected grade's ITA-Net for prediction, and shows the grades on screen.
4. Click **Export Results** to save a combined PDF report under `reports/` — one column-group per grade that has data.

Per-trial analysis statistics are saved in `output/grading_results/` (one CSV per `<sample>-<stage>-<trial>-<grade>`), so you can always re-inspect the raw features each ITA-Net model used.

## Folder layout

All folders below are auto-created at runtime and `.gitignored` — they hold local artifacts only.

| Folder | What's in it |
|---|---|
| `data/input_pictures/{for_grading,for_training}/` | Raw 8-image capture sets from the camera (stage-0 is the reference; higher stages are the pilled samples) |
| `data/difference_pictures/{for_grading,for_training}/` | Generated difference images (current vs. stage-0 reference) |
| `data/training_features/` | Per-grade feature CSVs (Mean/Std/Max/Mode + Grade) — inputs to `itanet_training.py` |
| `output/grading_results/` | Per-trial, per-grade prediction CSVs (`{sample}-{stage}-{trial}-{grade}-analysis.csv`) |
| `reports/` | Exported PDF reports |
| `models/itanet/itanet.dll` | Shared DLL (loaded once, used by all three grade networks) |
| `models/itanet/{grade}/run/` | Trained `Neuronalesnetz.NET`, `.TRN`, and training/recall `.dat` files for that grade (the DLL's cwd at call time) |
| `models/itanet/{grade}/data/` | `recall.fls` and `training.fls` — filelists the DLL opens as `..\data\*.fls` from the run dir |
| `models/itanet/archive/{grade}/` | Timestamped backups of `Neuronalesnetz.NET` + `.TRN` from before each training run |
| `models/tf/` | TensorFlow backend model (alternate prediction path) |

The only scripts at the root you would run:

- **`app.py`** — `python -m streamlit run app.py` to launch the app.
- **`itanet_training.py`** — train the model from a labeled feature CSV.
---

## Need more detail?

- **Extending the app's UI or features** → [docs/app/APP.md](docs/app/APP.md)
- **ITA-Net integration internals (DLL contract, file formats, recall data flow)** → [docs/itanet/WORKFLOW.md](docs/itanet/WORKFLOW.md)
- **Rebuilding the model bundle from C source** → [docs/itanet/BACKEND_SETUP.md](docs/itanet/BACKEND_SETUP.md)

Issues and feedback: open a ticket at <https://github.com/Ambuj-Choudha/automated-textile-grade-classifier>.
