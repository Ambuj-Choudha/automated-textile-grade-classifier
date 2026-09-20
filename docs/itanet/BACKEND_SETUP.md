# Rebuilding the model bundle from source

> **You usually don't need this.** The prototype ships with a working bundle under [`models/itanet/`](../../models/itanet/). For the end-user training/grading workflow see the [root README](../../README.md); for runtime integration internals see [WORKFLOW.md](WORKFLOW.md). This document is only for the case where you rebuild the DLL from C source or set the bundle up from scratch on a new machine.

To run ITA-Net recall from the prototype ([itanet_recall.py](../../itanet_recall.py)), the following files must exist under [models/itanet/](../../models/itanet/). They originate from the ITA-Net source tree; this doc describes which file comes from where.

## Required layout

```
models/itanet/
├── common/                     ← shared runtime, one binary for all grades
│   └── itanet.dll              ← compiled library
├── {pilling,matting,fuzzing}/  ← one full network bundle per grade
│   ├── run/                    ← DLL cwd at call time; holds the grade's own files
│   │   ├── Neuronalesnetz.NET  ← starts pristine (topology only, ~26 bytes);
│   │   │                         becomes weights-bearing after training runs
│   │   ├── Neuronalesnetz.TRN  ← training schedule (epochs, learning rate, momentum)
│   │   ├── TrainingData.dat    ← original training features (used at recall
│   │   │                         time to normalize inputs)
│   │   └── TrainingGrades.dat  ← original training labels (same reason)
│   └── data/                   ← sibling of run/ — DLL hardcodes ..\data\*.fls
│       ├── recall.fls          ← 5 filenames the recall path opens
│       └── training.fls        ← 4 filenames the training path opens
└── archive/                    ← auto-created; timestamped .NET + .TRN
    └── {grade}/                  snapshots taken before each training run
        └── <UTC>/
            ├── Neuronalesnetz.NET
            └── Neuronalesnetz.TRN
```

`RecallData.dat` and `Recall_output.dat` are written/read at runtime by the integration — do **not** ship them.

## Where each file comes from

| Destination in prototype | Source in ITA-Net repo | Notes |
|---|---|---|
| `models/itanet/common/itanet.dll` | `build/itanet.dll` (after `python build.py`) | Shared runtime, one binary for all grades. Rebuild whenever the C source changes. See [DLL rebuild](#dll-rebuild) below. |
| `models/itanet/{grade}/run/Neuronalesnetz.NET` | `experiment/train/run/Neuronalesnetz.NET` (pristine, before running training) | Topology-only, ~26 bytes. Ship one pristine copy per grade — training rewrites it in place. |
| `models/itanet/{grade}/run/Neuronalesnetz.TRN` | `experiment/train/run/Neuronalesnetz.TRN` | Training schedule. Same for train and recall. |
| `models/itanet/{grade}/run/TrainingData.dat` | `experiment/train/run/TrainingData.dat` | Training feature table the current weights were fit on. Used for normalization at recall time — see [why the training files travel](#why-the-training-files-travel-with-the-weights) below. |
| `models/itanet/{grade}/run/TrainingGrades.dat` | `experiment/train/run/TrainingGrades.dat` | Same — paired with `TrainingData.dat`. |
| `models/itanet/{grade}/data/recall.fls` | (this repo) | Already committed. Lists the files the DLL opens at recall time. |
| `models/itanet/{grade}/data/training.fls` | (this repo) | Already committed. Lists the files the DLL opens at training time. |

## `recall.fls` contents

Fixed and already present in this repo:

```
RecallData.dat
Recall_output.dat
TrainingData.dat
TrainingGrades.dat
Neuronalesnetz.NET
```

Order matters — do not edit. Same for `training.fls`:

```
TrainingData.dat
TrainingGrades.dat
Neuronalesnetz.NET
Neuronalesnetz.TRN
```

The integration re-writes both filelists from `itanet_recall.py` / `itanet_training.py` every call, so any hand-edits are silently overwritten.

## DLL rebuild

If you're changing the C source, rebuild from the ITA-Net repo checkout (see [`docs/itanet_latest/`](../itanet_latest/) for a snapshot of the upstream docs):

```bash
cd path/to/ITA-net-repo
python build.py                        # produces build/itanet.dll (+ .exe)
cp build/itanet.dll <prototype>/models/itanet/common/itanet.dll
```

The DLL must export both `run_training_session` and `run_recall_session`, each with a `(int net_type, int komma_punkt, int shuffle) -> int` signature. The prototype's [`load_itanet_dll()`](../../itanet_recall.py) binds argtypes at load time and caches the handle.

## Verifying the setup

After copying, from the repo root:

```bash
python itanet_recall.py output/grading_results/00000-100-1-analysis.csv
```

A successful run writes `output/grading_results/00000-100-1-analysis_predictions.csv` and prints `[ITANet] Recall completed: N predictions`.

Failure modes and their fixes:

- **`ITANet DLL not found at …`** — `itanet.dll` is missing from `models/itanet/common/`. Copy or rebuild it.
- **`… is only 26 bytes — looks pristine (topology only), not weights-bearing`** — the `.NET` under `run/` hasn't been trained yet, so `LOAD_FROM_FILE` can't read weights. Run `python itanet_training.py <csv>` once.
- **`… contains NaN on N of M lines — trained .NET has diverged weights`** — the last training run diverged; the on-disk `.NET` has NaN weights. Restore a good `.NET` + `.TRN` from `models/itanet/archive/<UTC>/`, or re-train with saner `.TRN` hyperparameters.
- **`ITANet recall failed with code: -1`** — an invalid flag value was passed to the DLL. Check that `--net-type`, `--decimal`, and `--shuffle` are in-range. The DLL prints the offending flag to stdout before returning.
- **Silent wrong predictions** — the `TrainingData.dat` / `TrainingGrades.dat` in `run/` don't match the training run that produced the current `.NET`. Restore the matching set (or re-train from the CSV that produced this `.NET`).

## Why the training files travel with the weights

ITA-Net normalizes recall inputs against the min/max of the original training features (handled inside `WANDEL.C` in the C source). Shipping the trained `.NET` without its paired `TrainingData.dat` / `TrainingGrades.dat` produces silently wrong predictions. Always keep these three files as a set — and if you archive an old `.NET` for reproducibility, snapshot the paired training `.dat` files alongside it (the archive step currently only saves `.NET` + `.TRN`; the `.dat` files live in `run/` and get overwritten on the next training run).
