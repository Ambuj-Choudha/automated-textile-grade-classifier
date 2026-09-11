# ITA-Net Integration

`itanet.dll` is a small C library compiled from the ITA-Net source tree. The prototype calls it via Python `ctypes` for two operations:

- **Training** — read a feature CSV, produce a weights-bearing `Neuronalesnetz.NET`.
- **Recall**   — read a feature CSV, return predicted grades.

This document is the integration reference for developers. For the end-user training/grading workflow start with the [root README](../../README.md). For app architecture see [docs/app/APP.md](../app/APP.md). For rebuilding the bundle from C source see [BACKEND_SETUP.md](BACKEND_SETUP.md).

## How it fits together

```
   feature CSV (Mean,Std,Max,Mode + Grade)
                       │
                       ▼
   ┌────────────────────────────────────────┐
   │  itanet_training.py  (training path)   │  ──►  updated Neuronalesnetz.NET
   │  itanet_recall.py    (recall path)     │  ──►  predicted grade(s)
   └────────────────────┬───────────────────┘
                        │   chdir into models/itanet/run/
                        ▼
                    itanet.dll
                        │   reads ..\data\*.fls, then reads & writes
                        │   the files listed there (in cwd)
                        ▼
          .NET, .TRN, .dat, .fls files
```

**Constraint:** the DLL hardcodes `..\data\training.fls` / `..\data\recall.fls` as its entry-point filelist. The caller must `chdir` into a folder that has `..\data\` as a sibling before invoking. The prototype's layout puts the DLL and data files in `models/itanet/run/`, and the filelists in `models/itanet/data/`, so `..\data\` resolves correctly.

## What lives under `models/itanet/`

```
models/itanet/
├── run/                        ← DLL cwd for both training and recall
│   ├── itanet.dll              ← compiled library (2 exported symbols)
│   ├── Neuronalesnetz.NET      ← starts as pristine topology; becomes
│   │                             weights-bearing after training
│   ├── Neuronalesnetz.TRN      ← training schedule (epochs, learning rate)
│   ├── TrainingData.dat        ← training features (also used at recall
│   │                             time to normalize inputs — see below)
│   ├── TrainingGrades.dat      ← training labels (same reason)
│   ├── RecallData.dat          ← written each recall
│   ├── Recall_output.dat       ← written by the DLL each recall
│   └── (transient .nrm/.nri/.nro/.err/.xcl/.neu files the DLL creates)
├── data/                       ← sibling of run/ so DLL sees ..\data\*.fls
│   ├── recall.fls              ← 5 filenames the recall path opens
│   └── training.fls            ← 4 filenames the training path opens
└── archive/                    ← timestamped .NET + .TRN snapshots
    └── <UTC>/
        ├── Neuronalesnetz.NET
        └── Neuronalesnetz.TRN
```

### The two `.fls` filelists

`recall.fls` — order matters, five filenames on separate lines:

```
RecallData.dat
Recall_output.dat
TrainingData.dat
TrainingGrades.dat
Neuronalesnetz.NET
```

`training.fls` — four filenames:

```
TrainingData.dat
TrainingGrades.dat
Neuronalesnetz.NET
Neuronalesnetz.TRN
```

Both files are (re)written by the Python integration on every call to keep them in sync with whatever the CLI flags asked for, so hand-edits get overwritten.

## DLL contract

Two exports, same `(int, int, int) -> int` signature:

```c
int run_training_session(int net_type, int komma_punkt, int shuffle);
int run_recall_session  (int net_type, int komma_punkt, int shuffle);
```

Return codes: `0` = ran to completion, `-1` = an invalid flag value was passed. Individual failures inside the C core (bad file read, bad data format) are **not** propagated — same fire-and-forget contract as the CLI exe.

Enum values (mirrored from `INCLUDE/HAUPTDEF.H` in the ITA-Net source; also pinned in [`app/tests/test_itanet_unit.py`](../../app/tests/test_itanet_unit.py) so drift breaks tests before it breaks the DLL):

| Constant | Value | Meaning |
|---|---|---|
| `CREATE_NEW_NET`         | `1` | Read topology from `.NET`, randomize weights. Requires a pristine `.NET`. |
| `LOAD_FROM_FILE`         | `2` | Read topology **and** trained weights from `.NET`. |
| `KOMMA_ZU_PUNKT`         | `0` | Data uses `,` decimals; DLL converts to `.` on read, back to `,` on write. |
| `PUNKT_ZU_KOMMA`         | `1` | Data uses `.` — skip conversion. |
| `SHUFFLE_OFF` / `SHUFFLE_ON` | `0` / `1` | Preserve / shuffle training patterns. Only meaningful for training. |

**Seeding.** The current DLL has no seed hook — it uses `srand(time(NULL))` inside `new_net` and `shuffle_data`, so consecutive runs are not reproducible. Any `seed=` kwarg on `predict_from_csv` was removed; passing one now raises `TypeError` (pinned by `test_seed_kwarg_no_longer_accepted`).

**Threading.** Not safe for concurrent calls. The DLL relies on process-wide state (cwd, `srand`, output files). Two calls in parallel will race, so the integration never runs them concurrently.

**Pristine vs. weights-bearing `.NET`.** `new_net` (topology-only reader) and `load_net` (topology + weights reader) parse different on-disk formats. Passing a weights-bearing `.NET` to `CREATE_NEW_NET`, or a pristine `.NET` to `LOAD_FROM_FILE`, silently produces garbage. `itanet_training.py` uses a `< 500 bytes = pristine` size heuristic (`PRISTINE_NET_MAX_BYTES`) to fail fast in either mistake.

## Training flow

Triggered by `python itanet_training.py <csv>` (see [README §3](../../README.md#3-train-the-ita-net-model) for user-facing flags).

1. `_guard_net_matches_net_type()` verifies the on-disk `.NET` matches the requested `--net-type` (pristine for `1`, weights-bearing for `2`).
2. `csv_to_training_dats()` rewrites `TrainingData.dat` + `TrainingGrades.dat` in the run dir from the input CSV. Whole-number features get a trailing `.0`/`,0` (the DLL's pattern loader access-violates on bare ints).
3. `training.fls` is rewritten in the data dir.
4. If `backup=True` (default), the existing `.NET` + `.TRN` are copied to `models/itanet/archive/<UTC>/`. The `.TRN` goes with the `.NET` because the hyperparameters (learning rate, epoch counts) are needed to reproduce the run — a `.NET` alone is not enough context.
5. `chdir` into `models/itanet/run/`, then `run_training_session(net_type, komma_punkt, shuffle)`.
6. On success, `.NET` has been rewritten in place with the trained weights. The next recall picks it up.

### `.dat` byte contract

The DLL is byte-picky about the training/recall `.dat` files it reads. `csv_to_training_dats()` and `ITANetFileWriter.write_data_file()` produce:

- Tab-separated columns, CRLF (`\r\n`) line endings.
- Header line `<N>\t<ncols>\r\n` where `N` is the row count.
- Whole-number features: written with a trailing `.0`/`,0` (`104` → `104.0` or `104,0`). Bare ints in feature data cause an access violation inside `run_recall_session`.
- Whole-number **grades**: written as bare ints (`4`, not `4.0`). The grade-file pattern loader accepts them.
- Decimal separator: controlled by `decimal="comma"|"point"`. `komma_punkt` (the DLL flag) is derived from the same value so the `.dat` format and the DLL's parser stay in sync — passing `decimal="point"` while leaving `komma_punkt=0` would silently produce mis-scaled inputs.

The unit tests in [`test_itanet_unit.py`](../../app/tests/test_itanet_unit.py) pin all four of these on both `comma` and `point` output.

## Recall flow

Triggered automatically by Grading Mode in the Streamlit app. The flow:

```
app.py
  └─ analyze_difference_images_and_predict_output()  ── app/analysis/histogram_analysis.py
       └─ predict_from_csv()                         ── itanet_recall.py
            └─ run_recall_session()                  ── itanet.dll
```

Step by step:

1. **Camera capture** → 8 images land in `data/input_pictures/<suffix>/`.
2. **Difference build** → 8 difference images in `data/difference_pictures/<suffix>/`.
3. **Feature extraction** → `Mean`, `Std`, `Max`, `Mode` per image, then averaged across the 8.
4. **Temp CSV** → the 1-row feature vector is written to a system temp file.
5. **DLL call** → `predict_from_csv()`:
   - loads the CSV via `_load_recall_csv()` (drops the leading non-numeric `Image` column, ignores any `Grade` column, filters trailing summary rows);
   - writes `RecallData.dat` and `recall.fls`;
   - `chdir`s into `models/itanet/run/` and calls `run_recall_session(LOAD_FROM_FILE, komma_punkt, SHUFFLE_OFF)`.
6. **Read back** → `Recall_output.dat` is parsed. If any line contains `nan`/`-nan(ind)` the reader raises with a clear "diverged weights — restore from archive" message instead of silently returning empty. Otherwise predictions are rounded to half-steps and clipped to `[1.0, 5.0]`.
7. **Persist** → final record written to `output/grading_results/<sample>-<stage>-<trial>-analysis.csv`.
8. **(Optional) PDF** → `reports/<sample>-<operator>-report.pdf` built by sweeping the analysis CSVs for the sample.

Steps 4–6 produce temp files that are never user-facing. The persistent artifact is the analysis CSV from step 7.

### What's in the per-trial analysis CSV

```
Image,Mean,Std,Max,Mode
000-100-1-dif.png,11.37,8.70,103.0,1.0      ← per-image features (rows 1–8)
…
000-100-8-dif.png,11.54,8.97,99.0,1.0
Average,11.35,8.79,104.0,1.0                ← mean across the 8 image rows
Grade,4.0,,,                                ← the prediction
Backend,itanet_dll,,,                       ← which backend produced it
```

This file is **both the input audit trail and the final result**: per-image stats, the averaged vector the model actually saw, the predicted grade, and the backend label. The PDF exporter walks `output/grading_results/` for matching `{sample}-{stage}-{trial}-analysis.csv` files.

### Recall without Streamlit

Run recall on any analysis CSV directly:

```bash
python itanet_recall.py output/grading_results/<sample>-<stage>-<trial>-analysis.csv
```

Summary rows (`Average`, `Grade`, `Backend`) at the bottom are filtered out automatically. Predictions are written next to the input as `<name>_predictions.csv` unless `--output` overrides that.

CLI flags mirror the DLL contract: `--net-type` (default `2` = `LOAD_FROM_FILE`), `--decimal comma|point` (drives `komma_punkt`), `--shuffle` (rarely useful for recall), plus `--feature-cols` and `--no-clip`. Paths are fixed to `models/itanet/{run,data}/`; override in `app/settings/config.py` if you need a non-standard layout.

## Integration tests — `app/tests/test_integration.py`

Legacy CLI-style runner that requires the actual DLL + a trained `.NET`. Exercises three CSV-format edge cases (summary-row filtering, single-row temp CSV, whole-number formatting) that have bitten the integration in the past.

## Unit tests — `app/tests/test_itanet_unit.py`

Pytest suite that runs against a fake DLL path — no DLL, no `.NET` needed, ~2 s to run. Covers:

- **Enum drift**: pins `CREATE_NEW_NET=1`, `LOAD_FROM_FILE=2`, `KOMMA_ZU_PUNKT=0`, `PUNKT_ZU_KOMMA=1`, `SHUFFLE_ON=1`, `SHUFFLE_OFF=0`. Any HAUPTDEF.H renumbering fails here first.
- **Byte contract**: `RecallData.dat` / `TrainingData.dat` header format (`<N>\t<cols>\r\n`), CRLF endings, whole-number `.0`/`,0` suffix, no bare ints, no accidental comma leaks in `point` mode (or vice-versa).
- **Decimal threading**: `csv_to_training_dats(..., decimal="point")` really writes dots — regression for a bug where `_fmt_value` / `_fmt_grade` always wrote commas.
- **`.NET`-shape guard**: pristine and weights-bearing `.NET`s each rejected for the wrong `--net-type`; the 500-byte boundary is pinned.
- **Archive step**: `_archive_existing_net()` copies both `.NET` and `.TRN`; warns (but still archives the `.NET`) if `.TRN` is absent.
- **NaN detection**: `read_recall_output()` raises on any NaN row (not just all-NaN), with row-number context.
- **DLL guard**: `ITANetManager(...)` raises `FileNotFoundError` before any `.dat` file is written when the DLL path is bad.
- **API surface**: `predict_from_csv(..., seed=42)` raises `TypeError` — regression against the seed kwarg accidentally coming back.
- **Folder-split invariant**: `default_run_dir()` and `default_fls_dir()` are siblings, so the DLL's hardcoded `..\data\*.fls` lookup resolves.

Run:

```bash
python -m pytest app/tests/test_itanet_unit.py -v
```
