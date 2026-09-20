# ITA-Net Integration

`itanet.dll` is a small C library the app calls via `ctypes`. This document is the developer reference for how the Python code talks to the DLL. For the user workflow (train, grade, reset), see the [root README](../README.md).

## How the app calls the DLL

```
   feature CSV (Mean, Std, Max, Mode [+ Grade])
                       │
                       ▼
   ┌────────────────────────────────────────┐
   │  itanet_training.py  (training path)   │  ──►  updated Neuronalesnetz.NET
   │  itanet_recall.py    (recall path)     │  ──►  predicted grade(s)
   └────────────────────┬───────────────────┘
                        │   chdir into models/itanet/{grade}/run/
                        ▼
                   itanet.dll
                        │   reads ..\data\*.fls, then reads/writes
                        │   the files listed there (in cwd)
                        ▼
                .NET, .TRN, .dat files
```

**Constraint that shapes the layout:** the DLL hardcodes `..\data\training.fls` and `..\data\recall.fls` as its entry points. The caller has to `chdir` into a folder that has `..\data\` as a sibling. That's why each grade has both:

- `models/itanet/{grade}/run/` — cwd when the DLL runs; holds `.NET`, `.TRN`, `.dat` files
- `models/itanet/{grade}/data/` — sibling of `run/`; holds the two `.fls` filelists

The DLL itself lives at `models/itanet/common/itanet.dll` and is loaded by absolute path, so it doesn't need to be in the cwd.

## DLL contract

The DLL must export two functions, same signature:

```c
int run_training_session(int net_type, int komma_punkt, int shuffle);
int run_recall_session  (int net_type, int komma_punkt, int shuffle);
```

Return codes: `0` = success, `-1` = an invalid flag was passed. Failures inside the C core (bad file, bad data) are **not** propagated — same fire-and-forget contract as the CLI.

| Flag | Value | Meaning |
|---|---|---|
| `CREATE_NEW_NET` / `LOAD_FROM_FILE` | `1` / `2` | Fresh weights (needs an empty `.NET`) / load trained weights from `.NET` |
| `KOMMA_ZU_PUNKT` / `PUNKT_ZU_KOMMA` | `0` / `1` | `.dat` files use `,` / `.` as decimal separator |
| `SHUFFLE_OFF` / `SHUFFLE_ON` | `0` / `1` | Preserve / shuffle training patterns (training only) |

Enum values mirror `INCLUDE/HAUPTDEF.H` in the ITA-Net C source, and are pinned in [test_itanet_unit.py](../app/tests/test_itanet_unit.py) so drift breaks a test before it breaks the DLL. [`load_itanet_dll()`](../itanet_recall.py) binds argtypes at load time and caches the handle.

**Not thread-safe.** Process-wide state (cwd, `srand`, output files) means concurrent calls race. The app never calls the DLL concurrently.

**Not seeded.** `srand(time(NULL))` inside `new_net` and `shuffle_data` — successive runs are not reproducible.

## The `.dat` byte format

The DLL is byte-picky. [`csv_to_training_dats()`](../itanet_training.py) and [`ITANetFileWriter.write_data_file()`](../itanet_recall.py) produce:

- Tab-separated columns, CRLF (`\r\n`) line endings.
- Header line `<N>\t<ncols>\r\n` (row count, column count).
- Whole-number **features**: written with trailing `.0` / `,0` (`104` → `104.0` or `104,0`). Bare ints cause an access violation inside the DLL.
- Whole-number **grades**: written as bare ints (`4`, not `4.0`).
- Decimal separator matches the `komma_punkt` flag exactly. Mixing them silently produces mis-scaled inputs.

The two `.fls` filelists (`recall.fls` — 5 lines, `training.fls` — 4 lines) and both `.dat` files are rewritten on every call, so hand-edits are lost.

## Why the training data ships with the weights

The DLL normalizes recall inputs against the min/max of the original training features (in `WANDEL.C`). Shipping a trained `.NET` without its paired `TrainingData.dat` / `TrainingGrades.dat` silently produces wrong predictions. Keep the three files as a set. (The archive step currently only snapshots `.NET` + `.TRN`, not the paired `.dat` files — a known limitation.)
