# Automated Textile Grade Classifier

A tool for grading textile samples. A camera takes photos of the sample, compares them to a clean reference photo, and predicts three grades — **pilling**, **matting**, and **fuzzing** (each from 1.0 to 5.0 in 0.5 steps) — using one trained neural network (ITA-Net) per grade.

## Who is this document for?

This project has three kinds of users. Follow the path that matches you.

| Role | What you do | Where to start |
|---|---|---|
| **Operator** | Use the app to take photos and read grades. No setup on your side. | Watch the walkthrough video: _[TODO: add link]_ |
| **Model owner** | Set up the machine, train the networks, check the results, then hand the app to the operator. | **This README, steps 1 – 7 below.** |
| **Developer** | Add features or change how the app works. | [docs/APP.md](docs/APP.md) for the app, [docs/INTEGRATION.md](docs/INTEGRATION.md) for the DLL contract. |

The steps below are for the **model owner**.

---

## 1. First-time environment setup

One command, hands-off:

```powershell
.\scripts\setup_env.ps1     # Windows (PowerShell)
```

```bash
./scripts/setup_env.sh      # Linux / macOS
```

Both scripts check for Python 3.11 → install it if missing (winget on Windows; apt / dnf / pacman / brew on Linux/macOS, sudo may be asked) → create `env\` → install [`requirements.txt`](requirements.txt). Safe to run again — it will skip steps that are already done.

## 2. Activate the environment (every new terminal)

```powershell
. scripts\activate.ps1        # PowerShell — the dot at the start matters
```

```bash
source scripts/activate.sh    # Git Bash / macOS / Linux
```

The dot (or `source`) is needed. Without it, your terminal will not stay activated.

## 3. Collect labeled training samples

Start the app to take photos of samples with known grades (skip this if you already have the CSV files):

```powershell
python -m streamlit run app.py
```

In the app: pick **Training Mode**, log in (default password `1234`), fill in the sample details and the known grade(s), then click **Capture**. Each capture takes 8 photos and writes numbers into two CSV files per grade in `data/training_features/`:

- `{grade}_per_image_features.csv` — one line per photo (8 lines per sample)
- `{grade}_averaged_features.csv` — one line per sample (**use this file for training**)

Repeat with as many samples as you can. More samples → better predictions.

## 4. Set up the training

Open [`scripts/training_config.toml`](scripts/training_config.toml). List one section per grade you want to train, so the CSV file, the grade, and the settings are written in one place — no mistakes at the command line:

```toml
[pilling]
csv = "data/training_features/pilling_averaged_features.csv"
net_type = 1        # 1 = start fresh (empty network), 2 = keep training the current one
decimal = "comma"
shuffle = 1
```

Delete or comment out a grade's section to leave that network alone.

## 5. Train

Train every grade in the config file:

```powershell
python scripts\train.py
```

Or one or two grades at a time:

```powershell
python scripts\train.py --grade pilling
python scripts\train.py --grade pilling matting
```

The script checks every grade first. If a CSV file is missing, or the network is not in the expected state (empty for `net_type=1`, trained for `net_type=2`), it stops before starting. Before writing new weights, the old network is copied into `models\itanet\archive\<grade>\<UTC>\`, so you can always go back.

## 6. Check the result

Test a trained network without the app:

```powershell
python itanet_recall.py output\grading_results\<sample>-<stage>-<trial>-<grade>-analysis.csv --grade <grade>
```

Or run the full test (needs a trained network for the chosen grade):

```powershell
python app\tests\test_integration.py all --grade pilling
```

## 7. Fix a bad training run

If the network gives bad results, or you want to start training over:

```powershell
python scripts\reset_network.py --grade pilling      # or --all
```

This puts the empty (untrained) network files back in place. To go back to an earlier trained network instead:

```powershell
python scripts\reset_network.py --grade pilling --list
python scripts\reset_network.py --grade pilling --from 20260901T120000Z
```

---

## Giving the app to the operator

When every grade is trained and checked:

1. Change these two lines in [`app/settings/config.py`](app/settings/config.py):
   ```python
   MOCK_HARDWARE = False       # use the real camera and motor
   MOCK_PREDICTION = False     # use the trained network (not fake grades)
   ```
2. Change `ADMIN_PASSWORD` if you want.
3. Start the app: `python -m streamlit run app.py`. In **Grading Mode**, only trained grades will be selectable.

The operator can export a PDF report from the app (saved in `reports/`). Raw per-trial data is saved in `output/grading_results/`.

## Folder layout

These folders are created when the app runs and are not committed to git.

| Folder | What is in it |
|---|---|
| `data/input_pictures/{for_grading,for_training}/` | Photos taken by the camera |
| `data/difference_pictures/{for_grading,for_training}/` | Difference images (photo vs. the clean reference) |
| `data/training_features/` | Feature CSV files per grade (used as input for training) |
| `output/grading_results/` | Per-trial prediction CSV files |
| `reports/` | PDF reports exported from the app |
| `models/itanet/common/itanet.dll` | The shared DLL (used by all three grades) |
| `models/itanet/{grade}/run/` | The `.NET`, `.TRN`, and data files for one grade |
| `models/itanet/{grade}/data/` | The `recall.fls` and `training.fls` file lists |
| `models/itanet/archive/{grade}/` | Old `.NET` + `.TRN` copies from before each training run |
| `models/tf/` | TensorFlow model (a different prediction backend) |

## List of scripts

| Script | What it does |
|---|---|
| [`scripts/setup_env.ps1`](scripts/setup_env.ps1) / [`.sh`](scripts/setup_env.sh) | Install Python 3.11 (if missing), create `env\`, install `requirements.txt`. Run once per machine. |
| [`scripts/activate.ps1`](scripts/activate.ps1) / [`.sh`](scripts/activate.sh) | Activate the environment in the current terminal. |
| [`scripts/training_config.toml`](scripts/training_config.toml) | Training settings for each grade. Edit before training. |
| [`scripts/train.py`](scripts/train.py) | Train one, two, or all grades using the settings file. |
| [`scripts/reset_network.py`](scripts/reset_network.py) | Reset a grade's network back to empty, or to an earlier saved copy. |

## More reading

- **How the app is wired together** → [docs/APP.md](docs/APP.md)
- **How the app talks to the ITA-Net DLL** → [docs/INTEGRATION.md](docs/INTEGRATION.md)

Questions or problems: <https://github.com/Ambuj-Choudha/automated-textile-grade-classifier/issues>
