"""Train ITA-Net for one or more grades from an explicit config file.

Reads `scripts/training_config.toml` and dispatches to
`itanet_training.train_from_csv` per grade. All grades are validated up
front — the run aborts before any DLL call if any CSV is missing or any
.NET is in the wrong state for the requested `net_type`.

    python scripts/train.py                          # every grade in the config
    python scripts/train.py --grade pilling          # just one
    python scripts/train.py --grade pilling matting  # two

The config file exists (rather than positional CLI args) so the mapping
CSV <-> grade <-> net_type is version-controlled and can't drift between
runs.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.settings import config as cfg  # noqa: E402
from itanet_training import PRISTINE_NET_MAX_BYTES, train_from_csv  # noqa: E402

CONFIG_PATH = ROOT / "scripts" / "training_config.toml"


def _load_config() -> dict[str, dict[str, Any]]:
    if not CONFIG_PATH.exists():
        sys.exit(f"ERROR: {CONFIG_PATH} not found.")
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _validate_grade(grade: str, section: dict[str, Any]) -> Path:
    """Return the resolved CSV path if this grade is ready to train; exit on error."""
    csv_rel = section.get("csv")
    if not csv_rel:
        sys.exit(f"ERROR [{grade}]: no 'csv' key in training_config.toml.")
    csv = (ROOT / csv_rel).resolve()
    if not csv.exists():
        sys.exit(f"ERROR [{grade}]: CSV not found: {csv}")

    net = Path(cfg.get_itanet_run_dir(grade)) / "Neuronalesnetz.NET"
    if not net.exists():
        sys.exit(f"ERROR [{grade}]: .NET not found: {net}")

    net_type = int(section.get("net_type", 1))
    is_trained = net.stat().st_size > PRISTINE_NET_MAX_BYTES

    if net_type == 1 and is_trained:
        sys.exit(
            f"ERROR [{grade}]: net_type=1 (fresh start) needs an empty network, "
            f"but {net} is {net.stat().st_size} bytes (already trained).\n"
            f"  - reset to empty:   python scripts/reset_network.py --grade {grade}\n"
            f"  - or set net_type=2 in training_config.toml to keep training this network."
        )
    if net_type == 2 and not is_trained:
        sys.exit(
            f"ERROR [{grade}]: net_type=2 continues training, but {net} is "
            f"{net.stat().st_size} bytes (looks empty). Use net_type=1 to start fresh."
        )
    return csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--grade", nargs="+", choices=list(cfg.GRADES),
        help="Grade(s) to train. Default: every grade with a section in training_config.toml.",
    )
    args = parser.parse_args()

    config = _load_config()
    grades = args.grade or [g for g in cfg.GRADES if g in config]
    if not grades:
        sys.exit("ERROR: No grades to train. Add sections to training_config.toml or pass --grade.")

    plans: list[tuple[str, dict[str, Any], Path]] = []
    for grade in grades:
        if grade not in config:
            sys.exit(f"ERROR: grade '{grade}' has no [{grade}] section in training_config.toml.")
        plans.append((grade, config[grade], _validate_grade(grade, config[grade])))

    print("Plan:")
    for grade, section, csv in plans:
        print(f"  {grade:8s}  csv={csv.name}  "
              f"net_type={section.get('net_type', 1)}  "
              f"decimal={section.get('decimal', 'comma')}  "
              f"shuffle={section.get('shuffle', 1)}")
    print()

    for grade, section, csv in plans:
        print(f"[train] === {grade} ===")
        try:
            train_from_csv(
                csv_path=str(csv),
                grade=grade,
                net_type=int(section.get("net_type", 1)),
                decimal=str(section.get("decimal", "comma")),
                shuffle=int(section.get("shuffle", 1)),
                backup=True,
            )
        except Exception as e:
            sys.exit(f"[train] {grade} FAILED: {e}")
        print(f"[train] {grade} DONE\n")

    print(f"All requested grades trained: {', '.join(g for g, _, _ in plans)}")


if __name__ == "__main__":
    main()
