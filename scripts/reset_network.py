"""Reset a grade's network files back to an empty (untrained) state, or to
an earlier saved copy.

Use this when a training run gives bad results, or when you want to start
training over from scratch.

Two sources:

  * Git HEAD (default) — copies back the empty network files that ship
    with the repo.
  * A saved copy under `models/itanet/archive/<grade>/<UTC>/` — for going
    back to an earlier trained network.

Examples:

    python scripts/reset_network.py --grade pilling
    python scripts/reset_network.py --all
    python scripts/reset_network.py --grade pilling --list
    python scripts/reset_network.py --grade pilling --from 20260901T120000Z
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.settings import config as cfg  # noqa: E402
from itanet_training import PRISTINE_NET_MAX_BYTES  # noqa: E402

_NET = "Neuronalesnetz.NET"
_TRN = "Neuronalesnetz.TRN"


def _run_dir(grade: str) -> Path:
    return Path(cfg.get_itanet_run_dir(grade))


def _archive_dir(grade: str) -> Path:
    return Path(cfg.get_itanet_archive_dir(grade))


def _git_show(path: Path) -> bytes:
    rel = path.resolve().relative_to(ROOT).as_posix()
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"], cwd=ROOT, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"git show HEAD:{rel} failed (is the file committed?):\n  {e.stderr.decode(errors='replace').strip()}"
        ) from None


def _reset_from_git(grade: str) -> None:
    run = _run_dir(grade)
    net_path = run / _NET
    trn_path = run / _TRN

    print(f"[{grade}] fetching empty network files from git HEAD ...")
    net_bytes = _git_show(net_path)
    trn_bytes = _git_show(trn_path)

    if len(net_bytes) > PRISTINE_NET_MAX_BYTES:
        sys.exit(
            f"[{grade}] STOPPED: the file in git is {len(net_bytes)} bytes "
            f"(> {PRISTINE_NET_MAX_BYTES}). It looks like a trained network, "
            f"not an empty one. Please restore manually from a saved copy."
        )

    run.mkdir(parents=True, exist_ok=True)
    net_path.write_bytes(net_bytes)
    trn_path.write_bytes(trn_bytes)
    print(f"[{grade}] reset done: {net_path} ({len(net_bytes)} bytes), {trn_path} ({len(trn_bytes)} bytes)")


def _list_archives(grade: str) -> list[Path]:
    ad = _archive_dir(grade)
    if not ad.exists():
        return []
    return sorted(p for p in ad.iterdir() if p.is_dir())


def _print_archives(grade: str) -> None:
    archives = _list_archives(grade)
    if not archives:
        print(f"[{grade}] no saved copies under {_archive_dir(grade)}")
        return
    print(f"[{grade}] saved copies in {_archive_dir(grade)}:")
    for a in archives:
        net = a / _NET
        size = net.stat().st_size if net.exists() else 0
        tag = "empty" if 0 < size <= PRISTINE_NET_MAX_BYTES else "trained"
        print(f"  {a.name:24s}  .NET={size:>7d} bytes  ({tag})")


def _reset_from_archive(grade: str, stamp: str) -> None:
    src_dir = _archive_dir(grade) / stamp
    if not src_dir.is_dir():
        sys.exit(f"[{grade}] saved copy not found: {src_dir}")
    src_net = src_dir / _NET
    src_trn = src_dir / _TRN
    if not src_net.exists():
        sys.exit(f"[{grade}] saved copy is missing {_NET}: {src_net}")

    run = _run_dir(grade)
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_net, run / _NET)
    if src_trn.exists():
        shutil.copy2(src_trn, run / _TRN)
        print(f"[{grade}] restored .NET + .TRN from {src_dir}")
    else:
        print(f"[{grade}] restored .NET from {src_dir} (no .TRN in the saved copy)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--grade", nargs="+", choices=list(cfg.GRADES),
                        help="Grade(s) to reset.")
    target.add_argument("--all", action="store_true",
                        help=f"Reset every grade ({', '.join(cfg.GRADES)}).")
    parser.add_argument("--from", dest="from_archive", metavar="TIMESTAMP",
                        help="Restore from a saved copy (archive/<grade>/<TIMESTAMP>/) "
                             "instead of resetting to empty.")
    parser.add_argument("--list", dest="list_archives", action="store_true",
                        help="List saved copies for the chosen grade(s) and exit.")
    args = parser.parse_args()

    grades = list(cfg.GRADES) if args.all else args.grade

    if args.list_archives:
        for g in grades:
            _print_archives(g)
        return

    if args.from_archive:
        if len(grades) != 1:
            sys.exit("--from is per-grade; pass exactly one --grade.")
        _reset_from_archive(grades[0], args.from_archive)
        return

    for g in grades:
        _reset_from_git(g)


if __name__ == "__main__":
    main()
