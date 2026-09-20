#!/usr/bin/env bash
# Hands-off first-time setup for Linux / macOS.
#
# Ensures Python 3.11 is installed (via apt/dnf/pacman/brew, sudo may be
# prompted), then creates env/ and installs requirements.txt into it.
#
# Run once per machine, from the repo root:
#
#     ./scripts/setup_env.sh
#
# Safe to re-run — every step is idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/env"
REQ_FILE="$REPO_ROOT/requirements.txt"

find_py311() {
    for cmd in python3.11 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            if "$cmd" --version 2>&1 | grep -q "Python 3\.11\."; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

install_python311() {
    case "$(uname -s)" in
        Darwin)
            if ! command -v brew >/dev/null 2>&1; then
                echo "ERROR: Homebrew not found. Install from https://brew.sh/ then re-run." >&2
                exit 1
            fi
            echo "[setup] Installing Python 3.11 via Homebrew ..."
            brew install python@3.11
            local prefix
            prefix="$(brew --prefix python@3.11 2>/dev/null || true)"
            if [ -n "$prefix" ] && [ -d "$prefix/bin" ]; then
                export PATH="$prefix/bin:$PATH"
            fi
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                echo "[setup] Installing Python 3.11 via apt-get (sudo required) ..."
                sudo apt-get update
                sudo apt-get install -y python3.11 python3.11-venv
            elif command -v dnf >/dev/null 2>&1; then
                echo "[setup] Installing Python 3.11 via dnf (sudo required) ..."
                sudo dnf install -y python3.11
            elif command -v pacman >/dev/null 2>&1; then
                echo "[setup] Installing Python via pacman (sudo required) ..."
                sudo pacman -Sy --noconfirm python
            else
                echo "ERROR: No supported package manager (apt/dnf/pacman) found." >&2
                echo "       Install Python 3.11 manually then re-run this script." >&2
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unsupported OS ($(uname -s)). Install Python 3.11 manually." >&2
            exit 1
            ;;
    esac
}

# --- 1. Python 3.11 -----------------------------------------------------------
if PY311=$(find_py311); then
    echo "[setup] Python 3.11 already available: $PY311"
else
    install_python311
    if ! PY311=$(find_py311); then
        echo "ERROR: Python 3.11 install did not register on PATH. Open a new shell and re-run." >&2
        exit 1
    fi
    echo "[setup] Python 3.11 installed: $PY311"
fi

# --- 2. Venv ------------------------------------------------------------------
if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: $REQ_FILE missing." >&2
    exit 1
fi

if [ -d "$VENV_DIR" ]; then
    echo "[setup] Reusing existing venv at $VENV_DIR"
else
    echo "[setup] Creating venv at $VENV_DIR"
    "$PY311" -m venv "$VENV_DIR"
fi

# --- 3. Install requirements --------------------------------------------------
VENV_PYTHON="$VENV_DIR/bin/python"

# python -m pip (not pip directly) to be safe against self-upgrade issues.
echo "[setup] Upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip

echo "[setup] Installing requirements.txt"
"$VENV_PYTHON" -m pip install -r "$REQ_FILE"

echo
echo "[setup] Done. Activate the venv in this shell:"
echo "  source scripts/activate.sh"
