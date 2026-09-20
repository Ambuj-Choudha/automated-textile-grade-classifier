# Activate the project venv in the current bash/zsh session.
#
# Must be sourced (not executed) so activation modifies the parent shell:
#
#     source scripts/activate.sh
#
# The venv layout differs between Windows-under-Git-Bash (env/Scripts/) and
# Linux/macOS (env/bin/), so we probe both.

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_venv_root="$_script_dir/../env"

if [ -f "$_venv_root/Scripts/activate" ]; then
    # shellcheck source=/dev/null
    . "$_venv_root/Scripts/activate"
elif [ -f "$_venv_root/bin/activate" ]; then
    # shellcheck source=/dev/null
    . "$_venv_root/bin/activate"
else
    echo "venv not found. Run:  python3.11 scripts/setup_env.py" >&2
    return 1 2>/dev/null || exit 1
fi

unset _script_dir _venv_root
