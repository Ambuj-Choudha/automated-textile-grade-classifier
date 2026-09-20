# Activate the project venv in the current PowerShell session.
#
# Must be dot-sourced so the activation modifies the parent shell:
#
#     . scripts\activate.ps1
#
# (Running `scripts\activate.ps1` without the leading dot activates in a
# child scope that dies with the script, leaving your shell unchanged.)

$venvActivate = Join-Path $PSScriptRoot "..\env\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Error "venv not found. Run:  py -3.11 scripts\setup_env.py"
    return
}
. $venvActivate
