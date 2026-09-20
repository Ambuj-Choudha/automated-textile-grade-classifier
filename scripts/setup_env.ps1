# Hands-off first-time setup for Windows.
#
# Ensures Python 3.11 is installed (via winget if missing), then creates
# env\ and installs requirements.txt into it.
#
# Run once per machine, from the repo root:
#
#     .\scripts\setup_env.ps1
#
# Safe to re-run — every step is idempotent.

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir  = Join-Path $RepoRoot "env"
$ReqFile  = Join-Path $RepoRoot "requirements.txt"

function Find-Py311 {
    # Return @{Cmd=...; Args=...} for the first Python 3.11 we find, else $null.
    $candidates = @(
        @{ Cmd = "py";         Args = @("-3.11") },
        @{ Cmd = "python3.11"; Args = @() },
        @{ Cmd = "python";     Args = @() }
    )
    foreach ($c in $candidates) {
        if (-not (Get-Command $c.Cmd -ErrorAction SilentlyContinue)) { continue }
        try {
            $out = & $c.Cmd @($c.Args + "--version") 2>&1
            if ($LASTEXITCODE -eq 0 -and $out -match "Python 3\.11\.") {
                return $c
            }
        } catch { }
    }
    return $null
}

# --- 1. Python 3.11 -----------------------------------------------------------
$py = Find-Py311
if ($py) {
    Write-Host "[setup] Python 3.11 already available: $($py.Cmd) $($py.Args -join ' ')"
} else {
    Write-Host "[setup] Python 3.11 not found. Installing via winget ..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Error @"
winget is not available on this machine.
Install Python 3.11 manually from
  https://www.python.org/downloads/release/python-3110/
(tick 'Add Python to PATH'), then re-run .\scripts\setup_env.ps1.
"@
        exit 1
    }
    winget install --id Python.Python.3.11 -e `
        --accept-source-agreements --accept-package-agreements

    # winget updates machine PATH but not this process's PATH — refresh it
    # so the newly-installed python.exe resolves before Find-Py311 re-runs.
    $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")

    $py = Find-Py311
    if (-not $py) {
        Write-Error "Python 3.11 install did not register on PATH. Open a new shell and re-run this script."
        exit 1
    }
    Write-Host "[setup] Python 3.11 installed."
}

# --- 2. Venv ------------------------------------------------------------------
if (-not (Test-Path $ReqFile)) {
    Write-Error "$ReqFile is missing."
    exit 1
}

if (Test-Path $VenvDir) {
    Write-Host "[setup] Reusing existing venv at $VenvDir"
} else {
    Write-Host "[setup] Creating venv at $VenvDir"
    & $py.Cmd @($py.Args + @("-m", "venv", $VenvDir))
    if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed."; exit $LASTEXITCODE }
}

# --- 3. Install requirements --------------------------------------------------
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# `python -m pip` (not pip.exe) — a running pip.exe can't overwrite itself.
Write-Host "[setup] Upgrading pip"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed."; exit $LASTEXITCODE }

Write-Host "[setup] Installing requirements.txt"
& $VenvPython -m pip install -r $ReqFile
if ($LASTEXITCODE -ne 0) { Write-Error "requirements install failed."; exit $LASTEXITCODE }

Write-Host ""
Write-Host "[setup] Done. Activate the venv in this shell:"
Write-Host "  . scripts\activate.ps1"
