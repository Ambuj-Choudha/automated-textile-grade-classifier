# Hands-off first-time setup for Windows.
#
# Ensures a compatible Python (3.11 or newer) is installed (via winget if
# missing), then creates env\ and installs requirements.txt into it.
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

$MinMinor = 11   # accept Python 3.$MinMinor or newer (3.12, 3.13, ...)

function Find-Python {
    # Return @{Cmd=...; Args=...} for the first Python >= 3.$MinMinor we find, else $null.
    #
    # Search order:
    #   1. py launcher (prefer 3.11 explicitly, then latest 3.x)
    #   2. python3.11 / python3 shims
    #   3. Known install paths under AppData and ProgramFiles
    #   4. plain `python` (current shell default)
    $directPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "C:\Python311\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Program Files (x86)\Python311\python.exe"
    )
    $candidates = @(
        @{ Cmd = "py";          Args = @("-3.11") },
        @{ Cmd = "py";          Args = @("-3") },
        @{ Cmd = "python3.11";  Args = @() },
        @{ Cmd = "python3";     Args = @() }
    )
    foreach ($p in $directPaths) {
        if (Test-Path $p) { $candidates += @{ Cmd = $p; Args = @() } }
    }
    $candidates += @{ Cmd = "python"; Args = @() }
    foreach ($c in $candidates) {
        if (-not (Get-Command $c.Cmd -ErrorAction SilentlyContinue)) { continue }
        try {
            $out = & $c.Cmd @($c.Args + "--version") 2>&1
            if ($LASTEXITCODE -eq 0 -and $out -match "Python 3\.(\d+)\.") {
                if ([int]$Matches[1] -ge $MinMinor) { return $c }
            }
        } catch { }
    }
    return $null
}

# --- 1. Python ----------------------------------------------------------------
$py = Find-Python
if ($py) {
    Write-Host "[setup] Compatible Python already available: $($py.Cmd) $($py.Args -join ' ')"
} else {
    Write-Host "[setup] No Python >= 3.$MinMinor found. Installing 3.11 via winget ..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Error @"
winget is not available on this machine.
Install Python 3.11 (or newer) manually from
  https://www.python.org/downloads/
(tick 'Add Python to PATH'), then re-run .\scripts\setup_env.ps1.
"@
        exit 1
    }
    winget install --id Python.Python.3.11 -e `
        --accept-source-agreements --accept-package-agreements

    # winget updates machine PATH but not this process's PATH — refresh it
    # so the newly-installed python.exe resolves before Find-Python re-runs.
    $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")

    $py = Find-Python
    if (-not $py) {
        Write-Error "Python install did not register on PATH. Open a new shell and re-run this script."
        exit 1
    }
    Write-Host "[setup] Python installed."
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
