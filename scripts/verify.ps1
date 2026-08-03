$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    $ProjectPython = "python"
}

Push-Location $ProjectRoot
try {
    & $ProjectPython -m pytest
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $ProjectPython -m ruff check .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $ProjectPython -m compileall -q src app.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $ProjectPython -m src.run demo --out-dir out
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "All project checks passed."
}
finally {
    Pop-Location
}

