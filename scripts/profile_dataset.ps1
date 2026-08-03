$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    $ProjectPython = "python"
}

$Dataset = Join-Path $ProjectRoot "data\raw\app_reviews.parquet"
if (-not (Test-Path -LiteralPath $Dataset)) {
    throw "Download the Hugging Face parquet file to data\raw\app_reviews.parquet first."
}

Push-Location $ProjectRoot
try {
    @'
import pandas as pd

frame = pd.read_parquet("data/raw/app_reviews.parquet")
frame["parsed_date"] = pd.to_datetime(frame["date"], format="%B %d %Y", errors="coerce")
targets = {
    "WordPress": "org.wordpress.android",
    "PPSSPP": "org.ppsspp.ppsspp",
    "AntennaPod": "de.danoeh.antennapod",
}
print(f"all rows={len(frame):,}; packages={frame.package_name.nunique():,}")
for name, package in targets.items():
    rows = frame[frame.package_name.eq(package)]
    print(
        f"{name}: rows={len(rows):,}; unique_text={rows.review.nunique():,}; "
        f"window={rows.parsed_date.min().date()}..{rows.parsed_date.max().date()}"
    )
'@ | & $ProjectPython -
}
finally {
    Pop-Location
}

