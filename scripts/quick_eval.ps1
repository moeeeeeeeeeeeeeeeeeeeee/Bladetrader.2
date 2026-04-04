param(
    [string]$SnapshotPath = "data/case4_dataset_snapshot.jsonl"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python not found at $python"
}

if (-not (Test-Path (Join-Path $root $SnapshotPath))) {
    Write-Warning "Snapshot not found at $SnapshotPath. Build dataset first if needed."
}

Push-Location $root
try {
    $env:PYTHONPATH = "src"
    & $python "scripts/validate_case4_earnings.py" --offline-only --snapshot-path $SnapshotPath
} finally {
    Pop-Location
}

