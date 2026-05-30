param(
    [switch]$SkipValidation,
    [switch]$SkipTrain,
    [switch]$SkipBacktest,
    [switch]$SkipPermutation,
    [switch]$Fast,
    [int]$UniverseLimit = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

try {
    $env:PYTHONPATH = "src"

    function Invoke-Py([string]$Script, [string[]]$ExtraArgs = @()) {
        # Build an explicit argv list; nested PowerShell splatting can drop extra flags.
        $pyArgs = @("-3", $Script) + [string[]]$ExtraArgs
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py @pyArgs
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python @($Script) @ExtraArgs
        } else {
            throw "Python not found on PATH."
        }
        if ($LASTEXITCODE -ne 0) { throw "Command failed: $Script" }
    }

    Write-Host "=== BladeTrader research pipeline (live APIs) ===" -ForegroundColor Cyan

    $validateArgs = @()
    if ($Fast) {
        $validateArgs += @("--skip-news-backfill", "--skip-news-ingest", "--skip-eodhd-news")
        Write-Host "Fast mode: skip news ingest/backfill; earnings + prices only." -ForegroundColor DarkYellow
    }
    if ($UniverseLimit -gt 0) {
        $validateArgs += @("--universe-limit", $UniverseLimit)
    }

    if (-not $SkipValidation) {
        if (-not $Fast) {
            Write-Host "`n[0/4] Populating news store..." -ForegroundColor Yellow
            $popArgs = @()
            if ($UniverseLimit -gt 0) { $popArgs += @("--universe-limit", $UniverseLimit) }
            Invoke-Py "scripts/populate_news.py" @popArgs
        }
        Write-Host "`n[1/4] Validating earnings events..." -ForegroundColor Yellow
        Invoke-Py "scripts/validate_case4_earnings.py" @validateArgs
    } else {
        Write-Host "`n[1/3] Skipped validation" -ForegroundColor DarkGray
    }

    if (-not $SkipTrain) {
        Write-Host "`n[2/3] Training baseline vs sentiment model..." -ForegroundColor Yellow
        Invoke-Py "scripts/train_case4_prototype_model.py"
    } else {
        Write-Host "`n[2/3] Skipped training" -ForegroundColor DarkGray
    }

    if (-not $SkipBacktest) {
        Write-Host "`n[3/4] Running backtest..." -ForegroundColor Yellow
        Invoke-Py "scripts/backtest_case4_earnings.py"
    } else {
        Write-Host "`n[3/4] Skipped backtest" -ForegroundColor DarkGray
    }

    if (-not $SkipPermutation) {
        Write-Host "`n[4/4] Walk-forward permutation tests..." -ForegroundColor Yellow
        Invoke-Py "scripts/run_case4_permutation_test.py"
    } else {
        Write-Host "`n[4/4] Skipped permutation tests" -ForegroundColor DarkGray
    }

    Write-Host "`nDone. Outputs under data/" -ForegroundColor Green
} finally {
    Pop-Location
}
