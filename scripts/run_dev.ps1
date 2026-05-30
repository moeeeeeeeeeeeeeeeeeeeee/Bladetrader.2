$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$port = if ($env:FINHACK_PORT) { [int]$env:FINHACK_PORT } else { 8080 }

Push-Location $root
try {
    $blocked8000 = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($blocked8000) {
        $owner = Get-Process -Id $blocked8000[0].OwningProcess -ErrorAction SilentlyContinue
        $name = if ($owner) { $owner.ProcessName } else { "unknown" }
        Write-Host "Port 8000 is in use by $name (PID $($blocked8000[0].OwningProcess)); using port $port instead."
    }

    Write-Host "Starting BladeTrader at http://127.0.0.1:$port/"
    & py -m uvicorn finhack.api:app --app-dir src --reload --host 127.0.0.1 --port $port
} finally {
    Pop-Location
}
