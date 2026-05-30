param(
    [Parameter(Mandatory = $true)]
    [string]$Eodhd,
    [Parameter(Mandatory = $true)]
    [string]$Gnews
)

$ErrorActionPreference = "Stop"
$envPath = Join-Path (Split-Path -Parent $PSScriptRoot) ".env"

if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path (Split-Path -Parent $PSScriptRoot) ".env.example") $envPath
}

$content = Get-Content $envPath -Raw
$content = $content -replace '(?m)^EODHD_API_KEY=.*$', "EODHD_API_KEY=$Eodhd"
$content = $content -replace '(?m)^GNEWS_API_KEY=.*$', "GNEWS_API_KEY=$Gnews"
$content = $content -replace '(?m)^MARKET_DATA_PROVIDER=.*$', "MARKET_DATA_PROVIDER=eodhd"
Set-Content -Path $envPath -Value $content.TrimEnd() -NoNewline
Add-Content -Path $envPath -Value ""

Write-Host "Updated $envPath (EODHD + GNEWS keys set, provider=eodhd)."
