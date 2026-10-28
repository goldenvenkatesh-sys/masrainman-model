# Run the ERA5 download until every file is there, retrying after errors.
# Runs in its own window, so it keeps going if the Claude app disconnects.
#   powershell -ExecutionPolicy Bypass -File scripts\download_era5.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\download_era5.ps1 -First 2018 -Last 2023
param([int]$First = 2018, [int]$Last = 2023, [string]$Log = "D:\masrainman_data\download.log")

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
$Host.UI.RawUI.WindowTitle = "MasRainMan ERA5 download $First-$Last"

while ($true) {
    Add-Content $Log "=== $(Get-Date -Format s) starting download $First-$Last"
    python -u -m masrainman.download_era5 --years $First $Last 2>&1 | ForEach-Object { "$_"; Add-Content -Path $Log -Value "$_" }
    if ($LASTEXITCODE -eq 0) {
        Add-Content $Log "=== $(Get-Date -Format s) all files downloaded"
        Write-Host "All ERA5 files downloaded. You can close this window." -ForegroundColor Green
        break
    }
    Add-Content $Log "=== $(Get-Date -Format s) stopped with an error; retrying in 5 minutes"
    Write-Host "Download stopped with an error; retrying in 5 minutes (Ctrl+C to quit)." -ForegroundColor Yellow
    Start-Sleep -Seconds 300
}
