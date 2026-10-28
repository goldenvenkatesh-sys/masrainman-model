# Make today's MasRainMan 10-day forecast from the latest ECMWF run, with maps.
# Meant for Windows Task Scheduler (e.g. daily at 14:00 IST, after the 00 UTC run is out;
# and 02:00 IST for the 12 UTC run). Old ECMWF caches beyond -KeepDays are removed.
#   powershell -ExecutionPolicy Bypass -File scripts\live_daily.ps1
param(
    [string]$Ckpt = "D:\masrainman_runs\baseline\best.pt",
    [string]$Log = "D:\masrainman_runs\live\live.log",
    [int]$KeepDays = 10
)

Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
if (-not (Test-Path $Ckpt)) {
    Add-Content $Log "$(Get-Date -Format s) no checkpoint at $Ckpt - train the model first"
    exit 1
}

Add-Content $Log "=== $(Get-Date -Format s) live forecast"
python -u -m masrainman.live_forecast --ckpt $Ckpt --plot 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append
$code = $LASTEXITCODE
Add-Content $Log "=== $(Get-Date -Format s) finished with exit code $code"

$cache = "D:\masrainman_data\ecmwf"
if (Test-Path $cache) {
    Get-ChildItem $cache -Directory | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$KeepDays) } |
        Remove-Item -Recurse -Force
}
exit $code
